#!/usr/bin/env python3
"""Run deterministic generation + oracle expert jobs from a coverage manifest."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from audit_deltasg_outputs import check_run


MODEL = os.environ.get("DELTASG_LLM_MODEL", "qwen3.8-max")
_ACTIVE_PROCESS = None


def _terminate_active_process():
    global _ACTIVE_PROCESS
    process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def _signal_exit(signum, _frame):
    _terminate_active_process()
    raise SystemExit(128 + signum)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run(command, log_path, timeout, env):
    global _ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_token = f"{os.getpid()}-{time.time_ns()}"
    process_env = env.copy()
    process_env["DELTASG_RUN_TOKEN"] = run_token
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%F %T')} {' '.join(command)} =====\n")
        log.flush()
        process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=process_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _ACTIVE_PROCESS = process
        acquired_at = None
        try:
            while process.poll() is None:
                if acquired_at is None:
                    try:
                        recent = log_path.read_text(encoding="utf-8", errors="replace")[-131072:]
                    except OSError:
                        recent = ""
                    if f"run_token={run_token}" in recent:
                        acquired_at = time.time()
                if acquired_at is not None and time.time() - acquired_at > timeout:
                    break
                time.sleep(5)
            if process.poll() is not None:
                return process.returncode, time.time() - started
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(f"\nTIMEOUT after {timeout}s\n")
            return 124, time.time() - started
        except ProcessLookupError:
            return process.wait(), time.time() - started
        finally:
            _ACTIVE_PROCESS = None


def _identity_errors(job, run, expected_solvability_profile=None):
    te = run.get("task_environment") or {}
    task = te.get("task") or {}
    errors = []
    generation_profile = (te.get("generation") or {}).get("solvability_profile")
    if (
        expected_solvability_profile is not None
        and generation_profile != expected_solvability_profile
    ):
        errors.append("generation_solvability_profile_mismatch")
    if task.get("primary_behavior_task") != job["task"]:
        errors.append("task_identity_mismatch")
    if job.get("target_asset_model"):
        targets = [
            item for item in te.get("added_objects") or []
            if "task_object" in set(item.get("semantic_roles") or [])
        ]
        if not any(
            item.get("category") == job["target_asset_category"]
            and item.get("model") == job["target_asset_model"]
            for item in targets
        ):
            errors.append("spawned_asset_identity_mismatch")
    if job.get("target_native_object_id"):
        targets = {
            step.get("target_object")
            for step in te.get("solution_plan") or []
            if str(step.get("primitive") or "").upper() == "INTERACT"
        }
        if targets != {job["target_native_object_id"]}:
            errors.append("native_object_identity_mismatch")
    return errors


def _accepted_expert(path, backend=None, robot=None):
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    steps = result.get("steps") or []
    native_targets = {
        (step.get("step") or {}).get("target_object")
        for step in steps
        if (step.get("step") or {}).get("primitive")
        in {"OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}
    }
    replayed_targets = {
        state.get("object_id")
        for state in result.get("replayed_initial_states") or []
    }
    manipulation_ok = all(
        ((step.get("manipulation_height") or {}).get("eligible") is True)
        for step in steps
        if (step.get("step") or {}).get("primitive")
        in {"GRASP", "PLACE_ON_TOP", "PLACE_INSIDE", "OPEN", "CLOSE", "TOGGLE_ON", "TOGGLE_OFF"}
    )
    result_backend = result.get("backend") or {}
    generation_profile = result_backend.get("generation_solvability_profile")
    if generation_profile is None:
        input_path = Path(str(result.get("input") or ""))
        if input_path.is_file():
            try:
                source = json.loads(input_path.read_text(encoding="utf-8"))
                generation_profile = (
                    ((source.get("task_environment") or {}).get("generation") or {})
                    .get("solvability_profile")
                )
            except (OSError, json.JSONDecodeError):
                pass
    profile_matches = (
        backend is None
        or generation_profile == backend
        or (backend == "oracle_symbolic" and generation_profile is None)
    )
    physical_trajectory_matches = (
        backend != "physical_control"
        or (
            result_backend.get("physical_trajectory_available") is True
            and int(result_backend.get("physical_action_count") or 0) > 0
            and int(result_backend.get("physical_nonzero_action_count") or 0) > 0
        )
    )
    return (
        result.get("accepted") is True
        and (backend is None or result_backend.get("name") == backend)
        and profile_matches
        and physical_trajectory_matches
        and (robot is None or result.get("robot") == robot)
        and (result.get("scene_integrity") or {}).get("ok") is True
        and all(step.get("postcondition_ok") is True for step in steps)
        and manipulation_ok
        and native_targets <= replayed_targets
    )


def _expert_acceptance_kind(path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("accepted") is not True:
        return None
    backend = result.get("backend") or {}
    if backend.get("name") != "physical_control":
        return backend.get("name")
    physical_trajectory = (
        backend.get("physical_trajectory_available") is True
        and int(backend.get("physical_action_count") or 0) > 0
        and int(backend.get("physical_nonzero_action_count") or 0) > 0
    )
    if backend.get("assisted_interaction") is True and physical_trajectory:
        return "assisted_physical"
    if (
        physical_trajectory
        and
        backend.get("generation_profile_verified") is True
        and backend.get("physical_solubility_validation") is True
        and backend.get("low_level_vla_actions_eligible") is True
        and backend.get("complete_action_trace") is True
    ):
        return "strict_physical"
    return "physical_not_vla_eligible"


def _expert_result_state(path, backend=None, robot=None):
    """Classify a result without hiding deterministic expert rejections."""
    if not path.exists():
        return "missing"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if _accepted_expert(path, backend, robot):
        return "accepted"
    if result.get("error") or result.get("traceback"):
        return "process_error"
    return "rejected"


def _coverage_status_for_expert_state(state):
    if state == "accepted":
        return "accepted"
    if state == "rejected":
        return "expert_rejected"
    return "expert_process_error"


def _find_generated_run(directory):
    for path in sorted(directory.glob("online_env_a_*.json"), reverse=True):
        if "rejected" in path.name:
            continue
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("ok") is True:
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _find_rejected_run(directory):
    for path in sorted(directory.glob("online_env_a_rejected_*.json"), reverse=True):
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def _scene_initialization_failure(directory):
    path = directory / "error.json"
    if not path.exists():
        return None
    try:
        error = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if error.get("error_type") != "RobotSpawnError":
        return None
    return {
        "stage": "robot_spawn",
        "reason": error.get("reason"),
        "detail": error.get("detail") or {},
    }


def _native_ineligibility(job, rejected):
    if not job.get("target_native_object_id") or not rejected:
        return None
    validation = rejected.get("validation") or {}
    llm_validation = validation.get("llm_validation") or {}
    issues = set(llm_validation.get("issues") or [])
    detail = llm_validation.get("detail") or {}
    if "native_target_initial_state_failed" in issues:
        return {"stage": "initial_state", "detail": detail}
    if "no_matching_native_stateful_target" not in issues:
        return None
    if detail.get("stage") not in {
        "identity", "state_capability", "room", "pose",
        "manipulation_height", "robot_approach",
    }:
        return None
    return detail


def _native_attempt_schedule(job, process_retries):
    """Condition robot spawn on each versioned native target candidate."""
    exact = job.get("target_native_object_id")
    if exact:
        return [exact] * process_retries
    candidates = list(dict.fromkeys(job.get("target_native_candidates") or []))
    if not candidates:
        return [None] * process_retries
    return candidates


def _selected_native_target(run):
    targets = {
        step.get("target_object")
        for step in ((run.get("task_environment") or {}).get("solution_plan") or [])
        if str(step.get("primitive") or "").upper() == "INTERACT"
        and step.get("target_object")
    }
    return next(iter(targets)) if len(targets) == 1 else None


def _all_native_candidates_resolved(native_candidates, ineligibility_reasons):
    """Require explicit evidence for every candidate before calling a job ineligible."""
    if not native_candidates:
        return False
    resolved = {
        reason.get("object_id")
        for reason in ineligibility_reasons
        if reason.get("object_id")
    }
    return set(native_candidates) <= resolved


def _saved_fingerprint(run):
    te = run.get("task_environment") or {}
    return (
        (run.get("validation") or {}).get("sample_fingerprint")
        or (te.get("validation") or {}).get("sample_fingerprint")
    )


def _resumable_generation(
    job, result_path, generation_robot, seen_fingerprints,
    expected_solvability_profile=None,
):
    """Return a previously verified generation artifact for expert-only retry."""
    if not result_path.exists():
        return None
    try:
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        generation_path = Path(str(prior.get("generation_json") or ""))
        if not generation_path.is_file():
            return None
        run = json.loads(generation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    selected_native_object_id = (
        prior.get("selected_native_object_id") or job.get("target_native_object_id")
    )
    identity_job = {
        **job,
        "target_native_object_id": selected_native_object_id,
    }
    formal_errors = check_run(generation_path, run)
    formal_errors.extend(
        _identity_errors(identity_job, run, expected_solvability_profile)
    )
    generated_robot = str((run.get("robot") or {}).get("model") or "")
    if generated_robot.casefold() != generation_robot.casefold():
        formal_errors.append("generation_robot_mismatch")
    fingerprint = _saved_fingerprint(run)
    if not fingerprint:
        formal_errors.append("missing_sample_fingerprint")
    elif fingerprint in seen_fingerprints:
        formal_errors.append("duplicate_coverage_fingerprint")
    if run.get("ok") is not True or formal_errors:
        return None
    return {
        "generation_json": generation_path,
        "sample_fingerprint": fingerprint,
        "selected_native_object_id": selected_native_object_id,
        "generation_process_codes": prior.get("generation_process_codes") or [],
        "generation_seconds": float(prior.get("generation_seconds") or 0.0),
    }


def _resumable_ineligible(job, result_path, generation_robot, expert_robot):
    """Return a prior fully resolved native-target ineligibility result."""
    if not result_path.exists():
        return None
    try:
        prior = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        prior.get("status") != "ineligible_target"
        or prior.get("job_id") != job.get("job_id")
        or prior.get("scene") != job.get("scene")
        or prior.get("task") != job.get("task")
        or str(prior.get("generation_robot") or "").casefold()
        != generation_robot.casefold()
        or str(prior.get("expert_robot") or "").casefold() != expert_robot.casefold()
    ):
        return None
    reasons = prior.get("ineligibility_reasons") or []
    exact = job.get("target_native_object_id")
    candidates = job.get("target_native_candidates") or ([exact] if exact else [])
    if exact:
        resolved = any(reason.get("object_id") == exact for reason in reasons)
    else:
        resolved = _all_native_candidates_resolved(candidates, reasons)
    if not resolved:
        return None
    prior["resumed"] = True
    return prior


def main():
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, _signal_exit)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", default="0", help="Physical GPU index or 'auto'")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--process-retries", type=int, default=2)
    parser.add_argument("--expert-process-retries", type=int, default=2)
    parser.add_argument("--generation-timeout", type=int, default=900)
    parser.add_argument("--expert-timeout", type=int, default=900)
    parser.add_argument("--min-eligible-accept-rate", type=float, default=0.90)
    parser.add_argument("--max-runtime-ineligible-rate", type=float, default=0.05)
    parser.add_argument(
        "--expert-backend",
        choices=("oracle_symbolic", "physical_control"),
        default="physical_control",
    )
    parser.add_argument("--robot", default=None)
    parser.add_argument("--expert-robot", default=None)
    args = parser.parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error(
            "DASHSCOPE_API_KEY is required because DeltaSG coverage forbids heuristic-only generation"
        )
    if args.process_retries < 1 or args.expert_process_retries < 1:
        parser.error("process retry counts must be at least 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard index must satisfy 0 <= index < num shards")
    if not 0.0 <= args.min_eligible_accept_rate <= 1.0:
        parser.error("--min-eligible-accept-rate must be between 0 and 1")
    if not 0.0 <= args.max_runtime_ineligible_rate <= 1.0:
        parser.error("--max-runtime-ineligible-rate must be between 0 and 1")
    if args.gpu != "auto":
        try:
            if int(args.gpu) < 0:
                raise ValueError
        except ValueError:
            parser.error("--gpu must be a non-negative index or 'auto'")

    generation_robot = args.robot or args.expert_robot or (
        "fetch" if args.expert_backend == "oracle_symbolic" else "Tiago"
    )
    expert_robot = args.expert_robot or generation_robot
    if generation_robot.casefold() != expert_robot.casefold():
        parser.error("generation --robot and --expert-robot must match")
    if args.expert_backend == "physical_control" and expert_robot not in {"R1", "Tiago"}:
        parser.error("physical_control requires the shared robot R1 or Tiago")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    jobs = manifest.get("jobs") or []
    if args.num_shards > 1:
        scenes = sorted({job["scene"] for job in jobs})
        shard_scenes = {
            scene for index, scene in enumerate(scenes)
            if index % args.num_shards == args.shard_index
        }
        jobs = [job for job in jobs if job["scene"] in shard_scenes]
        print(
            f"[coverage] shard={args.shard_index}/{args.num_shards} "
            f"scenes={sorted(shard_scenes)} jobs={len(jobs)}",
            flush=True,
        )
    if args.limit > 0:
        jobs = jobs[: args.limit]
    root = Path(args.output_root)
    env = os.environ.copy()
    env["DELTASG_GPU"] = str(args.gpu)
    results = []
    seen_fingerprints = set()
    for index, job in enumerate(jobs, 1):
        job_root = root / job["label"] / job["scene"] / job["job_id"]
        result_path = job_root / "job_result.json"
        expert_result = job_root / "expert" / "expert_result.json"
        if job.get("preflight_ineligibility"):
            reason = dict(job["preflight_ineligibility"])
            result = {
                **job,
                "selected_native_object_id": None,
                "status": "ineligible_target",
                "generation_process_codes": [],
                "generation_seconds": 0.0,
                "formal_errors": [],
                "generation_json": None,
                "sample_fingerprint": None,
                "ineligibility_reasons": [reason],
                "expert_process_code": None,
                "expert_process_codes": [],
                "expert_seconds": 0.0,
                "expert_backend": args.expert_backend,
                "generation_robot": generation_robot,
                "expert_robot": expert_robot,
            }
            _write_json(result_path, result)
            results.append(result)
            counts = Counter(item["status"] for item in results)
            print(
                f"[coverage] {index}/{len(jobs)} {job['job_id']} "
                f"status=ineligible_target stage={reason.get('stage')}",
                flush=True,
            )
            _write_json(root / "coverage_progress.json", {
                "manifest": args.manifest,
                "completed": len(results),
                "total": len(jobs),
                "status_counts": dict(counts),
                "results": results,
            })
            continue
        resumed_ineligible = _resumable_ineligible(
            job, result_path, generation_robot, expert_robot
        )
        if resumed_ineligible is not None:
            results.append(resumed_ineligible)
            counts = Counter(item["status"] for item in results)
            print(
                f"[coverage] {index}/{len(jobs)} resume ineligible {job['job_id']}",
                flush=True,
            )
            _write_json(root / "coverage_progress.json", {
                "manifest": args.manifest,
                "completed": len(results),
                "total": len(jobs),
                "status_counts": dict(counts),
                "results": results,
            })
            continue
        if _accepted_expert(expert_result, args.expert_backend, expert_robot):
            result = json.loads(result_path.read_text()) if result_path.exists() else dict(job)
            generation_path = Path(str(result.get("generation_json") or ""))
            fingerprint = None
            generation_robot_matches = False
            if generation_path.is_file():
                resumed_run = json.loads(generation_path.read_text(encoding="utf-8"))
                fingerprint = _saved_fingerprint(resumed_run)
                resumed_robot = str((resumed_run.get("robot") or {}).get("model") or "")
                generation_robot_matches = (
                    resumed_robot.casefold() == generation_robot.casefold()
                )
            if (
                generation_robot_matches
                and fingerprint
                and fingerprint not in seen_fingerprints
            ):
                seen_fingerprints.add(fingerprint)
                result.update({
                    "status": "accepted",
                    "resumed": True,
                    "sample_fingerprint": fingerprint,
                    "generation_robot": generation_robot,
                    "expert_robot": expert_robot,
                    "expert_acceptance_kind": _expert_acceptance_kind(expert_result),
                })
                results.append(result)
                print(f"[coverage] {index}/{len(jobs)} resume accepted {job['job_id']}", flush=True)
                counts = Counter(item["status"] for item in results)
                _write_json(root / "coverage_progress.json", {
                    "manifest": args.manifest,
                    "completed": len(results),
                    "total": len(jobs),
                    "status_counts": dict(counts),
                    "results": results,
                })
                continue
            print(
                f"[coverage] {index}/{len(jobs)} regenerate non-unique resume {job['job_id']}",
                flush=True,
            )

        generation_dir = job_root / "generation"
        generation_json = None
        category = "retrieval_delivery" if job["label"].endswith("retrieval_delivery") else (
            "open_close" if job["label"].endswith("open_close") else "appliance"
        )
        command = [
            "code/run_omnigibson_single_gpu.sh", "conda", "run", "--no-capture-output",
            "-n", "behavior", "python", "code/run_online_deltasg.py",
            "--scene", job["scene"], "--robot", generation_robot, "--env-type", "A",
            "--task", job["task"], "--task-categories", category,
            "--num-envs", "1", "--task-objects", "1", "--context-objects", "0",
            "--warmup-steps", "20", "--settle-steps", "5", "--llm-model", MODEL,
            "--max-llm-retries", "4", "--single-attempt",
            "--placement-timeout", "60", "--relation-timeout", "10",
            "--max-placement-attempts", "6", "--max-total-placement-time", "180",
            "--max-global-cameras", "3", "--max-camera-pose-attempts", "6",
            "--camera-pose-render-steps", "4", "--min-manipulation-height", "0.10",
            "--max-manipulation-height", "1.55",
            "--solvability-profile", args.expert_backend,
            "--output-dir", str(generation_dir),
        ]
        if job.get("target_asset_category"):
            command += ["--target-asset-category", job["target_asset_category"]]
        if job.get("target_asset_model"):
            command += ["--target-asset-model", job["target_asset_model"]]
        status = "generation_failed"
        process_codes = []
        generation_seconds = 0.0
        formal_errors = []
        sample_fingerprint = None
        attempt_fingerprints = set()
        ineligibility_reasons = []
        scene_initialization_failure = None
        selected_native_object_id = job.get("target_native_object_id")
        native_candidates = job.get("target_native_candidates") or (
            [selected_native_object_id] if selected_native_object_id else []
        )
        resumed_generation = _resumable_generation(
            job,
            result_path,
            generation_robot,
            seen_fingerprints,
            args.expert_backend,
        )
        if resumed_generation:
            generation_json = resumed_generation["generation_json"]
            sample_fingerprint = resumed_generation["sample_fingerprint"]
            selected_native_object_id = resumed_generation["selected_native_object_id"]
            process_codes = resumed_generation["generation_process_codes"]
            generation_seconds = resumed_generation["generation_seconds"]
            status = "generated"
            print(
                f"[coverage] {index}/{len(jobs)} resume generation {job['job_id']}",
                flush=True,
            )
        attempt_schedule = (
            [] if resumed_generation else _native_attempt_schedule(job, args.process_retries)
        )
        for attempt, scheduled_native_object_id in enumerate(attempt_schedule, 1):
            shutil.rmtree(generation_dir, ignore_errors=True)
            seed = int(job["job_id"][:8], 16) + attempt
            attempt_command = list(command)
            if scheduled_native_object_id:
                selected_native_object_id = scheduled_native_object_id
                attempt_command += ["--target-native-object-id", selected_native_object_id]
            code, elapsed = _run(
                attempt_command + ["--seed", str(seed)],
                job_root / "generation.log",
                args.generation_timeout,
                env,
            )
            generation_seconds += elapsed
            process_codes.append(code)
            generation_json = _find_generated_run(generation_dir)
            if generation_json is not None:
                run = json.loads(generation_json.read_text(encoding="utf-8"))
                formal_errors = check_run(job_root / job["label"] / generation_json.name, run)
                generated_robot = str((run.get("robot") or {}).get("model") or "")
                if generated_robot.casefold() != generation_robot.casefold():
                    formal_errors.append("generation_robot_mismatch")
                if scheduled_native_object_id is None and native_candidates:
                    selected_native_object_id = _selected_native_target(run)
                identity_job = {**job, "target_native_object_id": selected_native_object_id}
                formal_errors.extend(
                    _identity_errors(identity_job, run, args.expert_backend)
                )
                sample_fingerprint = _saved_fingerprint(run)
                if not sample_fingerprint:
                    formal_errors.append("missing_sample_fingerprint")
                elif sample_fingerprint in seen_fingerprints or sample_fingerprint in attempt_fingerprints:
                    formal_errors.append("duplicate_coverage_fingerprint")
                if sample_fingerprint:
                    attempt_fingerprints.add(sample_fingerprint)
                if code == 0 and run.get("ok") is True and not formal_errors:
                    status = "generated"
                    break
            elif code == 0:
                _, rejected = _find_rejected_run(generation_dir)
                reason = _native_ineligibility(
                    {**job, "target_native_object_id": selected_native_object_id}, rejected
                )
                if reason:
                    ineligibility_reasons.append({
                        "object_id": selected_native_object_id,
                        **reason,
                    })
                    if job.get("target_native_object_id") and reason.get("stage") in {
                        "identity", "state_capability", "room", "pose", "manipulation_height",
                        "robot_approach",
                    }:
                        break
            scene_initialization_failure = _scene_initialization_failure(generation_dir)
            if scene_initialization_failure:
                status = "scene_initialization_failed"
                break

        deterministic_ineligible = bool(
            job.get("target_native_object_id")
            and ineligibility_reasons
            and ineligibility_reasons[-1].get("stage")
            in {
                "identity", "state_capability", "room", "pose", "manipulation_height",
                "robot_approach",
            }
        )
        candidates_resolved = bool(
            not job.get("target_native_object_id")
            and _all_native_candidates_resolved(native_candidates, ineligibility_reasons)
        )
        if (
            status == "generation_failed"
            and native_candidates
            and (
                deterministic_ineligible
                or candidates_resolved
                or (
                    job.get("target_native_object_id")
                    and len(ineligibility_reasons) == len(process_codes)
                    and len(process_codes) == args.process_retries
                )
            )
        ):
            status = "ineligible_target"

        expert_code = None
        expert_process_codes = []
        expert_seconds = 0.0
        if status == "generated" and generation_json is not None:
            expert_command = [
                "code/run_omnigibson_single_gpu.sh", "conda", "run", "--no-capture-output",
                "-n", "behavior", "python", "code/run_deltasg_expert.py",
                "--input-json", str(generation_json), "--output-dir", str(job_root / "expert"),
                "--backend", args.expert_backend, "--robot", expert_robot, "--llm-model", MODEL,
                "--min-manipulation-height", "0.10", "--max-manipulation-height", "1.55",
            ]
            for _ in range(args.expert_process_retries):
                shutil.rmtree(job_root / "expert", ignore_errors=True)
                expert_code, elapsed = _run(
                    expert_command, job_root / "expert.log", args.expert_timeout, env
                )
                expert_seconds += elapsed
                expert_process_codes.append(expert_code)
                expert_state = _expert_result_state(
                    expert_result, args.expert_backend, expert_robot
                )
                if expert_state in {"accepted", "rejected"}:
                    break
            status = _coverage_status_for_expert_state(expert_state)
            if status == "accepted" and sample_fingerprint:
                seen_fingerprints.add(sample_fingerprint)

        result = {
            **job,
            "selected_native_object_id": selected_native_object_id,
            "status": status,
            "generation_process_codes": process_codes,
            "generation_seconds": round(generation_seconds, 3),
            "formal_errors": formal_errors,
            "generation_json": str(generation_json) if generation_json else None,
            "sample_fingerprint": sample_fingerprint,
            "ineligibility_reasons": ineligibility_reasons,
            "scene_initialization_failure": scene_initialization_failure,
            "expert_process_code": expert_code,
            "expert_process_codes": expert_process_codes,
            "expert_seconds": round(expert_seconds, 3),
            "expert_backend": args.expert_backend,
            "generation_robot": generation_robot,
            "expert_robot": expert_robot,
            "generation_resumed": bool(resumed_generation),
            "expert_acceptance_kind": (
                _expert_acceptance_kind(expert_result) if status == "accepted" else None
            ),
        }
        _write_json(result_path, result)
        results.append(result)
        counts = Counter(item["status"] for item in results)
        print(f"[coverage] {index}/{len(jobs)} {job['job_id']} status={status} counts={dict(counts)}", flush=True)
        _write_json(root / "coverage_progress.json", {
            "manifest": args.manifest,
            "completed": len(results),
            "total": len(jobs),
            "status_counts": dict(counts),
            "results": results,
        })

    counts = Counter(item["status"] for item in results)
    preflight_ineligible = sum(
        item["status"] == "ineligible_target" and bool(item.get("preflight_ineligibility"))
        for item in results
    )
    runtime_ineligible = counts["ineligible_target"] - preflight_ineligible
    runtime_eligible_jobs = len(results) - preflight_ineligible
    runtime_ineligible_rate = (
        runtime_ineligible / runtime_eligible_jobs if runtime_eligible_jobs else 0.0
    )
    acceptance_kinds = Counter(
        item.get("expert_acceptance_kind")
        for item in results
        if item.get("expert_acceptance_kind")
    )
    report = {
        "schema_version": "enva_expert_coverage_result.v1",
        "manifest": args.manifest,
        "expert_backend": args.expert_backend,
        "generation_solvability_profile": args.expert_backend,
        "generation_robot": generation_robot,
        "expert_robot": expert_robot,
        "completed": len(results),
        "total": len(jobs),
        "status_counts": dict(counts),
        "accepted_rate": counts["accepted"] / len(results) if results else 0.0,
        "eligible_accept_rate": (
            counts["accepted"] / (len(results) - counts["ineligible_target"])
            if len(results) > counts["ineligible_target"] else 0.0
        ),
        "expert_acceptance_kind_counts": dict(acceptance_kinds),
        "strict_physical_rate": (
            acceptance_kinds["strict_physical"]
            / (len(results) - counts["ineligible_target"])
            if len(results) > counts["ineligible_target"] else 0.0
        ),
        "assisted_physical_rate": (
            acceptance_kinds["assisted_physical"]
            / (len(results) - counts["ineligible_target"])
            if len(results) > counts["ineligible_target"] else 0.0
        ),
        "structural_resolution_rate": (
            (counts["accepted"] + counts["ineligible_target"]) / len(results)
            if results else 0.0
        ),
        "min_eligible_accept_rate": args.min_eligible_accept_rate,
        "preflight_ineligible": preflight_ineligible,
        "runtime_ineligible": runtime_ineligible,
        "runtime_ineligible_rate": runtime_ineligible_rate,
        "max_runtime_ineligible_rate": args.max_runtime_ineligible_rate,
        "results": results,
    }
    _write_json(root / "coverage_result.json", report)
    audit_command = [
            "python", "code/audit_deltasg_expert.py", "--root", str(root),
            "--output", str(root / "expert_audit.json"), "--min-accept-rate", "0.0",
            "--require-backend", args.expert_backend,
        ]
    if args.expert_backend == "physical_control":
        audit_command.append("--require-physical-trajectory")
    audit_code = subprocess.run(
        audit_command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    ).returncode
    report["expert_audit_process_code"] = audit_code
    _write_json(root / "coverage_result.json", report)
    raise SystemExit(
        0
        if (
            len(results) == len(jobs)
            and counts["accepted"] + counts["ineligible_target"] == len(jobs)
            and report["eligible_accept_rate"] >= args.min_eligible_accept_rate
            and report["runtime_ineligible_rate"] <= args.max_runtime_ineligible_rate
            and audit_code == 0
        )
        else 2
    )


if __name__ == "__main__":
    main()
