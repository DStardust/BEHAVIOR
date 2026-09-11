# Env-B Repair and Runtime Evidence

Updated: 2026-09-08. Workspace: `BEHAVIOR_envbc_release`.

## Acceptance

### 2026-09-08 14:05 CST checkpoint

- `envb_30min_20260908`: generation finished 3/4 slots. Independent expert
  `envb_30min_expert_20260908`: 2/3 accepted (dishes, broken object), fire failed
  navigation to the fire source. This is 2/4 requested slots end-to-end, not full
  Env-B completion. One expert environment load; 452.6 seconds for three inputs.
- Fire extinguisher separated floor selection is runtime verified: 442/729
  candidates survived filtering, accepted target/tool AABB gap 1.60055 m.
  Added exact expert-navigation preflight for spawned fire sources after this
  failure; this additional fix is NOT yet runtime verified.
- `envb_navfit2_20260908` uses container-size pairing and navigation preflight.
  Official Inside still rejects some sock/hamper and shard/bin pairs. Outer size
  compatibility alone is insufficient; no predicate substitution is allowed.
- User authorized laundry-only low grasp. Generator and expert now share a
  0.04 m AABB-centre minimum for collect_dirty_clothes clothing categories.
  Other objects/tasks retain their configured minima. Visibility, navigation,
  collision and official Inside checks remain required. Tests: 336 passed.
  `envb_laundry4cm_20260908` tmux on single GPU1 is validating this change;
  runtime completion is pending, not VERIFIED. LLM qwen3.8-max remains enabled.
- Report: `code/outputs/envb_report_latest_20260908/index.html`, two new accepted
  examples plus one explicitly legacy fire example. README distinguishes the
  old close-placement fire from current layout policy. Raw pre/post RGB only;
  these are symbolic state-supervision examples, not physical VLA trajectories.

Generation acceptance is not expert acceptance. Count an expert sample only
when its result has `accepted=true`, official final predicates pass, and the
observation / scene-integrity audit passes. These runs use `oracle_symbolic`,
not continuous physically controlled manipulation.

## Completed Runtime Checks

- `code/outputs/envb_replay_fix_20260908`: 5 expert attempts, 1 accepted
  (fire). Broken-object and dirty-clothes failed official `Inside`; dishes
  failed initial stain replay / particle-system reset. Not a successful batch.
- `code/outputs/envb_dishes_fix_20260908`: 2 attempts, 0 accepted.
  Both failed `Covered(stain)=True` initial replay. No manipulation was reached.
  This disproves collision-hold reordering alone as a complete fix.

## Current Changes

- Activate collisions before reconstructing official initial anomaly states;
  only then hold symbolic grasp targets.
- Clear noninitial particle systems while the simulator is stopped, before
  rebuilding physics views. The previous playing-mode cleanup invalidated views.
- Check the official destination volume and exercise official `Inside` during
  generation; restore the entire simulation state afterwards. Audit rejects
  new container tasks without successful preflight evidence. Old accepted JSON
  files must not be relabeled with fabricated preflight results.
- Save official rigid-object stain/dirt particle local poses, scales and link
  attachments. Expert and visualization restore this snapshot instead of
  randomly resampling it. Old JSON without a snapshot retains official sampling.
  Snapshot restoration still requires the official `Covered` predicate.
- Synchronize physics before ray-based Covered sampling after replay teleports.
- Fingerprint generated objects by category/model/location, not run-specific
  names. Ignore particle rendering details when detecting duplicate tasks.
  Include anomaly carriers in diversity reports; exclude support furniture.
- Fix Env-C family balancing; the multiscene runner optionally includes Env-A
  through `ENVA_NUM` and permits distinct repeated task types in Env-C.
- Reject failed anomaly setup before camera rendering. Env-B preparation and
  selection now honor per-task skip sets. These two changes were made after
  the current generation process started and require a fresh generation run.
- Preserve destination-preflight evidence in standardized task validation.
  The auditor also reads the root validation in early anomaly.v1 exports;
  a failed task-level preflight cannot be overridden by a successful root one.

Static regression: 329 tests passed; `git diff --check` passed.

## In Progress

`tmux`: `envb_snapshot_20260908`

Output: `code/outputs/envb_snapshot_20260908`

Single GPU 1, Ihlen_1_int, 4 requested Env-B samples, all four anomaly families,
expert replay enabled. LLM explicitly `qwen3.8-max`. Official camera placement,
2-3 global cameras; no markers. Child proxy variables are unset by the GPU
wrapper, not in the parent agent environment.

Generation finished: 3/4 requested slots accepted, 10 raw attempts. Generation
audit: 3/3 clean, no duplicate fingerprints. Dirty-clothes exhausted retries;
failures include official Inside preflight and floor navigation reachability.
Accepted targets: mug ehnmxj (20 saved stain particles), beeswax_candle ouzkdj,
broken_light_bulb cugtye with trash_can wklill (official Inside preflight true).
The earlier trash_can candidate had a fillable meta link but failed actual
sampling; therefore absent annotations are not the general failure mechanism.

Expert replay finished: 1/3 accepted (fire), 25% of requested generation slots.
Broken object failed navigation at step 1: no stand-off satisfied the route,
clearance, visibility and 1.15 m gates. Dishes restored the snapshot successfully
and reached WIPE at step 4, but its immediate Covered postcondition was stale:
the official setter reads the old value internally before removing particles,
repopulating the same-step cache. Added explicit state.clear_cache after WIPE.
Replay of all three inputs now runs in `envb_expert_cachefix_20260908`, output
`code/outputs/envb_expert_cachefix_20260908`. This does not repair broken-object
navigation or dirty-clothes generation. These changes are not yet fully
end-to-end runtime VERIFIED. In particular, a destination
preflight rejection is honest filtering, not proof that container placement has
been repaired. Inspect actual container meta links before changing placement.

Cache-fix replay finished: 2/3 expert accepted (dishes and fire), broken object
still rejected at navigation step 1. One environment load, 362.5 seconds total.
This is 50% end-to-end against the original four requested slots, not a broad
success-rate estimate. The exact previously failing dish now completes all four
steps. Robot RGB was inspected at pre-WIPE and post-WIPE; the cup remains stable,
but dirt contrast is weak in the robot image. Official state success must not be
treated as proof of visually obvious dirt. Improve particle visibility and add
visual evidence checks before claiming the visible-anomaly requirement complete.

Remaining: dirty-clothes container / floor-route compatibility; broken-object
generation-vs-expert navigation consistency; strong pre/post dirt visibility;
machine-wash recipe runtime coverage; multiscene and multi-seed regression.

## Release Gate - 2026-09-09

`code/outputs/envb_fire3_release_20260909` generated three Ihlen_1 fire samples.
Generation and camera audit passed 3/3 with no issues. Fire-source diversity was
`rice_cooker`, `beeswax_candle`, and `space_heater`; two extinguisher models were
used. The generator now tries only live navigation-component rooms plus the fire
room. Topology-only rooms remain diagnostic evidence instead of consuming about
90 seconds each in an official floor-placement attempt that cannot route.

The original expert batch accepted 2/3. The remaining sample reached and grasped
the extinguisher, then rejected the route to the fire. Generation had preflighted
that exact transition while excluding the carried extinguisher from obstacle
inflation, but replay omitted `held_object` from the final waypoint call. The
extinguisher therefore blocked its own route after grasp. Replay now uses the
same carried-object exclusion and the saved, already-preflighted transition when
the post-grasp observation stance differs by at most 0.5 m. Live route planning,
the 1.15 m operation envelope, visibility, scene-integrity, robot-stability, and
official `OnFire=False` postconditions remain required.

Targeted replay of that previously rejected sample is verified at
`code/outputs/runtime_checks/envb_fire_route_recovery_v3_20260909`: all four
steps accepted, `qa_eligible=true`, scene integrity passed, expert audit 1/1,
and process exit code 0. Step 3 records the recovered preflight start pose
`[0.2, -0.6, 0.56092]`. This is two original batch passes plus one targeted
repair verification, not a fresh whole-batch 3/3 replay.

The multiscene E2E script now treats a phase as complete only when its `.exit`
file contains zero. Failed generation, audit, or expert phases rerun on resume;
successful phases remain skipped. The exact tmux invocation and monitor command
are documented in `docs/deltasg_usage.md`.

Static release gate before upstream integration: 356 tests passed; Python
compilation, shell syntax, and `git diff --check` passed. Next runtime priority
is the broken-object official `OnTop` sweep transition; broom and dustpan task
semantics must remain intact.

## Presentation Export and Follow-up

`code/outputs/envb_report_20260908/index.html` contains the two accepted examples
(fire and hand-wash), all four steps' pre/post observations, robot RGB and both
global camera RGB streams, plus generation and expert JSON. Images are copied
unaltered at 640x480. `code/outputs/envb_report_20260908.tar.gz` is the portable
package. The HTML explicitly identifies symbolic supervision and weak stain
contrast, not continuous physical trajectories. All local image/data links
were checked; fire global RGB and dish robot RGB were visually inspected.

Follow-up tmux: `envb_remaining_20260908`, output
`code/outputs/envb_remaining_20260908`. GPU 1, Ihlen_1_int, four requested samples
limited to dirty_clothes and broken_object, then automatic expert replay.
This run includes the skip-set and early-rejection fixes plus container/subject
extent diagnostics. It is still running, not VERIFIED.

## Household Layout Correction

User review rejected the close sponge/dish and extinguisher/fire arrangement.
The existing report is now explicitly labeled as legacy layout examples.
Do not present those samples as validated realistic household storage.

Generator now anchors hand-wash sponge/soap at the bound sink instead of the
 dirty dish. Policy: horizontal AABB-edge gap to dish >=0.4m and gap to sink
<=1.0m, on an open support. Fire extinguisher is upright on a reachable floor,
>=2.5m horizontal edge gap from fire; no improvised furniture fallback. These
are dataset layout heuristics, not safety-code compliance claims. Constraints
are evaluated at candidate selection, actual placement, and after warmup;
operation reach remains 1.15m and navigation must connect the separate locations.

333 static tests passed before the last small reused-tool check. Fresh runtime
queued on GPU 1 in tmux `envb_layout_20260908`, output
`code/outputs/envb_layout_20260908` (Ihlen_1_int, four requests, dishes/fire,
then expert replay). Runtime outcome remains pending; the earlier remaining-task
batch uses the old source loaded at its launch.

## Next Acceptance Steps

1. Finish this generation and expert run; record counts separately.
2. Inspect every failure, especially official container-volume availability and
   particle snapshot reconstruction. Check pre/post RGB and segmentation for
   accepted samples. Do not weaken official predicates to increase counts.
3. Repeat all Env-B recipe paths, then multiple seeds and scenes. Four task
   families do not by themselves cover all alternative recipes or objects.
4. Run Env-A/B/C diversity batches using the same validated source; report
   unique task/model/location counts and expert acceptance, not only output files.

## Verified Fire and Env-A Stability - 2026-09-08

The fire visual is now the packaged `code/assets/Flame_Animation.usdz` flame
combined with the official OmniGibson Flow emitter configured as a visible
vertical smoke column. Generation and expert replay both record mode
`deltasg_usdz_flame_smoke_column_v1`; audit requires the flame asset, visible
flame and smoke, radius 0.18 m, upward velocity 1.5, and fade 0.12.

Fire-extinguisher placement is floor-only, upright, and at least 2.5 m from the
fire by horizontal AABB-edge distance. Reachable rooms other than the fire room
are tried first; the fire room is the last fallback. A floor-placement control
flow bug skipped the expert navigation preflight before breaking out of the
candidate loop. Floor and support placements now use the same preflight,
including the route from the extinguisher stance to the fire stance while the
tool is held. `run_envbc_multiscene_e2e.sh` also sets the per-task retry limit to
4, matching its existing per-sample limit, so a single anomaly family is not
discarded after two strict placement rejections.

Runtime evidence:

- `code/outputs/enva_stability6_final_20260908`: Beechwood_0_int generated 6/6
  representative Env-A tasks with zero rejected tasks. The independent audit
  reports 6 runs, no issues, six distinct task names, and 2-3 global cameras.
- `code/outputs/envb_fire_strict_final2_20260908`: Ihlen_1_int generated one
  audited fire sample after two rejected attempts in the same process. The
  accepted extinguisher-to-fire gap is 1.68 m and the serialized floor
  navigation preflight contains both valid stances. Generation audit is 1/1.
- The final expert replay is 1/1 accepted and QA-eligible. All four steps pass:
  navigate to extinguisher, grasp it, navigate to fire, and extinguish. The
  official `OnFire` postcondition is false; robot stability and scene integrity
  pass with no moved native or stationary DeltaSG object. The expert audit has
  no artifact or QA-gate violations.

Static verification: 354 tests passed; Python compilation, shell syntax, and
`git diff --check` passed. This is focused runtime evidence, not yet a broad
multi-scene Env-B rate estimate.

## Dirty-clothes Three-recipe Gate - 2026-09-09

The Beechwood_0 regression exposed two independent generation defects. The
machine recipe selected rigid clothing whose live AABB was larger than the
washer's official fillable volume. The `cloth_basket` recipe is explicitly
substituted with a real fillable `wicker_basket`, but incorrectly requested
`OnTop` on its narrow rim. Generation now filters clothing models against the
bound washer's official fillable/openfillable visual-boundary extents before
spawning. The substituted wicker basket, hamper, and washer all require the
official `Inside=True` predicate; no relation threshold was relaxed.

Fresh generation evidence is
`code/outputs/envb_clothes3_b0_fitfix_20260909`. Beechwood_0 produced 3/3
requested samples in 3 raw attempts: `machine_wash_clothes`, `put_in_hamper`,
and `put_on_cloth_basket`. All three selected the only installed rigid garment
model that conservatively fits the destination (`sock::vpafgj`) and passed
official `Inside` preflights. The machine sample additionally passed
reversible official `Open` and `ToggledOn` preflights. Camera coverage passed
for every generated sample.

The first expert replay accepted only the machine recipe. The hamper and wicker
basket both fell through the floor by 2.2345 m immediately after the first
navigation capture, while all native objects and the robot remained stable.
This was a reconstruction defect: generated task destinations were preloaded
as dynamic bodies even though the plan never actuates those containers.
Expert reconstruction now anchors rigid `task_destination` objects just like
generated `task_support` objects. Grasped task objects and interaction tools
remain dynamic.

Targeted replay evidence is
`code/outputs/runtime_checks/envb_clothes3_b0_anchorfix_20260909`. The three
recipes are 3/3 accepted and QA-eligible with no failure stages, artifact
violations, or scene-integrity movement. The washer recipe completes all eight
steps through `OPEN`, `PLACE_INSIDE`, `CLOSE`, and `TOGGLE_ON`; hamper and wicker
basket each complete four steps through `PLACE_INSIDE`. Static verification is
379 tests passed plus Python compilation and `git diff --check`. This is a
single-scene all-recipe gate; multi-scene Env-B regression remains required.

## Benevolence_0 Generation and Expert Gate - 2026-09-10

The previous two-scene run exposed a false grasp-height requirement on broken
pieces, which are swept rather than grasped. Broken anomalies now use the real
`SWEEP_INTO` floor-height contract; the expert applies the same floor-level
contract. Brooms and dustpans remain dynamic and are grasped normally, but their
task-specific grasp check recognizes the handle affordance instead of rejecting
a flat tool by its AABB centre. The official plan still forbids grasping broken
pieces by hand and requires one `SWEEP_INTO` plus one `EMPTY_INTO`.

Open-surface-only assets no longer fall through to an unvalidated structural
floor candidate. Spawned fire sources are filtered by their actual placement
mode before import: compact appliances require a real OnTop support, while a
space heater may use a validated floor pose. Sweep `OnTop` preflight retains two
bounded low-level samples: four samples were tested but made the official,
uninterruptible call approach 120 seconds in Beechwood_0, so that change was
reverted. Floor fallback retains live collision, navigation,
visibility, and scene-integrity validation and never selects a pose overlapping
the robot. A fire task whose every reachable extinguisher placement has zero
candidates under the 2.5 m AABB-edge storage gap is skipped for the remainder of
that simulator process; the distance requirement is not lowered.

Fresh end-to-end evidence is
`code/outputs/envb_b0_fix5_20260910/Benevolence_0_int`. Generation produced 4/4
requested samples in five raw attempts (80% raw-attempt success): one broken
object cleanup, two dirty-clothes recipes, and one fire emergency. The fire used
a floor-staged `space_heater`; its extinguisher gap is 2.749 m against the 2.5 m
minimum. Generation audit, expert process, and expert audit all exited zero.
Expert replay accepted 4/4 with no failure stage, QA-gate violation, artifact
violation, or scene-integrity rejection. This scene has no reachable official
dirty-dish infrastructure, so `clean_dirty_dishes` is correctly unavailable
rather than synthesized or counted as a failed generated sample.

The corresponding Beechwood_0 regression is terminal at
`code/outputs/envb_beech_fix5_20260910`. Generation produced all four requested
task families in seven raw attempts (4/4 generated, 57.1% raw-attempt success),
with four clean audit records, four different target categories/models, and
2-3 global cameras per sample. The pre-fix expert replay accepted fire, dirty
clothes, and broken cleanup (3/4). Dirty dishes failed because generation
validated the dishwasher exterior AABB while `PLACE_INSIDE` used the official
fillable-volume centre. Inside routes now use that same operation point; a
washer or dishwasher whose complete operation ring is unreachable is rejected
by exact object id for the remainder of the process instead of repeatedly
producing an unsolvable sample. No reach or state threshold was relaxed.

## Broken-object Stability and Camera-operation Gate - 2026-09-10

Broken cleanup now validates the official `OnTop` sweep relation after eight
physics steps before preserving its replay pose. Fresh generation at
`code/outputs/envb_broken_stable_20260910/Beechwood_0_int/generation` produced
2/2 requested samples in exactly two raw attempts with no errors. Both official
relations remained stable, and the generation audit is clean. The samples use
different installed anomaly assets: `broken_light_bulb::cugtye` and
`broken_glass::beltgg`.

The first expert diagnostics established that a post-sweep floor target could
fall below Tiago's official head-tilt envelope when the navigation stance was
only 0.608 m from the dustpan. Navigation before `SWEEP_INTO` now constrains the
real dustpan operation point to 0.75-1.15 m and faces the midpoint between the
pieces and dustpan. Candidate selection still validates traversability, route,
native occupancy, line of sight, and the 1.15 m distance to the pieces. It does
not translate the robot after the sweep. When official instance segmentation
assigns overlapping dustpan/payload pixels to either object, the post frame is
accepted only if one member has a valid robot-primary bbox while both remain
visible in the required robot/global union.

Final replay is
`code/outputs/envb_broken_stable_20260910/Beechwood_0_int/expert_distgate`:
2/2 accepted and QA-eligible, with all 16 plan steps accepted, no failure stage,
no artifact or QA-gate violation, stable robot and scene-integrity checks, and
both official `SWEEP_INTO` / `EMPTY_INTO` postconditions. The second official
`OnTop` transition took 71.4 seconds but completed within the existing sample
timeout. This closes the focused broken-cleanup gate; it is not an exhaustive
all-scene Env-B rate claim. Broad multi-scene regression remains required after
source freeze.

Final static verification: 391 tests passed; all five changed production Python
modules compiled; relevant shell entrypoints passed `bash -n`; and
`git diff --check` passed.

## Fifteen-scene Env-B Regression - 2026-09-10

Frozen-source run `code/outputs/envb_multiscene_d0ea68f_20260910_105000`
completed all 15 configured scenes without leaving a simulator process. It
requested four samples per scene and generated 46/60 (76.7%) from 144 raw
attempts. Expert replay accepted 37/46 (80.4%); requested-slot end-to-end yield
was therefore 37/60 (61.7%). Per-task expert results were fire 9/9, dirty
clothes 16/18, broken cleanup 10/14, and dirty dishes 2/5.

All four requested samples were generated in 10/15 scenes. Beechwood_1,
Benevolence_1, Benevolence_2, and Pomaria_0 generated only 1/4; Wainscott_0 and
Wainscott_1 generated 3/4. The low-yield scenes exhausted strict placement,
same-room navigation, complete-route, or official relation preflights; they did
not serialize rejected attempts as successful samples. Expert failures comprise
five post-visibility failures, three execution/navigation failures, and one
persistent-worker crash. This run verifies the 80% expert target over 46
multi-scene samples, but generation yield and the resulting 61.7% end-to-end
rate remain below the desired 70-80% range and are the next work item.

## Placement-diversity Gate - 2026-09-10

Accepted samples now persist the actual room, support, placement mode, model,
semantic role, XY position, and 25 cm position bin for every generated object.
Subsequent floor and OnTop sampling consumes that history using a 0.50 m
maximin preference. Candidate diversity is applied only after household-layout,
footprint, collision, support-capacity, and reachability filtering. In compact
spaces that cannot provide 0.50 m separation, the farthest remaining legal pose
is used. Three fallback floor candidates for one object also avoid each other.
The private sampler history is removed from exported placement records.

Fresh focused evidence is
`code/outputs/envb_fire_position_diversity_20260910_162854`: Beechwood_0
generated 3/3 fire samples in one simulator process. The audit reports 3 clean
runs, no duplicate fingerprints, 6 generated-object position records, 6/6
unique category-room-25 cm bins, and zero repeated bins. The fire sources use a
floor pose and two different tables; the extinguishers use three different
corridor positions. The corridor's first two extinguisher poses are more than
0.50 m apart; the third uses the maximum legal spacing after the narrow floor
region is saturated. Full static verification is 392 passing tests plus
`py_compile`, `bash -n`, and `git diff --check`.

## Active diversified 15-scene generation - 2026-09-10

Commit `a79a4c8` is running a generation-only Env-B batch in tmux session
`deltasg_envb_diverse15`. The output root is
`code/outputs/envb_diverse_15scenes_a79a4c8_20260910_163923`. It requests eight
samples in each of the 15 versioned scenes (120 slots total), rotates all four
Env-B anomaly types, uses `qwen3.8-max`, and records a 0.50 m placement-diversity
preference in `config.json`. Expert replay is intentionally disabled for this
generation pass and must use the accepted generation outputs in a subsequent
audited stage.

Progress is read without attaching to Kit:

```bash
python code/monitor_envbc_multiscene_e2e.py \
  code/outputs/envb_diverse_15scenes_a79a4c8_20260910_163923
```

The first-scene health check reached 2/8 generated from four raw attempts while
the same simulator process remained alive. The terminal result must replace
this launch snapshot after the batch completes; no success-rate claim is made
from this partial count.

## Diversified 15-scene terminal audit - 2026-09-11

The generation-only run above completed all scenes and left no simulator
process. It generated 87/120 requested slots (72.5%) from 312 raw attempts:
34 dirty-clothes, 28 broken-object, 17 fire, and 8 dirty-dish samples. Four
scenes reached 8/8; the weakest scenes were `Beechwood_1_int` and
`Pomaria_0_int` at 1/8. Every accepted JSON has two or three global cameras,
and no pair has the same full sample fingerprint.

A strict aggregate audit found that only 85/120 requested slots (70.8%) were
clean. Two `Merom_0_int` broken-object records had reused the same native trash
can in an unreachable utility room without recording task-destination approach
or height evidence. Reused manipulated objects are now checked before the
reuse branch can return success. An ineligible native object is left untouched
and a normally placed, collision/reachability/height-validated replacement is
attempted in the requested room. A real `Merom_0_int` diagnostic at
`code/outputs/runtime_checks/merom0_reusegate_20260911_095729` exercised this
branch: the native trash can was rejected with `approach_ok=False`, and the
replacement passed the official `Inside` preflight. The attempt later failed
the independent official broken-piece-to-dustpan `OnTop` preflight. The same
two-slot run produced one separately accepted broken-cleanup sample whose
reachable native destination records a 0.724 m approach and valid 0.199 m
operation height. Strict audit is 1/1 clean with three generated position
records in three unique bins; the process exit is 2 because the other requested
slot exhausted its strict retry budget.

Placement history and audit summaries now exclude `mode=reused` scene-native
objects and compare historical positions within the same object category.
This avoids counting an unchanged native container as generated-placement
diversity and prevents other tool categories from consuming the 0.50 m spacing
budget. The old output, re-audited under this definition, has 227 generated
placement records, 226 unique category-room-25 cm bins, and one repeated bin;
the next fresh batch must verify that the category-scoped maximin rule removes
that remaining Ihlen trash-can repeat.

The multiscene shell runner also no longer marks partial production as `DONE`:
generation audits use `--fail-on-issues`, per-scene and root state become
`PARTIAL` when any requested phase exits nonzero, and the shell returns nonzero
accordingly. Final static verification is 395 passing tests, Python compilation,
shell syntax validation, and `git diff --check`.
