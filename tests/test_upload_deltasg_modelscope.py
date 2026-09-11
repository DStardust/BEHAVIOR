import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "upload_deltasg_modelscope.sh"


def test_upload_uses_staged_release_and_cached_modelscope_login(tmp_path):
    staged = tmp_path / "release"
    staged.mkdir()
    (staged / "accepted_manifest.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (staged / "dataset_summary.json").write_text("{}\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "modelscope.args"
    fake_modelscope = bin_dir / "modelscope"
    fake_modelscope.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$MODELSCOPE_ARGS_FILE\"\n",
        encoding="utf-8",
    )
    fake_modelscope.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "MODELSCOPE_ARGS_FILE": str(args_file),
            "MODELSCOPE_COMMIT_MESSAGE": "DeltaSG regression release",
        }
    )
    result = subprocess.run(
        [str(SCRIPT), str(staged)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "upload",
        "DStardust/EM-STORM",
        str(staged),
        "--repo-type",
        "dataset",
        "--max-workers",
        "8",
        "--commit-message",
        "DeltaSG regression release",
        "--commit-description",
        "2 generation-and-expert accepted samples with complete robot and global visualizations",
    ]


def test_upload_rejects_unstaged_output(tmp_path):
    result = subprocess.run(
        [str(SCRIPT), str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 65
    assert "Refusing upload" in result.stderr


def test_upload_help_is_available_without_modelscope_login():
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "<staged-directory> [repo-id]" in result.stderr
