import json

from monitor_envbc_multiscene_e2e import checkpoint_counts


def test_monitor_counts_exceptions_without_double_counting_runs(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"run_counter": 2, "successful_samples": [{}, {}]}))
    (tmp_path / "summary.json").write_text(json.dumps({"runs": [{}, {}], "attempt_errors": [{}]}))
    assert checkpoint_counts(checkpoint) == (3, 2)
    checkpoint.write_text(json.dumps({"run_counter": 3, "successful_samples": [{}, {}]}))
    assert checkpoint_counts(checkpoint) == (3, 2)


def test_monitor_uses_checkpoint_before_summary_exists(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"run_counter": 3, "successful_samples": [{}]}))
    assert checkpoint_counts(checkpoint) == (3, 1)
