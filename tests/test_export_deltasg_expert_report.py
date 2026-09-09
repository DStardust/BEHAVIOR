import json
from pathlib import Path
from html.parser import HTMLParser

import pytest

from export_deltasg_expert_report import export_report


def test_report_preserves_images_and_links_all_steps(tmp_path):
    image = tmp_path / "original.png"
    image.write_bytes(b"unchanged-image-content")
    generation = tmp_path / "generation.json"
    generation.write_text('{}')
    result = {"accepted": True, "run_id": "example", "input": str(generation),
              "backend": {"name": "oracle_symbolic"}, "task_name": "task", "scene": "scene",
              "observation_events": [{"event_id": "pre", "robot_primary": {"paths": {"rgb": str(image)}},
                                      "global_cameras": []}],
              "steps": [{"step": {"step_id": 1, "primitive": "WIPE", "nl": "Clean <cup>"},
                         "pre_observation": "pre", "post_observation": "pre",
                         "postcondition": {"actual_systems": []}}]}
    source = tmp_path / "result.json"
    source.write_text(json.dumps(result))
    output = tmp_path / "report"
    export_report([source], output)
    assert (output / "example/pre/Robot/rgb.png").read_bytes() == image.read_bytes()
    page = (output / "index.html").read_text()
    assert "Clean &lt;cup&gt;" in page
    assert "not continuous physical motion" in page
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src", "href"}:
                    assert (output / value).is_file()
    Links().feed(page)
    result["accepted"] = False
    source.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Not an accepted"):
        export_report([source], tmp_path / "rejected")
