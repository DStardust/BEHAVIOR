"""Export accepted expert observations as a portable, unaltered RGB report."""

import argparse
import html
import json
import shutil
from pathlib import Path


def export_report(results, output):
    output.mkdir(parents=True, exist_ok=True)
    sections = []
    manifest = []
    for result_path in results:
        result = json.loads(result_path.read_text())
        if result.get("accepted") is not True:
            raise ValueError(f"Not an accepted expert result: {result_path}")
        sample_id = result["run_id"]
        sample_dir = output / sample_id
        sample_dir.mkdir(exist_ok=True)
        shutil.copy2(result_path, sample_dir / "expert_result.json")
        shutil.copy2(result["input"], sample_dir / "generation.json")
        events = {event["event_id"]: event for event in result["observation_events"]}
        rows = []
        for step_record in result["steps"]:
            step = step_record["step"]
            cells = []
            for phase in ("pre", "post"):
                event_id = step_record[f"{phase}_observation"]
                event = events[event_id]
                views = [("Robot", event["robot_primary"])] + [
                    (view["camera_id"], view) for view in event["global_cameras"]
                ]
                pictures = []
                for label, view in views:
                    source = Path(view["paths"]["rgb"])
                    target = sample_dir / event_id / label / "rgb.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    relative = target.relative_to(output).as_posix()
                    pictures.append(f'<figure><figcaption>{html.escape(label)}</figcaption>'
                                    f'<a href="{relative}"><img src="{relative}" loading="lazy"></a></figure>')
                cells.append(f'<div><h4>{phase.upper()} / {html.escape(event_id)}</h4>'
                             f'<div class="views">{"".join(pictures)}</div></div>')
            evidence = step_record.get("postcondition_after_capture") or step_record["postcondition"]
            rows.append(f'<section><h3>{step["step_id"]}. {html.escape(step["primitive"])}: '
                        f'{html.escape(step["nl"])}</h3>{"".join(cells)}'
                        f'<details><summary>Official postcondition</summary><pre>'
                        f'{html.escape(json.dumps(evidence, indent=2))}</pre></details></section>')
        title = f'{result["task_name"]} / {result["scene"]}'
        sections.append(f'<article id="{sample_id}"><h2>{html.escape(title)}</h2>'
                        f'<p>ACCEPTED | {html.escape(result["backend"]["name"])} | {len(rows)} steps | '
                        f'<a href="{sample_id}/generation.json">Generation JSON</a> | '
                        f'<a href="{sample_id}/expert_result.json">Expert evidence</a></p>'
                        f'{"".join(rows)}</article>')
        manifest.append({"run_id": sample_id, "task": result["task_name"],
                         "scene": result["scene"], "source": str(result_path.resolve()),
                         "accepted": True, "backend": result["backend"]})
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Env-B Expert Examples</title>
<style>body{font:15px system-ui;margin:24px;color:#202623;background:#f8faf9}
h1{font-size:28px}h2{font-size:23px}h3{font-size:18px}h4{font-size:14px}
article{border-top:3px solid #287b63;margin-top:36px}section{border-top:1px solid #bac8c0;padding:12px 0}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
figure{margin:0}img{width:100%;height:auto;aspect-ratio:4/3;object-fit:contain}
figcaption{padding:6px 0;overflow-wrap:anywhere}pre{white-space:pre-wrap}
@media(max-width:800px){.views{grid-template-columns:1fr}body{margin:12px}}
</style><h1>Env-B Expert Examples</h1>
<p>Original simulation RGB, all saved action pre/post observations, robot and all global cameras.
Symbolic expert with official state transitions. These are discrete observations, not continuous physical motion.</p>
<p>Fire: USDZ flame visualization, official OnFire extinction. Dishes: official Covered stain removal;
stain contrast is weak in some views. Neither example demonstrates physically controlled tool contact.</p>'''
    page += '<p>Expert acceptance alone does not certify realistic household storage. '
    page += 'Inspect the attached generation JSON for household_layout distance evidence; older samples may lack it.</p>'
    (output / "index.html").write_text(page + "".join(sections) + "</html>")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_report(args.result, args.output), indent=2))
