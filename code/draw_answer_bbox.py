#!/usr/bin/env python
"""把每道选择题「正确答案的 bounding box」叠加绘制到机器人主视角 RGB 图上的检测脚本。

用途: 人工核验 bbox 题 (以及带 target bbox 的规划题) 的正确答案确实框住了目标物体,
      即检查「答案里的坐标」与「机器人主视角画面中的真实物体」是否对齐。

坐标约定 (与 translator.py 保持一致):
- bbox 题 (question_type == "bbox"):
    ``A.bbox_xyxy`` 是归一化到 [0, BBOX_NORM_SCALE=1000] 的 [x1, y1, x2, y2];
    ``A.image_size`` 是原始图像 [W, H]。绘制时按 ``pixel = norm / 1000 * size`` 反归一化。
- 规划题 (question_type == "planning"):
    ``A.bbox`` 是机器人主视角的原始像素 [x1, y1, x2, y2] (目标物体不可见时为 None)。

输出: 在 out_dir 下为每个含 bbox 的样本生成 ``<qra_id>.png`` (绿框 + 物体名 + 答案坐标),
      并打印汇总 (含越界/缺图等异常检测)。

运行:
    /home2/jiaodian/anaconda3/envs/behavior_new/bin/python code/draw_answer_bbox.py \
        --qra generate_dataset/online_env_a_0000_task/qra.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# 与 translator.py 的 BBOX_NORM_SCALE 一致
BBOX_NORM_SCALE = 1000
BOX_COLOR = (0, 230, 0)          # 绿框 (RGB)
BOX_WIDTH = 3
LABEL_BG = (0, 0, 0)             # 标签底色
LABEL_FG = (0, 230, 0)


def _humanize(name: str | None) -> str:
    """id 风格名 (下划线分隔) → 人类可读 (空格分隔)。"""
    return (name or "").replace("_", " ").strip()


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    """加载一个可用的等宽字体, 失败则退回 PIL 默认字体。"""
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def denormalize_bbox(norm_bbox: list[Any], image_size: list[Any],
                     norm_scale: int = BBOX_NORM_SCALE) -> list[int]:
    """归一化 bbox [0,1000] → 原始像素 bbox。image_size 为 [W, H]。"""
    W, H = int(image_size[0]), int(image_size[1])
    x1, y1, x2, y2 = (float(v) for v in norm_bbox)
    return [
        int(round(x1 / norm_scale * W)),
        int(round(y1 / norm_scale * H)),
        int(round(x2 / norm_scale * W)),
        int(round(y2 / norm_scale * H)),
    ]


def correct_answer_bbox(sample: dict[str, Any]) -> tuple[list[int], str, str] | None:
    """从样本中抽取正确答案的像素 bbox, 返回 (pixel_bbox, label, coord_text) 或 None。

    coord_text 为「答案里的坐标」字符串 (bbox 题=归一化坐标, 规划题=像素坐标),
    用于叠印在图上供人工对照选项文本。
    """
    qt = sample.get("question_type")
    A = sample.get("A") or {}
    if qt == "bbox":
        if not A.get("visible"):
            return None
        norm = A.get("bbox_xyxy")
        size = A.get("image_size")
        if not (isinstance(norm, list) and len(norm) == 4
                and isinstance(size, list) and len(size) == 2):
            return None
        pixel = denormalize_bbox(norm, size)
        label = _humanize(A.get("category") or A.get("object_id") or "object")
        coord_text = f"[{norm[0]}, {norm[1]}, {norm[2]}, {norm[3]}]"
        return pixel, label, coord_text
    if qt == "planning":
        bbox = A.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            pixel = [int(v) for v in bbox]
            label = _humanize(A.get("category") or A.get("target_object") or "target")
            coord_text = f"[{pixel[0]}, {pixel[1]}, {pixel[2]}, {pixel[3]}]"
            return pixel, label, coord_text
    return None


def find_robot_rgb(sample: dict[str, Any], base_dir: Path) -> Path | None:
    """定位机器人主视角 rgb 图 (sample.images 中 view == "robot_primary" 的条目)。"""
    for img in sample.get("images") or []:
        if img.get("view") == "robot_primary":
            rel = img.get("rgb")
            if rel:
                p = base_dir / rel
                if p.exists():
                    return p
    return None


def draw_bbox(img: Image.Image, bbox: list[int], label: str,
              coord_text: str, font: ImageFont.ImageFont) -> Image.Image:
    """在原图副本上绘制绿框 + 左上角标签 (物体名 + 答案坐标)。"""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    # 规范化顺序, 兼容负值/倒序 bbox
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    draw.rectangle([xa, ya, xb, yb], outline=BOX_COLOR, width=BOX_WIDTH)

    # 左上角标签: 两行 (物体名 / 坐标)
    text = f"{label}\nbbox {coord_text}"
    tw = 0
    lines = text.split("\n")
    for ln in lines:
        tw = max(tw, draw.textlength(ln, font=font))
    th = len(lines) * (font.size + 4)
    ty = max(0, ya - th - 6)          # 框上方; 贴边时回落到框内
    if ty < 2:
        ty = ya + 2
    draw.rectangle([xa, ty, xa + tw + 8, ty + th + 6], fill=LABEL_BG)
    draw.text((xa + 4, ty + 3), text, fill=LABEL_FG, font=font)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="把正确答案 bbox 叠加到机器人主视角图上")
    ap.add_argument("--qra", required=True, help="qra.json 路径")
    ap.add_argument("--out", default=None, help="输出目录 (默认 <qra 目录>/annotated_bbox)")
    args = ap.parse_args()

    qra_path = Path(args.qra).resolve()
    if not qra_path.exists():
        print(f"[error] 找不到 qra.json: {qra_path}")
        return 2
    base_dir = qra_path.parent
    out_dir = Path(args.out) if args.out else base_dir / "annotated_bbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads(qra_path.read_text(encoding="utf-8"))
    samples = doc.get("samples") or []
    font = _load_font(18)

    n_bbox = 0
    n_drawn = 0
    n_skip_no_image = 0
    n_out_of_bounds = 0
    for s in samples:
        bbox = correct_answer_bbox(s)
        if bbox is None:
            continue
        n_bbox += 1
        pixel, label, coord_text = bbox

        img_path = find_robot_rgb(s, base_dir)
        if img_path is None:
            n_skip_no_image += 1
            print(f"[skip] 无 robot_primary 图: {s.get('qra_id')}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(f"[skip] 读图失败 {img_path}: {e}")
            n_skip_no_image += 1
            continue

        W, H = img.size
        x1, y1, x2, y2 = pixel
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            n_out_of_bounds += 1
            print(f"[warn] bbox 越界 (image {W}x{H}): {s.get('qra_id')} {pixel}")

        annotated = draw_bbox(img, pixel, label, coord_text, font)
        out_path = out_dir / f"{s.get('qra_id')}.png"
        annotated.save(out_path)
        n_drawn += 1

    print(f"\n样本数: {len(samples)} | 含正确答案 bbox 的样本: {n_bbox}")
    print(f"已绘制: {n_drawn} | 缺图跳过: {n_skip_no_image} | bbox 越界: {n_out_of_bounds}")
    print(f"输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
