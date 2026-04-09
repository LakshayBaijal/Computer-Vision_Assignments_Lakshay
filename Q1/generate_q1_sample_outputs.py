from __future__ import annotations

import argparse
import types
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import infer


def make_dirs(base_dir: Path) -> Dict[str, Path]:
    viz = base_dir / "visualize_outputs"
    dirs = {
        "bb": viz / "bb_assignments",
        "objness": viz / "objectness",
        "proposals": viz / "object_proposals",
        "roi": viz / "roi_head_outputs",
        "obb": base_dir / "oriented_bbox_results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def draw_box(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
    if len(box) >= 5:
        x1, y1, x2, y2, theta = box[:5]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(theta))
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    else:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def get_best_iou(pred_box: np.ndarray, gt_boxes: np.ndarray, use_angle: bool) -> float:
    if len(gt_boxes) == 0:
        return 0.0
    best = 0.0
    iou_fn = infer.get_rotated_iou if use_angle and len(pred_box) >= 5 and gt_boxes.shape[1] >= 5 else infer.get_iou
    for gt in gt_boxes:
        val = float(iou_fn(pred_box[:5] if iou_fn == infer.get_rotated_iou else pred_box[:4], gt[:5] if iou_fn == infer.get_rotated_iou else gt[:4]))
        if val > best:
            best = val
    return best


def save_gif(frames_bgr: List[np.ndarray], out_path: Path, duration_ms: int = 650):
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def load_run(config_path: Path, checkpoint_path: Path):
    args = types.SimpleNamespace(config_path=str(config_path), checkpoint_path=str(checkpoint_path), root_dir=None)
    model, dataset, test_loader, use_angle = infer.load_model_and_dataset(args)
    return model, dataset, test_loader, use_angle


def predict_single(model, dataset, idx: int):
    im_tensor, target, im_path = dataset[idx]
    inp = im_tensor.unsqueeze(0).float().to(infer.device)
    with torch.no_grad():
        out = model(inp, None)[0]
    img = cv2.imread(im_path)
    return img, target, out, im_path


def generate_visualize_outputs(run_model, dataset, use_angle: bool, dirs: Dict[str, Path]):
    indices = [0, 1]
    for out_idx, ds_idx in enumerate(indices, start=1):
        img, target, out, _ = predict_single(run_model, dataset, ds_idx)
        gt_boxes = target["bboxes"].detach().cpu().numpy()

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        # 1) bb_assignments (GT + proposal match quality)
        bb_img = img.copy()
        for gt in gt_boxes:
            draw_box(bb_img, gt, (0, 255, 0), 2)
        top_n = min(25, len(boxes))
        for b in boxes[:top_n]:
            iou = get_best_iou(b, gt_boxes, use_angle)
            color = (255, 0, 0) if iou >= 0.5 else (0, 0, 255)
            draw_box(bb_img, b, color, 1)
        cv2.putText(bb_img, "Green=GT, Blue=matched, Red=unmatched", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(dirs["bb"] / f"img_{out_idx}.png"), bb_img)

        # 2) object_proposals gif (top-k growth)
        proposal_frames = []
        for k in [5, 15, 30, 60]:
            frame = img.copy()
            for b in boxes[: min(k, len(boxes))]:
                draw_box(frame, b, (0, 255, 255), 2)
            cv2.putText(frame, f"Top-{k} proposals", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            proposal_frames.append(frame)
        save_gif(proposal_frames, dirs["proposals"] / f"img_{out_idx}.gif")

        # 3) objectness gif (score thresholds)
        obj_frames = []
        for thr in [0.9, 0.7, 0.5, 0.3]:
            frame = img.copy()
            keep = np.where(scores >= thr)[0]
            for i in keep[:80]:
                draw_box(frame, boxes[i], (255, 255, 0), 2)
            cv2.putText(frame, f"Objectness >= {thr:.1f} | count={len(keep)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            obj_frames.append(frame)
        save_gif(obj_frames, dirs["objness"] / f"img_{out_idx}.gif")

        # 4) roi_head_outputs gif (final detections by threshold)
        roi_frames = []
        for thr in [0.95, 0.85, 0.7, 0.5]:
            frame = img.copy()
            keep = np.where(scores >= thr)[0]
            for i in keep[:50]:
                draw_box(frame, boxes[i], (0, 0, 255), 2)
            cv2.putText(frame, f"RoI output score >= {thr:.2f} | count={len(keep)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            roi_frames.append(frame)
        save_gif(roi_frames, dirs["roi"] / f"img_{out_idx}.gif")


def collect_predictions(model, dataset, use_angle: bool):
    preds = []
    gts = []
    for idx in tqdm(range(len(dataset)), desc="Collecting eval predictions"):
        _, target, out, _ = predict_single(model, dataset, idx)
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        gt_boxes = target["bboxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        pred_map = {"text": []}
        gt_map = {"text": []}

        for i in range(len(boxes)):
            if labels[i] != 1:
                continue
            if use_angle and len(boxes[i]) >= 5:
                pred_map["text"].append([*boxes[i][:5].tolist(), float(scores[i])])
            else:
                pred_map["text"].append([*boxes[i][:4].tolist(), float(scores[i])])

        for i in range(len(gt_boxes)):
            if gt_labels[i] != 1:
                continue
            if use_angle and len(gt_boxes[i]) >= 5:
                gt_map["text"].append(gt_boxes[i][:5].tolist())
            else:
                gt_map["text"].append(gt_boxes[i][:4].tolist())

        preds.append(pred_map)
        gts.append(gt_map)
    return preds, gts


def precision_recall_for_threshold(preds, gts, score_thr: float, use_angle: bool, iou_thr: float = 0.5):
    filtered = []
    for im_pred in preds:
        cur = {"text": [p for p in im_pred["text"] if p[-1] >= score_thr]}
        filtered.append(cur)

    tp = 0
    fp = 0
    fn = 0
    iou_fn = infer.get_rotated_iou if use_angle else infer.get_iou

    for im_idx, im_pred in enumerate(filtered):
        pred_boxes = im_pred["text"]
        gt_boxes = gts[im_idx]["text"]
        matched = [False] * len(gt_boxes)

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt = -1
            for gt_i, gt in enumerate(gt_boxes):
                pb = pred[:-1]
                val = iou_fn(pb[:5] if use_angle else pb[:4], gt[:5] if use_angle else gt[:4])
                if val > best_iou:
                    best_iou = val
                    best_gt = gt_i
            if best_iou >= iou_thr and best_gt >= 0 and not matched[best_gt]:
                tp += 1
                matched[best_gt] = True
            else:
                fp += 1

        fn += sum(1 for m in matched if not m)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    mean_ap, _ = infer.compute_map(filtered, gts, iou_threshold=iou_thr, method="interp", use_angle=use_angle)
    return precision, recall, mean_ap


def generate_oriented_outputs(model, dataset, use_angle: bool, dirs: Dict[str, Path]):
    # qualitative_results.png (6 samples)
    sample_ids = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, sample_ids):
        img, target, out, _ = predict_single(model, dataset, idx)
        gt_boxes = target["bboxes"].detach().cpu().numpy()
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()

        canvas = img.copy()
        for gt in gt_boxes:
            draw_box(canvas, gt, (0, 255, 0), 2)
        keep = np.where(scores >= 0.5)[0][:20]
        for i in keep:
            draw_box(canvas, boxes[i], (0, 0, 255), 2)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        ax.imshow(canvas_rgb)
        ax.set_title(f"Sample {idx} (Green=GT, Red=Pred)")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(dirs["obb"] / "qualitative_results.png", dpi=180)
    plt.close(fig)

    # training_curves.png (evaluation curves across score thresholds)
    preds, gts = collect_predictions(model, dataset, use_angle)
    thresholds = np.linspace(0.1, 0.9, 9)
    precisions = []
    recalls = []
    maps = []
    for thr in thresholds:
        p, r, m = precision_recall_for_threshold(preds, gts, float(thr), use_angle=use_angle, iou_thr=0.5)
        precisions.append(p)
        recalls.append(r)
        maps.append(m)

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="o", label="Recall")
    plt.plot(thresholds, maps, marker="o", label="mAP@0.5")
    plt.xlabel("Score threshold")
    plt.ylabel("Metric value")
    plt.title("OBB Evaluation Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirs["obb"] / "training_curves.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Q1 sample-results style outputs locally")
    parser.add_argument("--base_dir", type=str, default="sample results/q1")
    parser.add_argument("--aabb_config", type=str, default="config/run1_aabb_hp1.yaml")
    parser.add_argument("--aabb_ckpt", type=str, default="Runs/Q1_RUN1_AABB_HP1_EXPORT/Q1_RUN1_AABB_HP1_final.pth")
    parser.add_argument("--obb_config", type=str, default="config/st.yaml")
    parser.add_argument("--obb_ckpt", type=str, default="Runs/Q1_RUN5_OBB_MULTIBIN60_EXPORT/Q1_RUN5_OBB_MULTIBIN60_final_20260405_153836.pth")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    base_dir = (root.parent / args.base_dir).resolve() if not Path(args.base_dir).is_absolute() else Path(args.base_dir)
    dirs = make_dirs(base_dir)

    print(f"Generating visualize_outputs in: {base_dir}")

    aabb_model, aabb_dataset, _, aabb_use_angle = load_run(root / args.aabb_config, root / args.aabb_ckpt)
    generate_visualize_outputs(aabb_model, aabb_dataset, aabb_use_angle, dirs)

    obb_model, obb_dataset, _, obb_use_angle = load_run(root / args.obb_config, root / args.obb_ckpt)
    generate_oriented_outputs(obb_model, obb_dataset, obb_use_angle, dirs)

    print("Done. Generated files:")
    for p in sorted(base_dir.rglob("*")):
        if p.is_file():
            print(p)


if __name__ == "__main__":
    main()
