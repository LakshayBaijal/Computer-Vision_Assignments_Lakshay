from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
from tqdm import tqdm

import infer


def evaluate_precision_recall(preds, gts, iou_threshold=0.5, score_threshold=0.5, use_angle=False):
    iou_fn = infer.get_rotated_iou if use_angle else infer.get_iou

    tp = 0
    fp = 0
    fn = 0

    for im_idx in range(len(preds)):
        pred_boxes = [p for p in preds[im_idx]["text"] if p[-1] >= score_threshold]
        gt_boxes = gts[im_idx]["text"]

        matched = [False] * len(gt_boxes)

        for pred in pred_boxes:
            pbox = pred[:-1]
            best_iou = 0.0
            best_gt = -1
            for gt_idx, gt in enumerate(gt_boxes):
                iou = iou_fn(pbox[:5] if use_angle else pbox[:4], gt[:5] if use_angle else gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= iou_threshold and best_gt >= 0 and not matched[best_gt]:
                tp += 1
                matched[best_gt] = True
            else:
                fp += 1

        fn += sum(1 for flag in matched if not flag)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    load_args = types.SimpleNamespace(config_path=args.config, checkpoint_path=args.checkpoint, root_dir=None)
    model, dataset, _, use_angle = infer.load_model_and_dataset(load_args)

    preds = []
    gts = []

    for idx in tqdm(range(len(dataset)), desc="Collecting predictions"):
        im, target, _ = dataset[idx]
        out = model(im.unsqueeze(0).float().to(infer.device), None)[0]

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

    iou_thresholds = [0.5, 0.7, 0.9]
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "use_angle": bool(use_angle),
        "score_threshold_for_pr": 0.5,
        "metrics": {},
    }

    for iou_thr in iou_thresholds:
        mAP, _ = infer.compute_map(preds, gts, iou_threshold=iou_thr, method="interp", use_angle=use_angle)
        precision, recall = evaluate_precision_recall(
            preds, gts, iou_threshold=iou_thr, score_threshold=0.5, use_angle=use_angle
        )
        result["metrics"][str(iou_thr)] = {
            "mAP": float(mAP),
            "mean_precision": float(precision),
            "mean_recall": float(recall),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved metrics: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
