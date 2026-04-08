# Q1 Report: Faster R-CNN Visualizations and Oriented Bounding Boxes

## Visualizations for Faster R-CNN

Place your visual outputs under `visualize_outputs` using the exact directory and file names below. We expect outputs for exactly two input images (`img_1` and `img_2`). You may change visualization styles but do not change the filenames or directory structure.

```
visualize_outputs
├── bb_assignments
│   ├── img_1.png
│   └── img_2.png
├── object_proposals
│   ├── img_1.gif
│   └── img_2.gif
├── objectness
│   ├── img_1.gif
│   └── img_2.gif
└── roi_head_outputs
   ├── img_1.gif
   └── img_2.gif
```

### Hyperparameters

Document the two hyperparameter sets used to generate the visualizations. Keep each set as a two-column table.

Hyperparameter set 1:

| Variable        | Value   |
| --------------- | ------- |
| <variable_name> | <value> |

Hyperparameter set 2:

| Variable        | Value   |
| --------------- | ------- |
| <variable_name> | <value> |

## Extending Faster R-CNN for Oriented Bounding Boxes

Place oriented-bbox outputs under `oriented_bbox_results` with these files:

```
oriented_bbox_results
├── qualitative_results.png
└── training_curves.png
```

- `training_curves.png`: plots of Precision, Recall and mAP across training epochs.
- `qualitative_results.png`: 6 validation images showing predicted oriented bounding boxes overlaid on ground truth.

### Evaluation tables

Provide mAP, mean precision and mean recall for oriented bounding boxes under the two setups below.

1) Theta predicted via regression — report mAP at multiple IoU thresholds:

| IoU threshold |  mAP  | Mean Precision | Mean Recall |
| ------------: | :---: | :------------: | :---------: |
|           0.5 |       |                |             |
|           0.7 |       |                |             |
|           0.9 |       |                |             |

2) Theta discretized (classification) — report at IoU = 0.5 for several bin counts:

| Total bins for theta |  mAP  | Mean Precision | Mean Recall |
| -------------------: | :---: | :------------: | :---------: |
|                    6 |       |                |             |
|                   12 |       |                |             |
|                   14 |       |                |             |

Fill in the numeric entries above with your final evaluation results.
