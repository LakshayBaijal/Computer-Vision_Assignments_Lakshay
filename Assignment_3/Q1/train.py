import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
import traceback

import detection
from detection.faster_rcnn import FastRCNNPredictor, FastRCNNPredictorWithAngle
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error loading config: {exc}")
            return
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    # Override root_dir if provided
    if args.root_dir:
        dataset_config['root_dir'] = args.root_dir

    # Validate dataset path exists
    if not os.path.exists(dataset_config['root_dir']):
        print(f"Error: Dataset path does not exist: {dataset_config['root_dir']}")
        return

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    print(f"Loading dataset from: {dataset_config['root_dir']}")
    st = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    print(f"Dataset loaded with {len(st)} images")

    train_dataset = DataLoader(st,
                               batch_size=4,
                               shuffle=True,
                               num_workers=0,
                               collate_fn=collate_function)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                        min_size=600,
                                                        max_size=1000,
    )
    
    # Use angle version if specified
    use_angle = train_config.get('use_angle', False)
    print(f"Using angle prediction: {use_angle}")
    
    if use_angle:
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictorWithAngle(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=dataset_config['num_classes'])
        bbox_reg_weights = tuple(model_config.get('bbox_reg_weights', [10.0, 10.0, 5.0, 5.0, 1.0]))
        faster_rcnn_model.roi_heads.box_coder = detection._utils.BoxCoderWithAngle(bbox_reg_weights)
    else:
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=dataset_config['num_classes'])

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    
    # Create checkpoint directory
    task_dir = train_config['task_name']
    os.makedirs(task_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {task_dir}")

    lr = train_config.get('lr', 1E-4)
    print(f"Effective learning rate: {lr}")
    optimizer = torch.optim.SGD(lr=lr,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        
        try:
            for ims, targets, _ in tqdm(train_dataset):
                try:
                    optimizer.zero_grad()
                    for target in targets:
                        bboxes = target['bboxes'].float().to(device)
                        target['boxes'] = bboxes[:, :4]
                        if use_angle and bboxes.shape[-1] == 5:
                            target['angles'] = bboxes[:, 4]
                        del target['bboxes']
                        target['labels'] = target['labels'].long().to(device)
                    images = [im.float().to(device) for im in ims]
                    batch_losses = faster_rcnn_model(images, targets)
                    
                    if batch_losses is None:
                        print("Warning: batch_losses is None, skipping batch")
                        continue
                    
                    loss = batch_losses['loss_classifier']
                    loss += batch_losses['loss_box_reg']
                    loss += batch_losses['loss_rpn_box_reg']
                    loss += batch_losses['loss_objectness']

                    rpn_classification_losses.append(batch_losses['loss_objectness'].item())
                    rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
                    frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
                    frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

                    loss.backward()
                    optimizer.step()
                    step_count += 1
                except Exception as e:
                    print(f"Error in batch: {e}")
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"Error in epoch {i}: {e}")
            traceback.print_exc()
            continue
            
        print('Finished epoch {}'.format(i))
        ckpt_path = os.path.join(task_dir, 'tv_frcnn_r50fpn_' + train_config['ckpt_name'])
        torch.save(faster_rcnn_model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
        
        loss_output = ''
        if rpn_classification_losses:
            loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        if rpn_localization_losses:
            loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        if frcnn_classification_losses:
            loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        if frcnn_localization_losses:
            loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    parser.add_argument('--root_dir', dest='root_dir',
                        default=None, type=str, help='Root directory for dataset')
    args = parser.parse_args()
    train(args)