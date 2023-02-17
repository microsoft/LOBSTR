"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import json
from datetime import datetime
import sys
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy

sys.path.append("../../lobstr")
from detr.engine import train_one_epoch
from lobstr import build as build_model
import detr.util.misc as utils
import detr.datasets.transforms as R

from datasets import get_transform
from datasets import ScatterPlotDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir',
                        required=True,
                        help="Root data directory for images and labels")
    parser.add_argument('--config_file',
                        required=True,
                        help="Filepath to the config containing the args")
    parser.add_argument('--backbone',
                        default='resnet18',
                        help="Backbone for the model")
    parser.add_argument('--model_load_path', help="The path to trained model")
    parser.add_argument('--load_weights_only', action='store_true')
    parser.add_argument('--model_save_dir', help="The output directory for saving model params and checkpoints")
    parser.add_argument('--debug_save_dir',
                        help='Filepath to save visualizations',
                        default='debug')                        
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        help="Modes: training (train) and evaluation (eval)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--lr_gamma', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--checkpoint_freq', default=1, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--train_max_size', type=int)
    parser.add_argument('--val_max_size', type=int)
    parser.add_argument('--test_max_size', type=int)

    return parser.parse_args()



def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training, 
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")

    if args.mode == "train":
        dataset_train = ScatterPlotDataset(os.path.join(args.data_root_dir, "train"),
                                           transforms=get_transform("train"), do_crop=False)
        dataset_val = ScatterPlotDataset(os.path.join(args.data_root_dir, "val"),
                                         transforms=get_transform("val_no_norm"), do_crop=False)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                            args.batch_size,
                                                            drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn,
                                       num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val,
                                     2 * args.batch_size,
                                     sampler=sampler_val,
                                     drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers)
        return data_loader_train, data_loader_val, dataset_val, len(
            dataset_train)

    elif args.mode == "eval":
        dataset_test = ScatterPlotDataset(os.path.join(args.data_root_dir, "test"),
                                           transforms=get_transform("test"), do_crop=False)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = DataLoader(dataset_test,
                                      2 * args.batch_size,
                                      sampler=sampler_test,
                                      drop_last=False,
                                      collate_fn=utils.collate_fn,
                                      num_workers=args.num_workers)
        return data_loader_test, dataset_test


def get_model(args, device):
    """
    Loads LOBSTR model for scatter plots on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path,
                                       map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def score_instance(gt, outputs, img_size, window=0.01):
    T = window * min(img_size)
    
    boxes = outputs['pred_boxes']
    m = outputs['pred_node_logits'].softmax(-1).max(-1)
    scores = m.values[0].tolist()
    labels = m.indices[0].tolist()
    pred_rescaled_boxes = rescale_bboxes(boxes[0].cpu(), img_size)
    pred_data_point_idxs = [idx for idx, (label, score) in enumerate(zip(labels, scores)) if label == 1 and score > 0.5]
    pred_rescaled_boxes = pred_rescaled_boxes.cpu().tolist()
    pred_rescaled_coords = [[(elem[0]+elem[2])/2, (elem[1]+elem[3])/2] for elem in pred_rescaled_boxes]
    pred_data_point_xy = [pred_rescaled_coords[idx] for idx in pred_data_point_idxs]

    gt_boxes = gt['boxes'].cpu().tolist()
    gt_labels = gt['labels'].cpu().tolist()
    gt_data_point_idxs = [idx for idx, label in enumerate(gt_labels) if label == 1]
    gt_coords = [[(elem[0]+elem[2])/2, (elem[1]+elem[3])/2] for elem in gt_boxes]
    gt_data_point_xy = [gt_coords[idx] for idx in gt_data_point_idxs]
    
    C = np.zeros((len(gt_data_point_xy), len(pred_data_point_xy)))
    C2 = np.zeros((len(gt_data_point_xy), len(pred_data_point_xy)))
    for gt_idx, gt_xy in enumerate(gt_data_point_xy):
        for pred_idx, pred_xy in enumerate(pred_data_point_xy):
            D = ((gt_xy[0]-pred_xy[0])**2 + (gt_xy[1]-pred_xy[1])**2) ** 0.5
            cost = max(0, 1 - (D/T)**2)
            C[gt_idx, pred_idx] = cost
            C2[gt_idx, pred_idx] = D
    

    gt_idxs, pred_idxs = scipy.optimize.linear_sum_assignment(C, maximize=True)
    total_cost = 0
    for gt_idx, pred_idx in zip(gt_idxs, pred_idxs):
        total_cost += C[gt_idx, pred_idx]
    chart_score = total_cost / max(len(gt_data_point_idxs), len(pred_data_point_idxs))

    gt_idxs, pred_idxs = scipy.optimize.linear_sum_assignment(C2, maximize=False)
    true_pos = 0
    for gt_idx, pred_idx in zip(gt_idxs, pred_idxs):
         if C2[gt_idx, pred_idx] <= T:
                true_pos += 1

    if len(gt_data_point_idxs) > 0:
        if len(pred_data_point_idxs) == 0:
            noreplace_recall = 0
            replace_recall = 0
        else:
            noreplace_recall = true_pos / len(gt_data_point_idxs)
            replace_recall = np.sum(np.min(C2, 1) <= T) / len(gt_data_point_idxs)
    else:
        noreplace_recall = 1
        replace_recall = 1

    if len(pred_data_point_idxs) > 0:
        if len(gt_data_point_idxs) == 0:
            noreplace_precision = 0
            replace_precision = 0
        else:
            noreplace_precision = true_pos / len(pred_data_point_idxs)
            replace_precision = np.sum(np.min(C2, 0) <= T) / len(pred_data_point_idxs)
    else:
        noreplace_precision = 1
        replace_precision = 1

    return chart_score, noreplace_recall, noreplace_precision, replace_recall, replace_precision, len(gt_data_point_idxs)


def evaluate_model(model, dataset_test, args, max_size=None, window=0.01):
    model.eval()

    chart_scores = []
    noreplace_recalls = []
    noreplace_precisions = []
    replace_recalls = []
    replace_precisions = []
    num_data_points = []
    
    if max_size is None:
        max_size = len(dataset_test)
    else:
        max_size = min(max_size, len(dataset_test))

    for idx in range(max_size):
        img, gt = dataset_test[idx]
        img_tensor = normalize(img)

        # propagate through the model
        with torch.no_grad():
            outputs = model([img_tensor.to(args.device)])

        (chart_score, noreplace_recall,
         noreplace_precision, replace_recall,
         replace_precision, num_data_point) = score_instance(gt, outputs, img.size,
                                                             window=window)

        chart_scores.append(chart_score)
        noreplace_recalls.append(noreplace_recall)
        noreplace_precisions.append(noreplace_precision)
        replace_recalls.append(replace_recall)
        replace_precisions.append(replace_precision)
        num_data_points.append(num_data_point)
        
    return chart_scores, noreplace_recalls, noreplace_precisions, replace_recalls, replace_precisions, num_data_points


def train(args, model, criterion, postprocessors, device):
    """
    Training loop
    """
    print("loading data")
    dataloading_time = datetime.now()
    data_loader_train, data_loader_val, dataset_val, train_len = get_data(args)
    print("finished loading data in :", datetime.now() - dataloading_time)

    model_without_ddp = model
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_drop,
                                                   gamma=args.lr_gamma)

    max_batches_per_epoch = int(train_len / args.batch_size)
    print("Max batches per epoch: {}".format(max_batches_per_epoch))

    resume_checkpoint = False
    if args.model_load_path:
        checkpoint = torch.load(args.model_load_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)

        if not args.load_weights_only and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resume_checkpoint = True
        elif args.load_weights_only:
            print("*** WARNING: Resuming training and ignoring optimzer state. "
                  "Training will resume with new initialized values. "
                  "To use current optimizer state, remove the --load_weights_only flag.")
        else:
            print("*** ERROR: Optimizer state of saved checkpoint not found. "
                  "To resume training with new initialized values add the --load_weights_only flag.")
            raise Exception("ERROR: Optimizer state of saved checkpoint not found. Must add --load_weights_only flag to resume training without.")          
        
        if not args.load_weights_only and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        elif args.load_weights_only:
            print("*** WARNING: Resuming training and ignoring previously saved epoch. "
                  "To resume from previously saved epoch, remove the --load_weights_only flag.")
        else:
            print("*** WARNING: Epoch of saved model not found. Starting at epoch {}.".format(args.start_epoch))

    # Use user-specified save directory, if specified
    if args.model_save_dir:
        output_directory = args.model_save_dir
    # If resuming from a checkpoint with optimizer state, save into same directory
    elif args.model_load_path and resume_checkpoint:
        output_directory = os.path.split(args.model_load_path)[0]
    # Create new save directory
    else:
        run_date = datetime.now().strftime("%Y%m%d%H%M%S")
        output_directory = os.path.join(args.data_root_dir, "output", run_date)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Output directory: ", output_directory)
    model_save_path = os.path.join(output_directory, 'model.pth')
    print("Output model path: ", model_save_path)
    if not resume_checkpoint and os.path.exists(model_save_path):
        print("*** WARNING: Output model path exists but is not being used to resume training; training will overwrite it.")

    if args.start_epoch >= args.epochs:
        print("*** WARNING: Starting epoch ({}) is greater or equal to the number of training epochs ({}).".format(
            args.start_epoch, args.epochs
        ))

    print("Start training")
    start_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        print('-' * 100)

        epoch_timing = datetime.now()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            max_batches_per_epoch=max_batches_per_epoch,
            print_freq=1000)
        print("Epoch completed in ", datetime.now() - epoch_timing)

        lr_scheduler.step()

        # Save current model training progress
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_save_path)

        # Save checkpoint for evaluation
        if (epoch+1) % args.checkpoint_freq == 0:
            model_save_path_epoch = os.path.join(output_directory, 'model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), model_save_path_epoch)

        (chart_scores, noreplace_recalls,
         noreplace_precisions, replace_recalls,
         replace_precisions, num_data_points) = evaluate_model(model, dataset_val, args, max_size=250, window=0.01)

        print("Mean CHART Score: {}".format(np.mean(chart_scores)))
        print("Mean No-Replace Recall: {}".format(np.mean(noreplace_recalls)))
        print("Mean No-Replace Precision: {}".format(np.mean(noreplace_precisions)))
        print("Mean Replace Recall: {}".format(np.mean(replace_recalls)))
        print("Mean Replace Precision: {}".format(np.mean(replace_precisions)))

    print('Total training time: ', datetime.now() - start_time)


def main():
    cmd_args = get_args().__dict__
    config_args = json.load(open(cmd_args['config_file'], 'rb'))
    for key, value in config_args.items():
        if 'emphasized' in key:
            print(key)
            value_temp = {int(k): v for k, v in value}
            value = defaultdict(lambda: 1)
            value.update(value_temp)
            config_args[key] = value
    for key, value in cmd_args.items():
        if not key in config_args or not value is None:
            config_args[key] = value
    #config_args.update(cmd_args)
    args = type('Args', (object,), config_args)
    print(args.__dict__)
    print('-' * 100)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("loading model")
    device = torch.device(args.device)
    model, criterion, postprocessors = get_model(args, device)

    if args.mode == "train":
        train(args, model, criterion, postprocessors, device)
    elif args.mode == "eval":
        print("Not supported yet.")


if __name__ == "__main__":
    main()
