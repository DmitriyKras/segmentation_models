import glob
import os
import yaml
import json
from utils.dataloader import SegmentationDataset
import argparse
from torch.utils.data import DataLoader
import torch
from math import ceil
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
# import metrics
import torch.nn.functional as F
from segmentation_models_pytorch.metrics import get_stats, precision, recall, iou_score
from torchmetrics.classification import BinaryAveragePrecision
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
# import models
from models.effunet import EfficientUnet
from models.bisenet import BiSeNet
from models.liteseg_mobilenet import LiteSegMobileNetV2
from models.liteseg_shufflenet import LiteSegShuffleNet
from models.esnet import ESNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, bisenet, 
                        liteseg_mobilenet, liteseg_shufflenet, esnet""")
    parser.add_argument('--batch-size', type=int, default=20, 
                        help='batch size')
    parser.add_argument('--img', type=int, default=512, 
                        help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--task', type=str, default='val', 
                        help='task of script usage: val or test')
    parser.add_argument('--weights', type=str, default='weights/effunet/best.pt',
                        help='path to weight file to validate')
    args = parser.parse_args()

    # create folders if not exist
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists(f'weights/{args.model}'):
        os.makedirs(f'weights/{args.model}')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists(f'logs/{args.model}'):
        os.makedirs(f'logs/{args.model}')
    # open data config
    with open(args.data, 'r') as f:
        data = yaml.full_load(f)
    # get images and masks, configure datasets and dataloaders
    assert args.task == 'val' or args.task == 'test', f'Task {args.task} is not supported. Choose val or test'
    images = sorted(list(glob.glob(data['path'] + data[args.task] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data[args.task].replace('images', 'masks') + '/*.jpg')))
    val_dataset = SegmentationDataset(images, masks, (args.img, args.img))
    val_dataloader = DataLoader(val_dataset, args.batch_size, num_workers=2)
    ### GET MODEL ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
    if torch.cuda.is_available():
        print('[INFO] Using CUDA device')
    else:
        print('[INFO] CUDA is not detected. Using CPU')
    if args.model == 'effunet':
        model = EfficientUnet().to(device)
    elif args.model == 'bisenet':
        model = BiSeNet(1, 'resnet18').to(device)
    elif args.model == 'liteseg_mobilenet':
        model = LiteSegMobileNetV2(1, pretrained=False).to(device)
    elif args.model == 'liteseg_shufflenet':
        model = LiteSegShuffleNet(1, pretrained=False).to(device)
    elif args.model == 'esnet':
        model = ESNet(1).to(device)
    else:
        print(f'[INFO] Model {args.model} is not supported. Using default effunet')
        model = EfficientUnet().to(device)
    
    model.load_state_dict(torch.load(args.weights))
    ### INIT METRICS ###
    history = {}  # result
    val_steps = ceil(len(val_dataset) / args.batch_size)
    bap = BinaryAveragePrecision(thresholds=50, validate_args=False)

    print(f'[INFO] Model {args.model} with weights {args.weights}. Starting to validate...')
    total_precision = 0
    total_recall = 0
    total_iou = 0
    total_ap = 0
    y_true_total = np.array(())
    y_pred_total = np.array(())
    with torch.no_grad():
        model.eval()
        with tqdm(val_dataloader, unit='batch') as vepoch:
            for X_val, y_val in vepoch:
                vepoch.set_description('VALIDATING...')
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                y_pred = model(X_val)  # get predictions 
                # get stats for metrics
                tp, fp, fn, tn = get_stats(y_pred, y_val.long(), mode='binary', threshold=0.5)
                # compute total metrics
                total_precision += precision(tp, fp, fn, tn, reduction='macro', zero_division=0)
                total_recall += recall(tp, fp, fn, tn, reduction='macro')
                total_iou += iou_score(tp, fp, fn, tn, reduction='macro', zero_division=0)
                # compute AP
                total_ap += bap(torch.reshape(y_pred, (-1,)), 
                                                torch.reshape(y_val.int(), (-1,)))
                y_pred = y_pred.cpu().numpy().flatten()
                y_val = y_val.cpu().numpy().flatten()
                y_pred_total = np.concatenate((y_pred_total, y_pred))
                y_true_total = np.concatenate((y_true_total, y_val))
                time.sleep(0.1)
    # save results
    history['precision'] = float(total_precision / val_steps)
    history['recall'] = float(total_recall / val_steps)
    history['iou'] = float(total_iou / val_steps)
    history['ap'] = float(total_ap / val_steps)
    print("""[INFO] Val precision: {:.3f}\nVal recall: {:.3f}\nVal IoU: {:.3f}\nVal AP: {:.3f}""".format(
        history['precision'], history['recall'],
            history['iou'], history['ap']))
    print('[INFO] Validation completed. Saving logs...')
    logs = {'model': args.model, 'batch': args.batch_size, 'weights': args.weights, 'data': args.data}
    logs.update(history)
    with open(f'logs/{args.model}/{args.task}_logs.json', 'w') as f:
        json.dump(logs, f)
    prec, rec, _ = precision_recall_curve(y_true_total, y_pred_total)
    plt.figure(figsize=(10, 10))
    plt.plot(prec, rec, 'b-')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-recall curve. Average precision - {:.3f}'.format(float(total_ap / val_steps)))
    plt.savefig(f'logs/{args.model}/pr_curve.png', bbox_inches='tight')
    fpr, tpr, _ = roc_curve(y_true_total, y_pred_total)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, 'b-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve. ROC AUC - {:.3f}'.format(roc_auc_score(y_true_total, y_pred_total)))
    plt.savefig(f'logs/{args.model}/roc_curve.png', bbox_inches='tight')
