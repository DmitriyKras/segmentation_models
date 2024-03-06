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
# import losses
from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss, JaccardLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
# import metrics
import torch.nn.functional as F
from segmentation_models_pytorch.metrics import get_stats, precision, recall, iou_score
from torchmetrics.classification import BinaryAveragePrecision
# import models
from models.effunet import EfficientUnet
from models.bisenet import BiSeNet
from models.liteseg_mobilenet import LiteSegMobileNetV2
from models.liteseg_shufflenet import LiteSegShuffleNet
from models.esnet import ESNet
# import callbacks
from utils.callbacks import EarlyStopping, ModelCheckpoint
# to draw results
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, bisenet, 
                        liteseg_mobilenet, liteseg_shufflenet, esnet""")
    parser.add_argument('--loss', type=str, default='dice', 
                        help="""Segmentation loss for training. Available losses: dice, tversky""")
    parser.add_argument('--epochs', type=int, default=100, 
                        help='maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=20, 
                        help='batch size')
    parser.add_argument('--augment', type=bool, default=True, 
                        help='whether to use augmentation during training')
    parser.add_argument('--img', type=int, default=512, 
                        help='train, val image size (pixels)')
    parser.add_argument('--save-period', type=int, default=-1, 
                        help='Save checkpoint every x epochs')
    parser.add_argument('--patience', type=int, default=5, 
                        help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='optimizer learning rate')
    parser.add_argument('--decay', type=float, default=0.9, 
                        help='optimizer learning rate exponential decay after each epoch')
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
    images = sorted(list(glob.glob(data['path'] + data['train'] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data['train'].replace('images', 'masks') + '/*.jpg')))
    train_dataset = SegmentationDataset(images, masks, (args.img, args.img), args.augment)
    images = sorted(list(glob.glob(data['path'] + data['val'] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data['val'].replace('images', 'masks') + '/*.jpg')))
    val_dataset = SegmentationDataset(images, masks, (args.img, args.img), False)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
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

    ### CONFIGURE OPTIMIZERS ###
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.decay)
    ### GET LOSS ###
    if args.loss == 'dice':
        train_loss = DiceLoss(mode='binary', smooth=1.0, from_logits=False)
        val_loss = DiceLoss(mode='binary', smooth=1.0, from_logits=False)
    elif args.loss == 'tversky':
        train_loss = TverskyLoss(mode='binary', smooth=1.0, from_logits=False)
        val_loss = TverskyLoss(mode='binary', smooth=1.0, from_logits=False)
    else:
        print(f'[WARNING] Loss {args.loss} not implemented, using default DiceLoss')
    ### INIT METRICS ###
    history = {'loss' : [], 'val_loss' : [], 'val_precision' : [], 'val_recall' : [], 'val_iou' : [], 
               'val_ap' : []}  # result
    train_steps = ceil(len(train_dataset) / args.batch_size)  # number of train and val steps
    val_steps = ceil(len(val_dataset) / args.batch_size)
    bap = BinaryAveragePrecision(thresholds=50, validate_args=False).cuda()
    ### INIT CALLBACKS ###
    es = EarlyStopping(args.patience)
    mc = ModelCheckpoint(args.save_period)
    ### TRAIN LOOP ###
    best_val_loss = 1
    print(f'[INFO] Model {args.model} with loss {args.loss}. Starting to train...')
    for epoch in range(args.epochs):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        total_val_loss = 0
        total_val_precision = 0
        total_val_recall = 0
        total_val_iou = 0
        total_val_ap = 0
        # iterate over batches
        with tqdm(train_dataloader, unit='batch') as tepoch:
            for X_train, y_train in tepoch:
                tepoch.set_description(f'EPOCH {epoch + 1}/{args.epochs} TRAINING')
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                y_pred = model(X_train)  # get predictions
                # special training strategy for several models
                if args.model == 'bisenet':
                    result, cx1_sup, cx2_sup = y_pred  # unpack outputs
                    loss1 = train_loss(F.sigmoid(result), y_train.float())  # compute loss for each output
                    loss2 = train_loss(F.sigmoid(cx1_sup), y_train.float())
                    loss3 = train_loss(F.sigmoid(cx2_sup), y_train.float())
                    loss = loss1 + loss2 + loss3  # sum all losses
                else:    
                    loss = train_loss(y_pred, y_train.float())  # compute loss
                optimizer.zero_grad()
                loss.backward()  # back propogation
                optimizer.step()  # optimizer's step
                total_loss += loss.item() / 3  # add to total loss
                tepoch.set_postfix(loss=loss.item() / 3)
                time.sleep(0.1)

        scheduler.step()  # apply lr decay
        history['loss'].append(float(total_loss / train_steps))  # write logs
        print('[INFO] Train loss: {:.4f}\n'.format(history['loss'][-1]))
        print('[INFO] Validating...')
        # perform validation
        with torch.no_grad():
            model.eval()
            with tqdm(val_dataloader, unit='batch') as vepoch:
                for X_val, y_val in vepoch:
                    vepoch.set_description(f'EPOCH {epoch + 1}/{args.epochs} VALIDATING')
                    X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                    y_pred = model(X_val)  # get predictions
                    loss = val_loss(y_pred, y_val.float())
                    total_val_loss += loss.item()  # compute val loss
                    # get stats for metrics
                    tp, fp, fn, tn = get_stats(y_pred, y_val.long(), mode='binary', threshold=0.5)
                    # compute total metrics
                    total_val_precision += precision(tp, fp, fn, tn, reduction='macro', zero_division=0)
                    total_val_recall += recall(tp, fp, fn, tn, reduction='macro')
                    total_val_iou += iou_score(tp, fp, fn, tn, reduction='macro', zero_division=0)
                    # compute AP
                    total_val_ap += bap(torch.reshape(y_pred, (-1,)), 
                                                    torch.reshape(y_val.int(), (-1,)))
                    vepoch.set_postfix(loss=loss.item())
                    time.sleep(0.1)

        history['val_loss'].append(float(total_val_loss / val_steps))  # write logs
        history['val_precision'].append(float(total_val_precision / val_steps))
        history['val_recall'].append(float(total_val_recall / val_steps))
        history['val_iou'].append(float(total_val_iou / val_steps))
        history['val_ap'].append(float(total_val_ap / val_steps))
        print("""[INFO] Val loss: {:.3f}\nVal precision: {:.3f}\nVal recall: {:.3f}\nVal IoU: {:.3f}\nVal AP: {:.3f}""".format(
            history['val_loss'][-1], history['val_precision'][-1], history['val_recall'][-1],
              history['val_iou'][-1], history['val_ap'][-1]))
        if history['val_loss'][-1] < best_val_loss:  # save best weights
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_best.pt')
        if es.step(history['val_loss'][-1]):  # check early stopping
            print(f'[INFO] Activating early stopping callback at epoch {epoch}')
            break
        if mc.step():  # check model checkpoint
            print(f'[INFO] Activating model checkpoint callback at epoch {epoch}')
            torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_epoch{epoch}.pt')

    print('[INFO] Training finished!')
    torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_last.pt')
    print('[INFO] Saving training logs...')
    logs = {'model': args.model, 'loss_type': args.loss, 'batch': args.batch_size, 'epochs': epoch, 'data': args.data}
    logs.update(history)
    with open(f'logs/{args.model}/train_logs.json', 'w') as f:
        json.dump(logs, f)

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, logs['epochs'] + 2), logs['loss'], '-b', label='train loss')
    plt.plot(range(1, logs['epochs'] + 2), logs['val_loss'], '-r', label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss {args.loss} over epochs')
    plt.legend()
    plt.savefig(f'logs/{args.model}/loss.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, logs['epochs'] + 2), logs['val_precision'], '-b', label='precision')
    plt.plot(range(1, logs['epochs'] + 2), logs['val_recall'], '-r', label='recall')
    plt.plot(range(1, logs['epochs'] + 2), logs['val_iou'], '--k', label='iou')
    plt.plot(range(1, logs['epochs'] + 2), logs['val_ap'], '--y', label='ap')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Metrics over epochs')
    plt.legend()
    plt.savefig(f'logs/{args.model}/metrics.png', bbox_inches='tight')
