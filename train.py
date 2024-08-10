import argparse
import time
import os
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from tempfile import TemporaryDirectory
from vision_transformer.vit_model import VisionTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils import FocalLoss

# Function to set the random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Vision Transformer Training Script")
    parser.add_argument('--data_dir', default='data/tongue', type=str, help='Directory of the dataset')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--weight_path', default='vit_weights/vit_base_patch16_224.pth', type=str, help='Path to pretrained weights')
    parser.add_argument('--output_dir', default='models', type=str, help='Directory to save the model')
    parser.add_argument('--learning_rate', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--seed', default=77, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the model')
    parser.add_argument('--log_dir', default='logs', type=str, help='Directory to save the logs')
    parser.add_argument('--results', default='results', type=str, help='Directory to save the results')
    return parser.parse_args()

# Training function
def train_model(writer, model, criterion, criterion2, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, best_acc=0.0, output_dir='models', device='cuda:1'):
    since = time.time()
    mean_metrics = {}
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(output_dir, 'tongue_best.pt')
        results = {}
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                all_preds = []
                all_labels = []

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels_zero = labels * 0
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        cls = outputs[:, 0, :]
                        loss1 = criterion(cls, labels)

                        instances = outputs[:, 0:-1, :]
                        c, i, l = instances.shape
                        instances_softmax = torch.softmax(instances, dim=2)
                        instances_softmax_pos = instances_softmax[:, :, 1]

                        max_res, preds = torch.max(instances_softmax_pos, dim=1)
            
                        max_instance = torch.zeros((c, l)).to(device)
                        for i in range(preds.shape[0]):
                            max_instance[i] = instances[i, preds[i], :]
    
                        _, preds = torch.max(max_instance, dim=1)
                        loss2 = criterion(max_instance, labels)
                        loss2_fc = criterion2(max_instance, labels)

                        loss =  0.5*(loss1 + loss2) + loss2_fc

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = accuracy_score(all_labels, all_preds)
                epoch_recall = recall_score(all_labels, all_preds)
                epoch_precision = precision_score(all_labels, all_preds)
                epoch_f1 = f1_score(all_labels, all_preds)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} F1: {epoch_f1:.4f}')
                # write to tensorboard

                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('Loss/val', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                    writer.add_scalar('Recall/val', epoch_recall, epoch)
                    writer.add_scalar('Precision/val', epoch_precision, epoch)
                    writer.add_scalar('F1/val', epoch_f1, epoch)

                if phase == 'val' and epoch_acc > best_acc:
                    print('Saving best model')
                    best_acc = epoch_acc
                    torch.save(model, best_model_params_path)

        time_elapsed = time.time() - since
        with open(os.path.join(output_dir, 'metric.txt'), 'a') as f:
            f.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} F1: {epoch_f1:.4f}\n')
        
        mean_metrics['loss'] = epoch_loss
        mean_metrics['acc'] = epoch_acc
        mean_metrics['recall'] = epoch_recall
        mean_metrics['precision'] = epoch_precision
        mean_metrics['f1'] = epoch_f1

        # 记录验证结果
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        print('Saving last epoch model')
        last_epoch_params_path = os.path.join(output_dir, 'tongue_last_epoch.pt')
        torch.save(model, last_epoch_params_path)

    return mean_metrics


def main():
    args = parse_args()
    setup_seed(args.seed)

    writer = SummaryWriter(log_dir=args.log_dir)

    # 写入metric.txt
    with open(os.path.join(args.output_dir, 'metric.txt'), 'a') as f:
        f.write('args: ' + str(args) + '\n')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = datasets.ImageFolder(args.data_dir)
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    device = torch.device(args.device)
    best_acc = 0.0

    weights_dict = torch.load(args.weight_path, map_location=device)
    del_keys = ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    
    mean_metrics = {'loss': 0, 'acc': 0, 'recall': 0, 'precision': 0, 'f1': 0}

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_datasets)):
        print(f'Fold {fold + 1}/{args.num_folds}')
        with open(os.path.join(args.output_dir, 'metric.txt'), 'a') as f:
            f.write(f'Fold {fold + 1}/{args.num_folds}\n')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataset = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, sampler=train_subsampler)
        val_dataset = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, sampler=val_subsampler)

        train_dataset.dataset.transform = data_transforms['train']
        val_dataset.dataset.transform = data_transforms['val']

        dataloaders = {
            'train': train_dataset,
            'val': val_dataset
        }

        dataset_sizes = {'train': len(train_idx), 'val': len(val_idx)}

        model_ft = VisionTransformer(img_size=224,
                                     patch_size=16,
                                     embed_dim=768,
                                     depth=12,
                                     num_heads=12,
                                     representation_size=None,
                                     num_classes=2,
                                     drop_path_ratio=0.1,
                                     drop_ratio=0.0,
                                     attn_drop_ratio=0.0,
                                     )
        


        if args.weight_path:
            model_ft.load_state_dict(weights_dict, strict=False)
        
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        criterion2 = FocalLoss(alpha=0.25, gamma=2.0).to(device)

        optimizer_ft = optim.AdamW(model_ft.parameters(), lr=args.learning_rate)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=5, T_mult=2, eta_min=1e-6)			

        
        epoch_metrics = train_model(writer, model_ft, criterion, criterion2, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, best_acc=best_acc, num_epochs=args.num_epochs, output_dir=args.output_dir, device=device)
        for key, value in epoch_metrics.items():
            mean_metrics[key] += value
        # print mean metrics\
    
    for key, value in mean_metrics.items():
        mean_metrics[key] /= args.num_folds
        with open(os.path.join(args.output_dir, 'metric.txt'), 'a') as f:
            f.write(key + ': ' + str(mean_metrics[key]) + '\n')
    print(mean_metrics)

if __name__ == "__main__":
    main()
