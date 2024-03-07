from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.datasets import folder as dataset_parser
# from PIL import Image
import pathlib
import os
import open_clip, clip
from torch.optim.lr_scheduler import _LRScheduler


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

# ----------------- Warmup Scheduler -----------------
class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

def get_optimizer(params, optim_type, wd,):
    if optim_type == 'SGD':
        
        # return optim.SGD(params, lr=lr, momentum = 0.9, weight_decay=wd)
        for param in params:
            param['momentum'] = 0.9
            param['weight_decay'] = wd
        return optim.SGD(params)        
        
    elif optim_type == 'AdamW':
        # return optim.AdamW(params, lr=lr, betas=(0.9,0.999), weight_decay=wd)
        for param in params:
            param['betas'] = (0.9,0.999)
            param['weight_decay'] = wd
        return optim.AdamW(params)

def get_warmup_scheduler(optimizer, scheduler, warmup_iter = 50, warmup_lr = 1e-6):
    return LinearWarmupScheduler(
        optimizer=optimizer,
        successor=scheduler,
        warmup_epoch=warmup_iter,
        min_lr=warmup_lr
    )

def set_optimizer_scheduler(opt, model, train_loader):
    # set the trainable parameters
    for param in model.visual.parameters():
        param.requires_grad = True
    for param in model.transformer.parameters():
        param.requires_grad = False    
    params_visual = [{'params': model.visual.parameters(), 'lr': opt.learning_rate}]
    params = params_visual

    optimizer = get_optimizer(params, optim_type=opt.optim, wd=opt.weight_decay)
    total_iter = len(train_loader)*opt.epochs
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iter, eta_min=1e-9)
    warmup_lr = 1e-5 if opt.learning_rate > 5e-5 else 1e-6
    scheduler = get_warmup_scheduler(optimizer=optimizer, scheduler=base_scheduler, warmup_iter=50, warmup_lr=warmup_lr)

    return optimizer, scheduler

# ----------------- Load CLIP model -----------------
CLIP_MODEL_DIC ={
    'vitb32_CLIP': 'ViT-B/32',
    'vitb16_CLIP': 'ViT-B/16',
}

OPENCLIP_MODEL_DIC = {
    'laion400m': {
        'vitb32': ('laion400m_e32','ViT-B-32-quickgelu'),
        'vitb16': ('laion400m_e32','ViT-B-16'),
        'vitl14': ('laion400m_e32','ViT-L-14'),
    },
    'openai': {
        'vitb32': ('openai','ViT-B-32-quickgelu'),
        'vitb16': ('openai','ViT-B-16'),
        'vitl14': ('openai','ViT-L-14')
    },
    'laion2b': {
        'vitb32': ('laion2b_s34b_b79k','ViT-B-32'),
        'vitb16': ('laion2b_s34b_b88k','ViT-B-16'),
        'vitl14': ('laion2b_s32b_b82k','ViT-L-14')
    }
}


def get_engine(model_cfg, device='cpu', mode='val'):

    arch = model_cfg.split('_')[0]
    model_name = model_cfg.split('_')[1]
    pretraining_dataset = model_cfg.split('_')[2]

    print(f'arch: {arch}')
    print(f'model_name: {model_name}')
    print(f'pretraining_dataset: {pretraining_dataset}')
    
    if model_name == 'clip':
        arch = CLIP_MODEL_DIC[arch]
        model, preprocess = clip.load(arch, device)
        tokenizer = clip.tokenize
        # get the train preprocess for CLIP
        # train_preprocess = transform(224, mode='train') 
        train_preprocess = preprocess

    elif model_name == 'openclip':
        corpus_config, model_arch = OPENCLIP_MODEL_DIC[pretraining_dataset][arch]
        model, train_preprocess, preprocess = open_clip.create_model_and_transforms(model_arch, pretrained=corpus_config)
        # print('train_preprocess:', train_preprocess)
        tokenizer = open_clip.get_tokenizer(model_arch)
    
    else:
        raise NotImplementedError

    # not using mixed precision
    model.float()
    model.to(device)

    if mode == 'train':
        return model, train_preprocess, preprocess, tokenizer
    elif mode == 'val':
        return model, preprocess, tokenizer
    else:
        raise NotImplementedError


# ----------------- Datasets -----------------
    
class SemiAvesDataset(Dataset):
    def __init__(self, dataset_root, split, transform, tokenized_text_prompts=None,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        with open(os.path.join(self.dataset_root, split), 'r') as f:
            lines = f.readlines()

        self.data = []
        for line in lines:    
            path, id = line.strip('\n').split(' ')
            file_path = os.path.join(self.dataset_root, path)
            self.data.append((file_path, int(id)))
    
        self.transform = transform
        print(f'# of images in {split}: {len(self.data)}')
        self.tokenized_text_prompts = tokenized_text_prompts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        img = self.transform(img)
        # if self.tokenized_text_prompts is not None:
        #     tokenized_text = self.tokenized_text_prompts[str(label)]['all'][:1]
        # else:
        #     tokenized_text = None

        # return img, label, tokenized_text
        return img, label
