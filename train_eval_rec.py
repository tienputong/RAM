import _init_paths
import argparse
import datetime
import logging
import os
from torch.cuda.amp import autocast, GradScaler
import time
import math

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import collections.abc
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import timm
from data.dataloader import get_tri_loader

from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from evaluate_sbi import evaluate_all_sbi
from utils.Net import load_model


import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]
    
    
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a classifier network')
    parser.add_argument('--resume', default='', help='path to previous models', type=str)
    
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', type=str)

    parser.add_argument('--num_workers', dest='num_workers', help='number of worker to load data', default=10, type=int)
    parser.add_argument('--input_size', dest='input_size', help='input_size', default=224, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=128, type=int)
    # parser.add_argument('--iters', dest='iters', help='max_iter', default=20000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='save_interval', default=10000, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', type=float, default=1e-4)
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--model', type=str)
    parser.add_argument('--compression', default='c23', type=str)
    
    parser.add_argument('--aug', default='base', type=str)
    parser.add_argument('--t', default=1., type=float)

    parser.add_argument('--o', dest='optimizer', help='Training optimizer.', default='SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    parser.add_argument('--multiprocessing_distributed', type=bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--shuffle', type=bool)
    parser.add_argument('--dist-url', default='tcp://localhost:88888', type=str,
                        help='url used to set up distributed training') 
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    return parser.parse_args()

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu 
    args.start_iter = 0

    criteron = {
        'ce': torch.nn.CrossEntropyLoss()
    } 
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.model == 'vit_base_rae_mask_mtf':
        from blendingae.models_dynamic_mae_mtf import mae_vit_base_patch16
        model = mae_vit_base_patch16()

  
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)            
        else:
            model.cuda()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    elif args.gpu is not None:
        model.cuda(args.gpu)

    if args.optimizer == 'SAM':
        from utils.sam import SAM
        # model_optimizer=SAM(filter(lambda p: p.requires_grad, model.module.parameters()),torch.optim.SGD,lr=args.lr, momentum=0.9)
        model_optimizer=SAM(model.module.parameters(),torch.optim.SGD,lr=args.lr, momentum=0.9)
        # model_optimizer=SAM(filter(lambda p: p.requires_grad, model.module.parameters()),torch.optim.SGD,lr=args.lr, momentum=0.8)
    if args.optimizer == 'AdamW':
        import timm.optim.optim_factory as optim_factory
        model_without_ddp = model.module
        # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        model_optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    scheduler = LinearDecayLR(model_optimizer, args.epochs, 60)
    
    data_loader_train = get_tri_loader(args.batch_size, args.num_workers, args.input_size, 'train', True, args)
    if args.resume:
        ckpt = torch.load(args.resume, 'cpu')['model']
        model.load_state_dict(ckpt)
        scheduler.last_epoch = torch.load(args.resume, 'cpu')['epoch']
        args.start = torch.load(args.resume, 'cpu')['epoch']
    
    train(data_loader_train, model, model_optimizer, scheduler, args, ngpus_per_node)

def train(dataloader, model, model_optimizer, scheduler, args, ngpus_per_node):
    
    model.train()
    start_time = time.time()

    for epoch in range(args.start,args.epochs):
        for iter_cnt, data in enumerate(dataloader):
            images, labels, mask, target = data['img'], data['label'], data['mask'], data['target']
            
            images = images.cuda(args.gpu, non_blocking=True).float()
            target = target.cuda(args.gpu, non_blocking=True).float()
            labels = labels.cuda(args.gpu, non_blocking=True).long()
            mask = mask.cuda(args.gpu, non_blocking=True).float()
            labels[labels!=0]=1
            # with autocast():
            if args.optimizer == 'SAM':
                for i in range(2):
                    pre_rec_loss, pre_ce_loss, pre_rec_img, pred_cls = model(images, labels, mask, target)
                    alpha = 0.1
                    t_loss = 1 * pre_ce_loss + alpha * pre_rec_loss
                    model_optimizer.zero_grad()
                    t_loss.backward()
                    if i==0:
                        model_optimizer.first_step(zero_grad=True)
                    else:
                        model_optimizer.second_step(zero_grad=True)
                adjust_learning_rate(model_optimizer, iter_cnt / len(dataloader) + epoch, args)
            else:
                pre_rec_loss, pre_ce_loss, pre_rec_img, pred_cls = model(images, labels, mask, target)
                alpha = 0.1
                t_loss = 1. * pre_ce_loss + alpha * pre_rec_loss
                model_optimizer.zero_grad()
                t_loss.backward()
                model_optimizer.step()
                adjust_learning_rate(model_optimizer, iter_cnt / len(dataloader) + epoch, args)
        # scheduler.step()
        

        if args.rank % ngpus_per_node == 0: 
            loss = {}
            loss['S/totalloss'] = t_loss.item()
            loss['S/celoss'] = pre_ce_loss.item()
            loss['S/recloss'] = pre_rec_loss.item()
            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))
            log = 'time cost: {} epoch: {} / {}'.format(elapsed, epoch + 1, args.epochs)
            for tag, value in loss.items():
                log += ", {}: {:.6f}".format(tag, value)
            log += ", {}: {:.8f}".format('lr', model_optimizer.param_groups[0]["lr"])
            print(log)
        if (epoch+1) % 20 == 0 and args.rank % ngpus_per_node == 0 and epoch+1>50: 
            model.eval()
            try:
                res = evaluate_all_sbi(model, args)
                evaluate_dff_sbi(model, args)
                with open(os.path.join(args.save_dir,'res.txt'), 'a') as f:
                    for key, value in res.items():
                        f.write(f'{key}:{value}\n')
            except:
                pass
        model.train()
        if (epoch+1) % 20 == 0 and args.rank % ngpus_per_node == 0:
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'model_optimizer': model_optimizer.state_dict()
            }
            save_name = os.path.join(args.save_dir, 'model_{}.pth'.format(epoch+1))
            torch.save(state, save_name)

def main():
    args = parse_args()
    logger.info('\t Called with args:')
    logger.info(args)

    model_tag = args.model
    save_dir = os.path.join(args.save_dir, model_tag, 'ckpt')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info('\t output will be saved to {}'.format(save_dir))
    args.save_dir = save_dir


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
