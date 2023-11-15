from dataset import *
from loss import *
from lars import *
from model import *
from utils import *

import argparse
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule


tracker = None
writer = None



def init():
    # Define master address and port
    import idr_torch

    # Get task information
    global_rank = idr_torch.rank
    local_rank = idr_torch.local_rank
    world_size = idr_torch.size
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])
    n_gpu_per_node = world_size // n_nodes
    is_master = global_rank == 0
    
    # Print all task information
    print('{:>2}> Task {:>2} in {:>2} | Node {} in {} | GPU {} in {} | {}'.format(
        global_rank, global_rank, world_size, node_id, n_nodes, local_rank, n_gpu_per_node, 'Master' if is_master else '-'))
    
    # Set the device to use
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')
    
    # Init the process group
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
        world_size=world_size, 
        rank=global_rank
    )

    return global_rank, local_rank, world_size, is_master, device



def train_repr(model, optimizer, criterion, loader, scheduler, scaler, args, device, is_master, global_rank, world_size):
    model.train()

    tracker.reset('train_loss_z')
    tracker.reset('train_loss_eq_y')
    tracker.reset('train_loss_repr')

    tracker.reset('train_y0_norm_mean')
    tracker.reset('train_y0_norm_std')
    tracker.reset('train_yt_norm_mean')
    tracker.reset('train_yt_norm_std')
    tracker.reset('train_ythat_norm_mean')
    tracker.reset('train_ythat_norm_std')

    tracker.reset('train_h_norm_mean')
    tracker.reset('train_h_norm_std')
    tracker.reset('train_z_norm_mean')
    tracker.reset('train_z_norm_std')

    tracker.reset('train_eqgain')
    tracker.reset('train_eqgain_bis')

    if args.dataset_name == 'cifar10':
        p_mean = torch.tensor([[4.3122e+00, 4.3216e+00, 2.3369e+01, 2.3374e+01, 4.9998e-01, 8.0087e-01,
            1.2025e+00, 1.4007e+00, 1.5964e+00, 1.8004e+00, 9.9993e-01, 1.0002e+00,
            9.9986e-01, 2.2321e-07, 1.9965e-01]]).to(device)
        p_std = torch.tensor([[3.9740, 3.9851, 4.9544, 4.9539, 0.5000, 0.3993, 1.1651, 1.0210, 1.0200,
            1.1669, 0.2066, 0.2068, 0.2066, 0.0517, 0.3997]]).to(device)
    elif args.dataset_name == 'imagenet':
        p_mean = torch.tensor([[6.8162e+01, 9.9199e+01, 2.6933e+02, 2.7457e+02, 4.9905e-01, 8.0054e-01,
            1.1998e+00, 1.3994e+00, 1.6014e+00, 1.7995e+00, 1.0001e+00, 1.0000e+00,
            1.0005e+00, 1.5640e-04, 2.0018e-01, 5.0030e-01, 5.2507e-01]]).to(device)
        p_std = torch.tensor([[7.7370e+01, 9.8681e+01, 1.3686e+02, 1.4349e+02, 5.0000e-01, 3.9959e-01,
            1.1661e+00, 1.0201e+00, 1.0201e+00, 1.1657e+00, 4.1347e-01, 4.1323e-01,
            4.1349e-01, 1.0333e-01, 4.0013e-01, 5.0000e-01, 6.5251e-01]]).to(device)

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        (img_0, img_1, param_1, img_2, param_2), label = batch

        img_0 = img_0.to(device, non_blocking=True)

        img_1 = img_1.to(device, non_blocking=True)
        param_1 = param_1.to(device, non_blocking=True)
        img_2 = img_2.to(device, non_blocking=True)
        param_2 = param_2.to(device, non_blocking=True)

        x = torch.cat([img_0, img_1, img_2], dim=0)
        p = torch.cat([param_1, param_2], dim=0)
        p = (p - p_mean)/p_std

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            h, z, y0, yt, yt_hat = model(x, p)

        batch_size = img_1.shape[0]        

        # Gather all embeddings arcoss all devices to compute contrastive loss over all negatives
        if args.hardware == 'multi-gpu':
            all_z = gather_pairs(z, global_rank, world_size)
            all_y = gather_pairs(torch.cat([yt, yt_hat], dim=0), global_rank, world_size)
        else:
            all_z = z
            all_y = torch.cat([yt, yt_hat], dim=0)

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            loss_z = criterion['NTXentLoss'](all_z, args.temperature_z)*world_size
            loss_eq_y = criterion['NTXentLoss'](all_y, args.temperature_y)*world_size

            loss = loss_z + loss_eq_y*args.lmbd

        scaler.scale(loss).backward() if args.precision == 'mixed' else loss.backward()
        scaler.step(optimizer) if args.precision == 'mixed' else optimizer.step()
        if args.precision == 'mixed': scaler.update()
        if scheduler is not None: scheduler.step()

        tracker.add('train_loss_z', loss_z.detach()*batch_size/world_size, batch_size*2)
        tracker.add('train_loss_eq_y', loss_eq_y.detach()*batch_size/world_size, batch_size*2)
        tracker.add('train_loss_repr', loss.detach()*batch_size, batch_size*2)

        tracker.add('train_y0_norm_mean', y0.detach().norm(dim=1).mean(), 1)
        tracker.add('train_y0_norm_std', y0.detach().norm(dim=1).std(), 1)
        tracker.add('train_yt_norm_mean', yt.detach().norm(dim=1).mean(), 1)
        tracker.add('train_yt_norm_std', yt.detach().norm(dim=1).std(), 1)
        tracker.add('train_ythat_norm_mean', yt_hat.detach().norm(dim=1).mean(), 1)
        tracker.add('train_ythat_norm_std', yt_hat.detach().norm(dim=1).std(), 1)

        tracker.add('train_h_norm_mean', h.detach().norm(dim=1).mean(), 1)
        tracker.add('train_h_norm_std', h.detach().norm(dim=1).std(), 1)
        tracker.add('train_z_norm_mean', z.detach().norm(dim=1).mean(), 1)
        tracker.add('train_z_norm_std', z.detach().norm(dim=1).std(), 1)

        tracker.add('train_eqgain', (cos(yt, yt_hat) - cos(yt, y0)).sum(), batch_size*2)
        tracker.add('train_eqgain_bis', ((1. - cos(yt, y0))/(1. - cos(yt, yt_hat))).sum(), batch_size*2)

        if is_master:
            print('\rTrain Batch: {:>3}/{:<3}...'.format(i + 1, len(loader)), end='')

    if args.hardware == 'multi-gpu':
        tracker.get('train_loss_z').reduce()
        tracker.get('train_loss_eq_y').reduce()         
        tracker.get('train_loss_repr').reduce()
    
    if is_master:
        writer.add_scalar('train_loss_z', tracker.get('train_loss_z').average, tracker.get('epoch').count)
        writer.add_scalar('train_loss_eq_y', tracker.get('train_loss_eq_y').average, tracker.get('epoch').count)
        writer.add_scalar('train_loss_repr', tracker.get('train_loss_repr').average, tracker.get('epoch').count)

        writer.add_scalar('train_y0_norm_mean', tracker.get('train_y0_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_y0_norm_std', tracker.get('train_y0_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_yt_norm_mean', tracker.get('train_yt_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_yt_norm_std', tracker.get('train_yt_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_ythat_norm_mean', tracker.get('train_ythat_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_ythat_norm_std', tracker.get('train_ythat_norm_std').average, tracker.get('epoch').count)

        writer.add_scalar('train_h_norm_mean', tracker.get('train_h_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_h_norm_std', tracker.get('train_h_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_z_norm_mean', tracker.get('train_z_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_z_norm_std', tracker.get('train_z_norm_std').average, tracker.get('epoch').count)
        
        writer.add_scalar('train_eqgain', tracker.get('train_eqgain').average, tracker.get('epoch').count)
        writer.add_scalar('train_eqgain_bis', tracker.get('train_eqgain_bis').average, tracker.get('epoch').count)



def eval_repr(model, criterion, loader, args, device, is_master, global_rank, world_size):
    model.eval()

    tracker.reset('eval_loss_z')
    tracker.reset('eval_loss_eq_y')
    tracker.reset('eval_loss_repr')

    tracker.reset('eval_y0_norm_mean')
    tracker.reset('eval_y0_norm_std')
    tracker.reset('eval_yt_norm_mean')
    tracker.reset('eval_yt_norm_std')
    tracker.reset('eval_ythat_norm_mean')
    tracker.reset('eval_ythat_norm_std')

    tracker.reset('eval_h_norm_mean')
    tracker.reset('eval_h_norm_std')
    tracker.reset('eval_z_norm_mean')
    tracker.reset('eval_z_norm_std')

    tracker.reset('eval_eqgain')
    tracker.reset('eval_eqgain_bis')

    if args.dataset_name == 'cifar10':
        p_mean = torch.tensor([[4.3122e+00, 4.3216e+00, 2.3369e+01, 2.3374e+01, 4.9998e-01, 8.0087e-01,
            1.2025e+00, 1.4007e+00, 1.5964e+00, 1.8004e+00, 9.9993e-01, 1.0002e+00,
            9.9986e-01, 2.2321e-07, 1.9965e-01]]).to(device)
        p_std = torch.tensor([[3.9740, 3.9851, 4.9544, 4.9539, 0.5000, 0.3993, 1.1651, 1.0210, 1.0200,
            1.1669, 0.2066, 0.2068, 0.2066, 0.0517, 0.3997]]).to(device)
    elif args.dataset_name == 'imagenet':
        p_mean = torch.tensor([[6.8162e+01, 9.9199e+01, 2.6933e+02, 2.7457e+02, 4.9905e-01, 8.0054e-01,
            1.1998e+00, 1.3994e+00, 1.6014e+00, 1.7995e+00, 1.0001e+00, 1.0000e+00,
            1.0005e+00, 1.5640e-04, 2.0018e-01, 5.0030e-01, 5.2507e-01]]).to(device)
        p_std = torch.tensor([[7.7370e+01, 9.8681e+01, 1.3686e+02, 1.4349e+02, 5.0000e-01, 3.9959e-01,
            1.1661e+00, 1.0201e+00, 1.0201e+00, 1.1657e+00, 4.1347e-01, 4.1323e-01,
            4.1349e-01, 1.0333e-01, 4.0013e-01, 5.0000e-01, 6.5251e-01]]).to(device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            (img_0, img_1, param_1, img_2, param_2), label = batch

            img_0 = img_0.to(device, non_blocking=True)

            img_1 = img_1.to(device, non_blocking=True)
            param_1 = param_1.to(device, non_blocking=True)
            img_2 = img_2.to(device, non_blocking=True)
            param_2 = param_2.to(device, non_blocking=True)

            x = torch.cat([img_0, img_1, img_2], dim=0)
            p = torch.cat([param_1, param_2], dim=0)
            p = (p - p_mean)/p_std

            with torch.cuda.amp.autocast(args.precision == 'mixed'):
                h, z, y0, yt, yt_hat = model(x, p)

            batch_size = img_1.shape[0]        

            # Gather all embeddings arcoss all devices to compute contrastive loss over all negatives
            if args.hardware == 'multi-gpu':
                all_z = gather_pairs(z, global_rank, world_size)
                all_y = gather_pairs(torch.cat([yt, yt_hat], dim=0), global_rank, world_size)
            else:
                all_z = z
                all_y = torch.cat([yt, yt_hat], dim=0)

            with torch.cuda.amp.autocast(args.precision == 'mixed'):
                loss_z = criterion['NTXentLoss'](all_z, args.temperature_z)*world_size
                loss_eq_y = criterion['NTXentLoss'](all_y, args.temperature_y)*world_size

                loss = loss_z + loss_eq_y*args.lmbd

            tracker.add('eval_loss_z', loss_z*batch_size/world_size, batch_size*2)
            tracker.add('eval_loss_eq_y', loss_eq_y*batch_size/world_size, batch_size*2)
            tracker.add('eval_loss_repr', loss*batch_size, batch_size*2)

            tracker.add('eval_y0_norm_mean', y0.norm(dim=1).mean(), 1)
            tracker.add('eval_y0_norm_std', y0.norm(dim=1).std(), 1)
            tracker.add('eval_yt_norm_mean', yt.norm(dim=1).mean(), 1)
            tracker.add('eval_yt_norm_std', yt.norm(dim=1).std(), 1)
            tracker.add('eval_ythat_norm_mean', yt_hat.norm(dim=1).mean(), 1)
            tracker.add('eval_ythat_norm_std', yt_hat.norm(dim=1).std(), 1)

            tracker.add('eval_h_norm_mean', h.norm(dim=1).mean(), 1)
            tracker.add('eval_h_norm_std', h.norm(dim=1).std(), 1)
            tracker.add('eval_z_norm_mean', z.norm(dim=1).mean(), 1)
            tracker.add('eval_z_norm_std', z.norm(dim=1).std(), 1)

            tracker.add('eval_eqgain', (cos(yt, yt_hat) - cos(yt, y0)).sum(), batch_size*2)
            tracker.add('eval_eqgain_bis', ((1. - cos(yt, y0))/(1. - cos(yt, yt_hat))).sum(), batch_size*2)
            
            if is_master:
                print('\rEval Batch: {:>3}/{:<3}...'.format(i + 1, len(loader)), end='')

        if args.hardware == 'multi-gpu':
            tracker.get('eval_loss_z').reduce()
            tracker.get('eval_loss_eq_y').reduce()
            tracker.get('eval_loss_repr').reduce()
        
        if is_master:
            writer.add_scalar('eval_loss_z', tracker.get('eval_loss_z').average, tracker.get('epoch').count)
            writer.add_scalar('eval_loss_eq_y', tracker.get('eval_loss_eq_y').average, tracker.get('epoch').count)
            writer.add_scalar('eval_loss_repr', tracker.get('eval_loss_repr').average, tracker.get('epoch').count)

            writer.add_scalar('eval_y0_norm_mean', tracker.get('eval_y0_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_y0_norm_std', tracker.get('eval_y0_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_yt_norm_mean', tracker.get('eval_yt_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_yt_norm_std', tracker.get('eval_yt_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_ythat_norm_mean', tracker.get('eval_ythat_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_ythat_norm_std', tracker.get('eval_ythat_norm_std').average, tracker.get('epoch').count)

            writer.add_scalar('eval_h_norm_mean', tracker.get('eval_h_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_h_norm_std', tracker.get('eval_h_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_z_norm_mean', tracker.get('eval_z_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_z_norm_std', tracker.get('eval_z_norm_std').average, tracker.get('epoch').count)

            writer.add_scalar('eval_eqgain', tracker.get('eval_eqgain').average, tracker.get('epoch').count)
            writer.add_scalar('eval_eqgain_bis', tracker.get('eval_eqgain_bis').average, tracker.get('epoch').count)



def train_clsf(model, linear, optimizer, criterion, loader, scheduler, scaler, args, device, is_master, global_rank, world_size):
    model.eval()
    linear.train()

    tracker.reset('train_loss_clsf')
    tracker.reset('train_acc_1')
    tracker.reset('train_acc_5')
    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        image_0, target = batch
        image_0 = image_0.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = image_0.shape[0]

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            with torch.no_grad():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    h = model.module.resnet(image_0)
                else:
                    h = model.resnet(image_0)
            c = linear(h)

            loss = criterion(c, target)

        scaler.scale(loss).backward() if args.precision == 'mixed' else loss.backward()
        scaler.step(optimizer) if args.precision == 'mixed' else optimizer.step()
        if args.precision == 'mixed': scaler.update()
        scheduler.step()

        # Compute accuracy (top-1 and top-5)
        with torch.no_grad():
            _, pred = c.topk(5, 1, True, True)
            correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).float()

            tracker.add('train_loss_clsf', loss.detach()*batch_size, batch_size)
            tracker.add('train_acc_1', correct[:, :1].sum(), batch_size)
            tracker.add('train_acc_5', correct[:, :5].sum(), batch_size)

        if is_master:
            print('\rTrain Batch: {:>3}/{:<3}'.format(i + 1, len(loader)), end='')

    if args.hardware == 'multi-gpu':
        tracker.get('train_loss_clsf').reduce()
        tracker.get('train_acc_1').reduce()
        tracker.get('train_acc_5').reduce()
    
    if is_master:
        writer.add_scalar('curve_{}_train_loss_clsf'.format(tracker.get('epoch').count), tracker.get('train_loss_clsf').average, tracker.get('epoch_clsf').count)
        writer.add_scalar('curve_{}_train_acc_1'.format(tracker.get('epoch').count), tracker.get('train_acc_1').average, tracker.get('epoch_clsf').count)
        writer.add_scalar('curve_{}_train_acc_5'.format(tracker.get('epoch').count), tracker.get('train_acc_5').average, tracker.get('epoch_clsf').count)



def eval_clsf(model, linear, criterion, loader, args, device, is_master, global_rank, world_size):
    model.eval()
    linear.eval()

    tracker.reset('eval_loss_clsf')
    tracker.reset('eval_acc_1')
    tracker.reset('eval_acc_5')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            image_0, target = batch
            image_0 = image_0.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            batch_size = image_0.shape[0]

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                h = model.module.resnet(image_0)
            else:
                h = model.resnet(image_0)
            c = linear(h)

            loss = criterion(c, target)

            # Compute accuracy (top-1 and top-5)
            _, pred = c.topk(5, 1, True, True)
            correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).float()

            tracker.add('eval_loss_clsf', loss.detach()*batch_size, batch_size)
            tracker.add('eval_acc_1', correct[:, :1].sum(), batch_size)
            tracker.add('eval_acc_5', correct[:, :5].sum(), batch_size)

            if is_master:
                print('\rTrain Batch: {:>3}/{:<3}'.format(i + 1, len(loader)), end='')

        if args.hardware == 'multi-gpu':
            tracker.get('eval_loss_clsf').reduce()
            tracker.get('eval_acc_1').reduce()
            tracker.get('eval_acc_5').reduce()
        
        if is_master:
            writer.add_scalar('curve_{}_eval_loss_clsf'.format(tracker.get('epoch').count), tracker.get('eval_loss_clsf').average, tracker.get('epoch_clsf').count)
            writer.add_scalar('curve_{}_eval_acc_1'.format(tracker.get('epoch').count), tracker.get('eval_acc_1').average, tracker.get('epoch_clsf').count)
            writer.add_scalar('curve_{}_eval_acc_5'.format(tracker.get('epoch').count), tracker.get('eval_acc_5').average, tracker.get('epoch_clsf').count)



def linear_eval(args, model, train_clsf_loader, eval_clsf_loader, train_clsf_sampler, eval_clsf_sampler, scaler, global_rank, local_rank, world_size, is_master, device):
    ######### MODEL and CO #########
    # CREATE LINEAR MODEL
    in_dim = model.module.h_dim if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.h_dim
    linear = torch.nn.Linear(in_dim, get_nbclasses(args.dataset_name)).to(device)
    if args.hardware == 'multi-gpu':
        linear = DDP(linear, device_ids=[local_rank])
    
    # CREATE CRITERION
    criterion_clsf = torch.nn.CrossEntropyLoss()

    # CREATE OPTIMIZER
    optimizer_clsf = torch.optim.SGD(
        linear.parameters(),
        lr=args.lr_init_clsf,
        momentum=args.momentum_clsf,
        weight_decay=args.weight_decay_clsf,
        nesterov=True
    )

    # CREATE SCHEDULER
    cosine_scheduler_clsf = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_clsf, args.nb_epochs_clsf*len(train_clsf_loader), eta_min=0.0001
    )


    ######### ITERATE EPOCHS CLSF #########
    tracker.reset('epoch_clsf')
    for epoch in range(args.nb_epochs_clsf):
        start = time.time()
        tracker.add('epoch_clsf', 0)

        if args.hardware == 'multi-gpu':
            train_clsf_sampler.set_epoch(epoch)
            eval_clsf_sampler.set_epoch(epoch)

        train_clsf(model, linear, optimizer_clsf, criterion_clsf, train_clsf_loader, cosine_scheduler_clsf, scaler, args, device, is_master, global_rank, world_size)
        eval_clsf(model, linear, criterion_clsf, eval_clsf_loader, args, device, is_master, global_rank, world_size)

        if is_master:
            print('\rEpoch: {:>3}/{:<3}  |  Loss Train: {:e}  |  Loss Eval: {:e}  |  Top-1 Train: {:>.5f}  |  Top-1 Eval: {:>.5f}  |  Time taken: {}'
                .format(epoch + 1, args.nb_epochs_clsf, tracker.get('train_loss_clsf').average, tracker.get('eval_loss_clsf').average, tracker.get('train_acc_1').average, tracker.get('eval_acc_1').average, time.time() - start))
        
    if is_master:
        writer.add_scalar('train_loss_clsf', tracker.get('train_loss_clsf').average, tracker.get('epoch').count)
        writer.add_scalar('train_acc_1', tracker.get('train_acc_1').average, tracker.get('epoch').count)
        writer.add_scalar('train_acc_5', tracker.get('train_acc_5').average, tracker.get('epoch').count)

        writer.add_scalar('eval_loss_clsf', tracker.get('eval_loss_clsf').average, tracker.get('epoch').count)
        writer.add_scalar('eval_acc_1', tracker.get('eval_acc_1').average, tracker.get('epoch').count)
        writer.add_scalar('eval_acc_5', tracker.get('eval_acc_5').average, tracker.get('epoch').count)



def run(args):
    ######### INIT #########
    # INIT DEVICES
    if args.hardware == 'multi-gpu':
        if args.computer == 'jeanzay':
            global_rank, local_rank, world_size, is_master, device = init()
        else:
            raise Exception('Multi-GPU is only supported on JeanZay')
    else:
        global_rank, local_rank, world_size, is_master = 0, 0, 1, True
        device = torch.device('cuda') if args.hardware == 'mono-gpu' else torch.device('cpu')
    
    if args.hardware != 'cpu': torch.backends.cudnn.benchmark = True

    # INIT CHECKPOINT
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # INIT TRACKER
    global tracker
    tracker = Tracker(device)

    # INIT WRITER
    if is_master:
        global writer

        if args.computer == 'jeanzay':
            log_dir = './runs/{}_{}_{}'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], args.expe_name)
        else:
            expe_time = int(time.time())
            log_dir = './runs/{}_{}_{}'.format('expe', expe_time, args.expe_name)
        
        if checkpoint is not None:
            log_dir = checkpoint['log_dir']

        writer = SummaryWriter(log_dir)

        print('\nArgs:', args, '\n')

    
    ######### DATASETS #########
    # CREATE DATASETS
    train_repr_set = get_dataset(args.dataset_name, 'train_repr')
    eval_repr_set = get_dataset(args.dataset_name, 'eval_repr')
    train_clsf_set = get_dataset(args.dataset_name, 'train_clsf')
    eval_clsf_set = get_dataset(args.dataset_name, 'eval_clsf')

    # CREATE SAMPLERS
    if args.hardware == 'multi-gpu':
        train_repr_sampler = torch.utils.data.distributed.DistributedSampler(
            train_repr_set,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        eval_repr_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_repr_set,
            num_replicas=world_size,
            rank=global_rank
        )
        train_clsf_sampler = torch.utils.data.distributed.DistributedSampler(
            train_clsf_set,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        eval_clsf_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_clsf_set,
            num_replicas=world_size,
            rank=global_rank
        )
    else:
        train_repr_sampler = torch.utils.data.RandomSampler(
            train_repr_set
        )
        eval_repr_sampler = torch.utils.data.SequentialSampler(
            eval_repr_set
        )
        train_clsf_sampler = torch.utils.data.RandomSampler(
            train_clsf_set
        )
        eval_clsf_sampler = torch.utils.data.SequentialSampler(
            eval_clsf_set
        )
    
    # CREATE DATALOADERS
    train_repr_loader = torch.utils.data.DataLoader(
        train_repr_set,
        batch_size=args.batch_size//world_size,
        drop_last=True,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=train_repr_sampler
    )
    eval_repr_loader = torch.utils.data.DataLoader(
        eval_repr_set,
        batch_size=args.batch_size//world_size,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=eval_repr_sampler
    )
    train_clsf_loader = torch.utils.data.DataLoader(
        train_clsf_set,
        batch_size=args.batch_size_clsf//world_size,
        drop_last=True,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=train_clsf_sampler
    )
    eval_clsf_loader = torch.utils.data.DataLoader(
        eval_clsf_set,
        batch_size=args.batch_size_clsf//world_size,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=eval_clsf_sampler
    )


    ######### MODEL and CO #########
    # CREATE MODEL
    model = EquiMod(args.resnet_type, args.z_dim, args.y_dim, train_repr_set.nb_params, args.dataset_name=='cifar10', args.proj_head_eq_layers, args.proj_head_t_layers, args.predictor_eq_layers).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    if args.hardware == 'multi-gpu':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # CREATE CRITERION
    criterion_repr = {
        'NTXentLoss': NTXentLoss()
    }

    # CREATE OPTIMIZER
    wd_params, no_wd_params = get_model_params_groups(model)
    optimizer_repr = LARSW(
        [
            {'params': no_wd_params, 'weight_decay': 0.0, 'lars_weight': 0.0},
            {'params': wd_params}
        ],
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        eta=args.eta
    )
    
    if checkpoint is not None:
        optimizer_repr.load_state_dict(checkpoint['optimizer_repr'])

    # CREATE SCHEDULERS
    last_epoch = checkpoint['last_epoch'] if checkpoint is not None else -1 
    last_step = last_epoch*len(train_repr_loader) if checkpoint is not None else -1 
    warm_up_scheduler_repr = torch.optim.lr_scheduler.LinearLR(
        optimizer_repr, start_factor=1e-8, end_factor=1, total_iters=args.nb_epochs_warmup*len(train_repr_loader), last_epoch=last_step
    )

    cosine_scheduler_repr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_repr, (args.nb_epochs - args.nb_epochs_warmup)*len(train_repr_loader), last_epoch=last_step, eta_min=1e-8
    )
    if checkpoint is not None:
        warm_up_scheduler_repr.load_state_dict(checkpoint['warm_up_scheduler_repr'])
        cosine_scheduler_repr.load_state_dict(checkpoint['cosine_scheduler_repr'])

    # CREATE SCALER
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'mixed' else None


    ######### ITERATE EPOCHS #########
    tracker.reset('epoch')
    tracker.add('epoch', 0, max(0, last_epoch))
    for epoch in range(max(0, last_epoch + 1), args.nb_epochs):
        start = time.time()
        tracker.add('epoch', 0)

        if args.hardware == 'multi-gpu':
            train_repr_sampler.set_epoch(epoch)
            eval_repr_sampler.set_epoch(epoch)

        scheduler_repr = None
        if epoch < args.nb_epochs_warmup:
            scheduler_repr = warm_up_scheduler_repr
        else:
            scheduler_repr = cosine_scheduler_repr

        train_repr(model, optimizer_repr, criterion_repr, train_repr_loader, scheduler_repr, scaler, args, device, is_master, global_rank, world_size)
        eval_repr(model, criterion_repr, eval_repr_loader, args, device, is_master, global_rank, world_size)

        if is_master:
            if scheduler_repr is not None:
                writer.add_scalar('lr_repr', scheduler_repr.get_last_lr()[0], epoch)

            print('\rEpoch: {:>3}/{:<3}  |  Train loss: {:e}  |  Eval loss: {:e}  |  Time taken: {}'
                .format(epoch + 1, args.nb_epochs, tracker.get('train_loss_repr').average, tracker.get('eval_loss_repr').average, time.time() - start))

        # SAVE MODEL       
        if is_master:
            checkpoint_data = {
                'last_epoch': epoch,
                'log_dir': log_dir,
                'model': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                'optimizer_repr': optimizer_repr.state_dict(),
                'warm_up_scheduler_repr': warm_up_scheduler_repr.state_dict(),
                'cosine_scheduler_repr': cosine_scheduler_repr.state_dict()
            }

            # Save when it is time/last epoch
            if (epoch + 1)%args.save_every == 0 or (epoch + 1) == args.nb_epochs:
                if args.computer == 'jeanzay':
                    checkpoint_file = './checkpoints/{}_{}_{}.pt'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], epoch)
                else:
                    checkpoint_file = './checkpoints/{}_{}_{}.pt'.format('expe', expe_time, epoch)

                torch.save(checkpoint_data, checkpoint_file)
            
            # Save in case of crash
            if args.computer == 'jeanzay':
                checkpoint_file = './checkpoints/{}_{}_{}.pt'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], 'even' if epoch%2 == 0 else 'odd')
            else:
                checkpoint_file = './checkpoints/{}_{}_{}.pt'.format('expe', expe_time, 'even' if epoch%2 == 0 else 'odd')
            torch.save(checkpoint_data, checkpoint_file)


        # If it is time to linear eval/last epoch
        if (epoch + 1)%args.clsf_every == 0 or (epoch + 1) == args.nb_epochs:
            linear_eval(args, model, train_clsf_loader, eval_clsf_loader, train_clsf_sampler, eval_clsf_sampler, scaler, global_rank, local_rank, world_size, is_master, device)
        
    if is_master:
        writer.close()



if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--computer',
        choices=['jeanzay', 'other'], default='other',
        help='Computer used')

    parser.add_argument('--hardware',
        choices=['multi-gpu', 'mono-gpu', 'cpu'], default='cpu',
        help='Type of hardware to use')

    parser.add_argument('--precision',
        choices=['normal', 'mixed'], default='normal',
        help='Type of precision to use')

    parser.add_argument('--nb_workers', type=int,
        default=10,
        help='Number of workers')

    parser.add_argument('--expe_name',
        default='default',
        help='Name of the expe')

    parser.add_argument('--dataset_name',
        choices=['cifar10', 'imagenet'], default='cifar10',
        help='Name of the dataset')

    parser.add_argument('--resnet_type',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18',
        help='Type of resnet')

    parser.add_argument('--lmbd', type=float,
        default=1.0,
        help='Ponderation of the equivariant loss')
    
    parser.add_argument('--nb_epochs', type=int,
        default=800,
        help='Number of epochs')
    
    parser.add_argument('--nb_epochs_warmup', type=int,
        default=10,
        help='Number of epochs for the lr warmup')

    parser.add_argument('--batch_size', type=int,
        default=512,
        help='Size of the global batch')

    parser.add_argument('--lr_init', type=float,
        default=4.0,
        help='Initial learning rate')

    parser.add_argument('--momentum', type=float,
        default=0.9,
        help='Momentum')

    parser.add_argument('--weight_decay', type=float,
        default=1e-6,
        help='Weight decay')

    parser.add_argument('--eta', type=float,
        default=1e-3,
        help='Eta LARS parameter')

    parser.add_argument('--z_dim', type=int,
        default=128,
        help='Dimension of z')

    parser.add_argument('--y_dim', type=int,
        default=128,
        help='Dimension of y')

    parser.add_argument('--temperature_z', type=float,
        default=0.5,
        help='Temperature of the NTXent loss for the z')

    parser.add_argument('--temperature_y', type=float,
        default=0.2,
        help='Temperature of the NTXent loss for the y')
    
    parser.add_argument('--clsf_every', type=int,
        default=100,
        help='Number of epochs between each linear eval')

    parser.add_argument('--save_every', type=int,
        default=100,
        help='Number of epochs between each checkpoint')

    parser.add_argument('--nb_epochs_clsf', type=int,
        default=90,
        help='Number of epochs for linear eval')

    parser.add_argument('--batch_size_clsf', type=int,
        default=256,
        help='Size of the global batch for linear eval')

    parser.add_argument('--lr_init_clsf', type=float,
        default=0.2,
        help='Initial learning rate for linear eval')

    parser.add_argument('--momentum_clsf', type=float,
        default=0.9,
        help='Momentum for linear eval')

    parser.add_argument('--weight_decay_clsf', type=float,
        default=0.0,
        help='Weight decay for linear eval')

    parser.add_argument('--checkpoint',
        default=None,
        help='Path to the checkpoint to start from')

    parser.add_argument('--proj_head_eq_layers',
        default="2048-2048-",
        help='Size of layers of eq head')
    
    parser.add_argument('--proj_head_t_layers',
        default="128",
        help='Size of layers of param head')

    parser.add_argument('--predictor_eq_layers',
        default="one",
        help='Size of layer of eq net')

    # Parse arguments
    args = parser.parse_args()

    # Run main fonction
    run(args)
