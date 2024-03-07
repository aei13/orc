import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import wandb

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torch.utils.data import Subset

from mr_dataset import build_dataset
from engine_2d import train_one_epoch, evaluate, evaluate_save
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Off-ResNet training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--wandb', action='store_true', help='Log using wandb')
    parser.add_argument('--project', default='orc-3d-kneetr', type=str, help='Project name of wandb')
    parser.add_argument('--id', default='abcd', type=str, help='Experiment ID of wandb')
    parser.add_argument('--crop', action='store_true', help='Crop 3D input to input size')
    parser.add_argument('--crop_eval', action='store_true', help='Crop 3D input to input size while evaluation')
    parser.add_argument('--input_type', default='RIMP', choices=['RI', 'MP', 'M', 'RIMP', 'MMM'], type=str, help='Input type')
    parser.add_argument('--output_type', default='RIMP', choices=['RI', 'MP', 'M', 'RIMP', 'MMM'], type=str, help='Output type')
    parser.add_argument('--loss_lda_l1', default=[0.25, 0.25, 0.25, 0.25], type=float, nargs=4,
                        help='L1 loss coefficients of R/I/M/P ([float, float, float, 0.] for MMM)')
    parser.add_argument('--loss_lda_per', default=[0., 0., 0., 0.], type=float, nargs=4,
                        help='Perceptual loss coefficients of R/I/M/P ([float, float, float, 0.] for MMM)')
    parser.add_argument('--idx_layers_per', default=[3], type=int, nargs='*',
                        help='Layer index of VGGNet for perceptual loss (from 0)')
    parser.add_argument('--type_layers_per', default='conv', choices=['conv', 'relu', 'pool'], type=str, help='Perceptual loss layer type')
    parser.add_argument('--loss_per_type', default='normal', choices=['normal', 'sub', 'mul', 'frac', 'mulfrac', 'mulsub'], type=str, help='Perceptual loss type')
    parser.add_argument('--act_type', default='gelu', choices=['gelu', 'swish', 'mish'], type=str, help='Activation function type')
    parser.add_argument('--translation_type', default='None', choices=['all', 'odd', 'None'], type=str, help='Translated input - returned output(mean or median)')
    parser.add_argument('--translation_iter', default=3, type=int, help='# of iteration')
    parser.add_argument('--translation_fill_type', default='mean', choices=['mean', 'median'], type=str, help='Pixel filling type when translation')
    parser.add_argument('--patchsize_stride', default=[0, 0], type=int, nargs=2, help='Patch size / stride of patch splitted inference')
    parser.add_argument('--eval_target', action='store_true', help='Use target (reference) images as model input')

    # Model parameters
    parser.add_argument('--model', default='offresnet2d', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=64, type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=3e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data-path', default='/raid/MRI/Cones_Phantom/patch2d/', type=str, help='dataset path')
    parser.add_argument('--data-set', default='MRI2D', choices=['MRI2D', 'MRI3D'],
                        type=str, help='Dataset path')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_train', action='store_true', help='Perform evaluation on trainset')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--test-freq', default=1, type=int, help='Number of epochs between \
                                                                  validation runs.')
    parser.add_argument("--pretrained", default=None, type=str, help='Path to pre-trained checkpoint')
    parser.add_argument('--dist_eval', action='store_true', help='distributed evaluation')

    return parser


def main(args):
    utils.init_distributed_mode(args)

    wandb_name = os.path.basename(Path(args.output_dir))
    run = None
    
    if not args.eval and args.wandb:
        if args.distributed:
            if args.rank == 0:  # only on main process
                # Initialize wandb run
                if args.start_epoch > 0:
                    wandb_id = args.id
                    run = wandb.init(id=wandb_id, project=args.project, name=wandb_name, resume="allow")
                else:
                    wandb_id = args.id
                    run = wandb.init(id=wandb_id, project=args.project, name=wandb_name)
                    with open('wandb/latest_id', 'w') as f:
                        f.write(wandb_id)
                wandb.config = args
        else:
            # Initialize wandb run
            if args.start_epoch > 0:
                wandb_id = args.id
                run = wandb.init(id=wandb_id, project=args.project, name=wandb_name, resume="allow")
            else:
                wandb_id = args.id
                run = wandb.init(id=wandb_id, project=args.project, name=wandb_name)
                with open('wandb/latest_id', 'w') as f:
                    f.write(wandb_id)
            wandb.config = args
    
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, _ = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    dataset_train = Subset(dataset_train, np.arange(8))
    if not args.eval:
        dataset_val = Subset(dataset_val, np.arange(10))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        args=vars(args)
    )

    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)
        print("Checkpoint loaded")

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    criterion = torch.nn.L1Loss()

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    resume_path = args.resume
    if args.resume and os.path.exists(resume_path):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading from checkpoint ...")
            checkpoint = torch.load(resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        if args.eval_train:
            evaluate_save(data_loader_val, model, device, args)
        else:
            evaluate_save(data_loader_val, model, device, args)
        print('evaluate_save done.')
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 100.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args=args
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': 'None',
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        if (epoch % args.test_freq == 0) or (epoch == args.epochs - 1):
            test_stats = evaluate(data_loader_val, model, device, args, run)

            print(f"Loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.1f}")
            min_loss = min(min_loss, test_stats["loss"])
            print(f'Min Loss: {min_loss:.6f}')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            if not args.eval and args.wandb:
                if args.distributed:
                    if args.rank == 0:  # only on main process
                        wandb.log(log_stats)
                else:
                    wandb.log(log_stats)

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Off-Resonance Correction models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'offres' in args.model:
        import offresnet2d
    else:
        import restormer_2d
    main(args)
