import math
import os
import sys
import numpy as np
from typing import Iterable

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models._utils import IntermediateLayerGetter

from skimage.metrics import structural_similarity as ssim

import utils
import lpips
import wandb


class VGGLoss(nn.Module):
    models = {'vgg11': models.vgg11, 'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layers=[3], layer_type='conv', shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
        pooling_layers = [4, 9, 16, 23, 30]
        if layer_type == 'conv':
            detailed_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 25, 28]
        elif layer_type == 'relu':
            detailed_layers = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
        elif layer_type == 'pool':
            detailed_layers = pooling_layers
        else:
            raise NotImplementedError('Please select Perceptual loss layer type in [conv, relu, pool].')
        detailed_layers = [detailed_layers[idx] for idx in layers]
        self.layers = {str(i):str(i) for i in detailed_layers}
        self.model = self.models[model](weights=models.VGG16_Weights.DEFAULT).features
        self.model = IntermediateLayerGetter(self.model, return_layers=self.layers)
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, x):
        return self.model(self.normalize(x))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, output, target, type_per='normal'):
        loss = 0.
        sep = output.shape[0]
        if input is None or type_per == 'normal':
            batch = torch.cat([output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:]
                loss += F.mse_loss(output_feats, target_feats, reduction=self.reduction)    # Minimize |target_feats-output_feats|
        elif type_per == 'sub':
            batch = torch.cat([input, output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                input_feats, output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:2*sep], multi_feats[feat_key][2*sep:]
                loss -= 0.1 * F.mse_loss(output_feats, input_feats, reduction=self.reduction)     # Maximize |input_feats-output_feats|
                loss += F.mse_loss(output_feats, target_feats, reduction=self.reduction)    # Minimize |target_feats-output_feats|
        elif type_per == 'mul':
            batch = torch.cat([input, output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                input_feats, output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:2*sep], multi_feats[feat_key][2*sep:]
                salient_feats = torch.abs(target_feats - input_feats)
                loss += torch.mean(salient_feats * F.mse_loss(output_feats, target_feats, reduction='none'))    # Minimize salient_feats*|target_feats-output_feats|
        elif type_per == 'frac':
            batch = torch.cat([input, output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                input_feats, output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:2*sep], multi_feats[feat_key][2*sep:]
                loss += F.mse_loss(output_feats, target_feats, reduction=self.reduction) / F.mse_loss(output_feats, input_feats, reduction=self.reduction)    # Minimize |target_feats-output_feats| and Maximize |input_feats-output_feats|
        elif type_per == 'mulfrac':
            batch = torch.cat([input, output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                input_feats, output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:2*sep], multi_feats[feat_key][2*sep:]
                salient_feats = torch.abs(target_feats - input_feats)
                loss += torch.mean(salient_feats * F.mse_loss(output_feats, target_feats, reduction='none')) / F.mse_loss(output_feats, input_feats, reduction=self.reduction)    # Minimize salient_feats*|target_feats-output_feats|
        elif type_per == 'mulsub':
            batch = torch.cat([input, output, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            multi_feats = self.get_features(batch)
            for feat_key in multi_feats:
                input_feats, output_feats, target_feats = multi_feats[feat_key][:sep], multi_feats[feat_key][sep:2*sep], multi_feats[feat_key][2*sep:]
                salient_feats = torch.abs(target_feats - input_feats)
                loss -= 0.1 * torch.mean(salient_feats * F.mse_loss(output_feats, input_feats, reduction='none'))     # Maximize |input_feats-output_feats|
                loss += torch.mean(salient_feats * F.mse_loss(output_feats, target_feats, reduction='none'))    # Minimize |target_feats-output_feats|
        else:
            raise NotImplementedError('Please select Perceptual loss type in [normal, sub, mul].')
        return loss

def PSNR(img1, img2):
    psnr_slice = 0.
    for i in range(len(img1)):
        mse_ = np.mean(np.abs(img1[i] - img2[i]) ** 2)
        if mse_ == 0:
            psnr_slice += 100
        else:
            psnr_slice += 20 * math.log10(1. / np.sqrt(mse_))
    return psnr_slice / (i+1)

def SSIM(img1, img2):
    ssim_slice = 0.
    for i in range(len(img1)):
        ssim_slice += ssim(img1[i], img2[i], data_range=img2[i].max()-img2[i].min())
    return ssim_slice / (i+1)

def LPIPS(img1, img2, device):
    import torch
    img1 = torch.from_numpy(img1).to(torch.float32).to(device)
    img2 = torch.from_numpy(img2).to(torch.float32).to(device)
    alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_slice = 0.
    for i in range(len(img1)):
        lpips_slice += alex(img1[i], img2[i], normalize=True).item()
    return lpips_slice / (i+1)

def eval_scores(img1, img2, device):
    '''
    Calculating Scores of Magnitude Images
    img1: Output Image, (Slice, B, C, H, W)
    img2: Ground Truth Image, (Slice, B, C, H, W)
    '''
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if img1.shape[2] == 4:  # RIMP
        img1 = img1[:,0,2]
        img2 = img2[:,0,2]
    elif img1.shape[2] == 3:  # MMM
        img1 = np.mean(img1, axis=(1,2))
        img2 = np.mean(img2, axis=(1,2))
    elif img1.shape[2] == 2 or img1.shape[2] == 1:  # MP (Ignore RI case) or M
        print('Assuming FIRST channel of array as magnitude.')
        img1 = img1[:,0,0]
        img2 = img2[:,0,0]
    else:
        print('Check shape (channel)!! Now shape: {}'.format(img1.shape))
        psnr_score = 0.
        ssim_score = 0.
        lpips_score = 0.
        return psnr_score, ssim_score, lpips_score
    
    psnr_score = PSNR(img1, img2)
    ssim_score = SSIM(img1, img2)
    lpips_score = 0.1 #LPIPS(img1, img2, device)
    return psnr_score, ssim_score, lpips_score


def tensor2numpy(tensor, real2comp=False, out_type=np.double, min_max=(-1, 1)):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # _tensor = _tensor.clamp_(*min_max).cpu()
        _tensor = _tensor.cpu()

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = _tensor.numpy()
            if img_np.shape[1] == 1 and img_np.dtype == complex:  # complex image array
                img_np = np.squeeze(img_np, axis=1)
            elif img_np.shape[1] == 4:  # real/imag/mag/phase image array
                img_np = img_np#[:,2:]   # note that it outputs mag/ph 2ch image, not complex 1ch image
            else:
                if real2comp:
                    img_np = img_np[:, 0, ...] + 1j * img_np[:, 1, ...]
        elif n_dim == 3:
            img_np = _tensor.numpy()
            if img_np.shape[0] == 1 and img_np.dtype == complex:  # complex image array
                img_np = np.squeeze(img_np, axis=0)
            elif img_np.shape[0] == 4:  # real/imag/mag/phase image array
                img_np = img_np#[2:]
            else:
                if real2comp:
                    img_np = img_np[0, ...] + 1j * img_np[1, ...]
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result # np.concatenate(result, axis=1).squeeze()


def _affine(sample, num_iter=3, forward=True, fill_type='mean'):
    # forward=True: Translate image in x,y,xy direction
    # forward=False: Translate image back in minus x,y,xy direction
    # considering repeated maxpooling in VGG, it can repeat translation (1,3,5,7,...)
    samples = []
    if forward: # add 3*num_iter translated samples to the original sample
        samples.append(sample)
        for n in range(num_iter):
            samples.append(TF.affine(sample, 0, (n+1,0), 1, 0))
            samples.append(TF.affine(sample, 0, (0,n+1), 1, 0))
            samples.append(TF.affine(sample, 0, (n+1,n+1), 1, 0))
        sample = torch.cat(samples, dim=0)
    else:       # translate back 3*num_iter translated samples and average them to get the original sample
        samples.append(sample[0])
        for n in range(num_iter):
            samples.append(TF.affine(sample[3*n+1], 0, (-(n+1),0), 1, 0))
            samples.append(TF.affine(sample[3*n+2], 0, (0,-(n+1)), 1, 0))
            samples.append(TF.affine(sample[3*n+3], 0, (-(n+1),-(n+1)), 1, 0))
        sample_all = torch.stack(samples, dim=0)
        if fill_type == 'mean':
            sample = sample_all.mean(dim=0, keepdim=True)
        elif fill_type == 'median':
            sample = sample_all.median(dim=0, keepdim=True).values
        
    return sample

def _affine_odd(sample, num_iter=3, forward=True, fill_type='mean'):
    # forward=True: Translate image in x,y,xy direction
    # forward=False: Translate image back in minus x,y,xy direction
    # considering repeated maxpooling in VGG, it can repeat translation (1,3,5,7,...)
    samples = []
    if forward: # add 3*num_iter translated samples to the original sample
        samples.append(sample)
        for n in range(num_iter):
            samples.append(TF.affine(sample, 0, (2*n+1,0), 1, 0))
            samples.append(TF.affine(sample, 0, (0,2*n+1), 1, 0))
            samples.append(TF.affine(sample, 0, (2*n+1,2*n+1), 1, 0))
        sample = torch.cat(samples, dim=0)
    else:       # translate back 3*num_iter translated samples and average them to get the original sample
        samples.append(sample[0])
        for n in range(num_iter):
            samples.append(TF.affine(sample[3*n+1], 0, (-(2*n+1),0), 1, 0))
            samples.append(TF.affine(sample[3*n+2], 0, (0,-(2*n+1)), 1, 0))
            samples.append(TF.affine(sample[3*n+3], 0, (-(2*n+1),-(2*n+1)), 1, 0))
        sample_all = torch.stack(samples, dim=0)
        if fill_type == 'mean':
            sample = sample_all.mean(dim=0, keepdim=True)
        elif fill_type == 'median':
            sample = sample_all.median(dim=0, keepdim=True).values
        
    return sample

# Patchify images
def patchify(img, kernel_size, stride):
    patches = img.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    unfold_shape = patches.size()   # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]
    patches = patches.permute(0,2,3,1,4,5).reshape(-1, unfold_shape[1], kernel_size, kernel_size)   # [B, C, nb_patches_all, kernel_size*kernel_size]
    return patches, unfold_shape

# Reshape back
def reconstruct(patches, output_size, unfold_shape):
    # reshape output to match F.fold input
    patches = patches.reshape(unfold_shape[0], -1, unfold_shape[1], unfold_shape[-2]*unfold_shape[-1])
    patches = patches.permute(0, 2, 3, 1) 
    patches = patches.contiguous().view(unfold_shape[0], unfold_shape[1]*unfold_shape[-2]*unfold_shape[-1], -1)
    output = F.fold(patches, output_size=output_size, kernel_size=unfold_shape[-2], stride=unfold_shape[-1])
    return output

# Reshape back with overlapping
def reconstruct_overlap(patches, output_size, stride):
    patches_shape = patches.shape
    num_patches = patches_shape[0]
    num_rows = (output_size[0] - patches_shape[-2]) // (stride - 1) + 1
    num_cols = (output_size[1] - patches_shape[-1]) // (stride - 1) + 1
    num_batches = num_patches // (num_rows*num_cols)
    output = torch.zeros(num_batches, patches_shape[1], output_size[0], output_size[1]).to(patches.device)
    count = torch.zeros(num_batches, patches_shape[1], output_size[0], output_size[1]).to(patches.device)
    for i in range(num_patches):
        batch = i //(num_rows*num_cols)
        row = i // num_rows % num_cols
        col = i % num_cols
        h_start = row * stride#patches_shape[-2]
        w_start = col * stride#patches_shape[-1]
        h_end = h_start + patches_shape[-2]
        w_end = w_start + patches_shape[-1]
        output[batch:batch+1, :, h_start:h_end, w_start:w_end] += patches[i:i+1]
        count[batch:batch+1, :, h_start:h_end, w_start:w_end] += 1
    output /= count
    return output

def convert_samples(samples, args):
    if args.input_type == 'RI' and args.output_type == 'MP':
        samples = torch.stack([torch.sqrt(samples[:,0]**2 + samples[:,1]**2), torch.atan2(samples[:,1], samples[:,0])], 1)
    elif args.input_type == 'RI' and args.output_type == 'M':
        samples = torch.stack([torch.sqrt(samples[:,0]**2 + samples[:,1]**2)], 1)
    elif args.input_type == 'RIMP' and args.output_type == 'RI':
        samples = samples[:,:2]
    elif args.input_type == 'RIMP' and args.output_type == 'MP':
        samples = samples[:,2:]
    elif args.input_type == 'RIMP' and args.output_type == 'M':
        samples = samples[:,2:3]
    elif args.input_type == 'MP' and args.output_type == 'M':
        samples = samples[:,0:1]
    elif args.input_type == args.output_type:
        samples = samples
    else:
        raise NotImplementedError('Check in/out channel type or implement new cases.')
    return samples

def train_one_epoch(model: torch.nn.Module, criterion: nn.L1Loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # ldas = torch.tensor(args.loss_l1_lda)
    ldas_l1 = [float(elem or 0) for elem in args.loss_lda_l1]
    ldas_per = [float(elem or 0) for elem in args.loss_lda_per]
    if sum(ldas_l1+ldas_per) == 0.:
        raise RuntimeError('You need to specify ldas for losses!! (at least l1)')
    
    max_loop_length = len(args.output_type)
    
    if sum(ldas_per) > 0.:
        criterion_vgg = VGGLoss(layers=args.idx_layers_per.copy(), layer_type=args.type_layers_per).to(device)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]
        batch_size = samples.shape[0]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            loss = 0.
            if sum(ldas_l1) > 0.:
                loss_l1 = 0.
                for i, lda_l1 in enumerate(ldas_l1):
                    if i >= max_loop_length:
                        break
                    if lda_l1 > 0.:
                        loss_l1 += lda_l1 * criterion(outputs[:,i], targets[:,i])
                loss = loss + loss_l1
            if sum(ldas_per) > 0.:
                loss_perceptual = 0.
                for i, lda_per in enumerate(ldas_per):
                    if i >= max_loop_length:
                        break
                    if lda_per > 0.:
                        if args.loss_per_type == 'normal':
                            loss_perceptual += lda_per * criterion_vgg(None, outputs[:,i:i+1].repeat(1,3,1,1), targets[:,i:i+1].repeat(1,3,1,1), type_per=args.loss_per_type) / (3. * len(args.idx_layers_per))  # divide by 3(#repeat) and #layers
                        else:
                            samples = convert_samples(samples, args)
                            loss_perceptual += lda_per * criterion_vgg(samples[:,i:i+1].repeat(1,3,1,1), outputs[:,i:i+1].repeat(1,3,1,1), targets[:,i:i+1].repeat(1,3,1,1), type_per=args.loss_per_type) / (3. * len(args.idx_layers_per))  # divide by 3(#repeat) and #layers
                loss = loss + loss_perceptual
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if sum(ldas_l1) > 0.:
            metric_logger.meters['loss_l1'].update(loss_l1.item(), n=batch_size)
        if sum(ldas_per) > 0.:
            metric_logger.meters['loss_perceptual'].update(loss_perceptual.item(), n=batch_size)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, run=None):
    if args.eval_train:
        save_path = os.path.join(args.output_dir, 'results_train_' + os.path.basename(args.data_path))
    else:
        save_path = os.path.join(args.output_dir, 'results_test_' + os.path.basename(args.data_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    ldas_l1 = [float(elem or 0) for elem in args.loss_lda_l1]
    ldas_per = [float(elem or 0) for elem in args.loss_lda_per]
    
    max_loop_length = len(args.output_type)
    
    if sum(ldas_per) > 0.:
        criterion_vgg = VGGLoss(layers=args.idx_layers_per.copy(), layer_type=args.type_layers_per).to(device)
    criterion = torch.nn.L1Loss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    output_img = []
    gt_img = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets = batch[0], batch[1]
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if args.translation_type == 'all':
            samples = _affine(samples, num_iter=args.translation_iter, forward=True, fill_type=args.translation_fill_type)
        elif args.translation_type == 'odd':
            samples = _affine_odd(samples, num_iter=args.translation_iter, forward=True, fill_type=args.translation_fill_type)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(samples)
        if args.translation_type == 'all':
            outputs = _affine(outputs, num_iter=args.translation_iter, forward=False, fill_type=args.translation_fill_type)
        elif args.translation_type == 'odd':
            outputs = _affine_odd(outputs, num_iter=args.translation_iter, forward=False, fill_type=args.translation_fill_type)

        loss = 0.
        if sum(ldas_l1) > 0.:
            loss_l1 = 0.
            for i, lda_l1 in enumerate(ldas_l1):
                if i >= max_loop_length:
                    break
                if lda_l1 > 0.:
                    loss_l1 += lda_l1 * criterion(outputs[:,i], targets[:,i])
            loss = loss + loss_l1
        
        if sum(ldas_per) > 0.:
            loss_perceptual = 0.
            for i, lda_per in enumerate(ldas_per):
                if i >= max_loop_length:
                    break
                if lda_per > 0.:
                    if args.loss_per_type == 'normal':
                        loss_perceptual += lda_per * criterion_vgg(None, outputs[:,i:i+1].repeat(1,3,1,1), targets[:,i:i+1].repeat(1,3,1,1), type_per=args.loss_per_type) / (3. * len(args.idx_layers_per))  # divide by 3(#repeat) and #layers
                    else:
                        samples = convert_samples(samples, args)
                        loss_perceptual += lda_per * criterion_vgg(samples[:,i:i+1].repeat(1,3,1,1), outputs[:,i:i+1].repeat(1,3,1,1), targets[:,i:i+1].repeat(1,3,1,1), type_per=args.loss_per_type) / (3. * len(args.idx_layers_per))  # divide by 3(#repeat) and #layers
            loss = loss + loss_perceptual
        
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        if sum(ldas_l1) > 0.:
            metric_logger.meters['loss_l1'].update(loss_l1.item(), n=batch_size)
        if sum(ldas_per) > 0.:
            metric_logger.meters['loss_perceptual'].update(loss_perceptual.item(), n=batch_size)
    
        # samples = tensor2numpy(samples)
        outputs = tensor2numpy(outputs)
        targets = tensor2numpy(targets)
        
        # input_img.append(samples)
        output_img.append(outputs)
        gt_img.append(targets)
        
    print('Calculating Scores...')
    score_psnr, score_ssim, score_lpips = eval_scores(output_img, gt_img, device)
    metric_logger.meters['score_psnr'].update(score_psnr)
    metric_logger.meters['score_ssim'].update(score_ssim)
    metric_logger.meters['score_lpips'].update(score_lpips)
    
    # Save Input, Output, Ground Truth Images of (Slice, C, H, W) (ex. (900, 4, 240, 240))
    # np.save(os.path.join(save_path, 'input'+str(0)), input_img)
    np.save(os.path.join(save_path, 'output'+str(0)), output_img)
    np.save(os.path.join(save_path, 'gt'+str(0)), gt_img)
    
    if run is not None:
        z_idx = min(190, len(output_img)-1)
        print('Uploading Example Slice (z={}) to wandb...'.format(z_idx))
        if output_img[0].shape[1] == 4:     # RIMP
            samples = wandb.Image(output_img[z_idx][0,2], caption="Output Example")
        elif output_img[0].shape[1] == 3:   # MMM
            samples = wandb.Image(np.mean(output_img[z_idx], axis=(0,1)), caption="Output Example")
        elif output_img[0].shape[1] == 2 or output_img[0].shape[1] == 1:   # MP (Ignore RI) or M
            samples = wandb.Image(output_img[z_idx][0,0], caption="Output Example")
        else:
            samples = wandb.Image(output_img[z_idx][0,0], caption="Output Example")
            
        run.log({"output_example": samples})
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Total loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    if sum(ldas_l1) > 0.:
        print('* L1 {loss_l1.global_avg:.3f}'.format(loss_l1=metric_logger.loss_l1))
    if sum(ldas_per) > 0.:
        print('* Perceptual {loss_perceptual.global_avg:.3f}'.format(loss_perceptual=metric_logger.loss_perceptual))
    
    print('* PSNR {score_psnr.global_avg:.2f}, SSIM {score_ssim.global_avg:.2f},\
        LPIPS {score_lpips.global_avg:.3f}'.format(
        score_psnr=metric_logger.score_psnr, score_ssim=metric_logger.score_ssim,
        score_lpips=metric_logger.score_lpips))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_save(data_loader, model, device, args):
    if args.eval_train:
        save_path = os.path.join(args.output_dir, 'results_train_' + os.path.basename(args.data_path))
    elif args.eval_target:
        save_path = os.path.join(args.output_dir, 'results_target_' + os.path.basename(args.data_path))
    else:
        save_path = os.path.join(args.output_dir, 'results_test_' + os.path.basename(args.data_path))
    if args.translation_type == 'all':
        save_path = save_path + '_trans' + str(args.translation_iter) + '_' + args.translation_fill_type
    elif args.translation_type == 'odd':
        save_path = save_path + '_trans' + str(2*args.translation_iter-1) + '_' + args.translation_fill_type
    if args.patchsize_stride[0] > 0:
        save_path = save_path + '_patch{}{}'.format(str(args.patchsize_stride[0]), str(args.patchsize_stride[1]))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    input_img = []
    output_img = []
    gt_img = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples, targets = batch[0], batch[1]
        samples_tosave = samples.clone().detach()
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if args.translation_type == 'all':
            samples = _affine(samples, num_iter=args.translation_iter, forward=True, fill_type=args.translation_fill_type)
        elif args.translation_type == 'odd':
            samples = _affine_odd(samples, num_iter=args.translation_iter, forward=True, fill_type=args.translation_fill_type)
            
        with torch.cuda.amp.autocast():
            if args.patchsize_stride[0] > 0:
                patch_size = args.patchsize_stride[0]
                stride = args.patchsize_stride[1]
                outputs = torch.zeros_like(samples, device=device)
                for b in range(len(samples)):
                    output = torch.zeros_like(samples[b:b+1], device=device)
                    count = torch.zeros_like(samples[b:b+1], device=device)
                    
                    # Patch
                    patches = samples[b:b+1].unfold(2, patch_size, stride).unfold(3, patch_size, stride).contiguous()
                    patches = patches.permute(0,2,3,1,4,5).reshape(-1, patches.shape[1], patch_size, patch_size)
                    
                    # Patchwise inference
                    patch_outputs = model(patches)
                    
                    # Accumulate patches
                    h = (samples.shape[2] - patch_size) // stride + 1
                    w = (samples.shape[3] - patch_size) // stride + 1
                    for i in range(h):
                        for j in range(w):
                            if args.output_type == 'M':
                                output[..., i*stride:i*stride+patch_size, j*stride:j*stride+patch_size] += patch_outputs.reshape(-1, h, w, 1, patch_size, patch_size).permute(0,3,1,2,4,5)[..., i, j, :, :]
                            else:
                                output[..., i*stride:i*stride+patch_size, j*stride:j*stride+patch_size] += patch_outputs.reshape(-1, h, w, patches.shape[1], patch_size, patch_size).permute(0,3,1,2,4,5)[..., i, j, :, :]
                            count[..., i*stride:i*stride+patch_size, j*stride:j*stride+patch_size] += 1
                    output /= count
                    outputs[b:b+1] += output
                del patches
                del patch_outputs
                del count
            else:
                outputs = model(samples)
            
        if args.translation_type == 'all':
            outputs = _affine(outputs, num_iter=args.translation_iter, forward=False, fill_type=args.translation_fill_type)
        elif args.translation_type == 'odd':
            outputs = _affine_odd(outputs, num_iter=args.translation_iter, forward=False, fill_type=args.translation_fill_type)
    
        samples = tensor2numpy(samples_tosave, min_max=(samples_tosave.min(), samples_tosave.max()))
        outputs = tensor2numpy(outputs, min_max=(outputs.min(), outputs.max()))
        targets = tensor2numpy(targets, min_max=(targets.min(), targets.max()))
        
        input_img.append(samples)
        output_img.append(outputs)
        gt_img.append(targets)
        
    # Save Input, Output, Ground Truth Images of (Slice, C, H, W) (ex. (900, 4, 240, 240))
    np.save(os.path.join(save_path, 'input'+str(0)), input_img)
    np.save(os.path.join(save_path, 'output'+str(0)), output_img)
    np.save(os.path.join(save_path, 'gt'+str(0)), gt_img)
