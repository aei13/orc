"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import time
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as TF
import torchvision.transforms.functional as TTF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from tqdm import tqdm

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3
    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=False,
                 requires_grad=False,
                 use_fid_inception=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    if version >= (0, 6):
        kwargs['init_weights'] = False

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def get_activations(files, model, batch_size=50, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Make sure that the number of samples is a multiple of the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the activations of the given tensor.
    """
    model.eval()

    if isinstance(files, np.ndarray):
        img = torch.from_numpy(files).to(torch.float32).to(device)
    else:
        img = files.to(torch.float32).to(device)
    n_dim = img.dim()
    img_shape = img.size()
    if n_dim == 4:
        if img_shape[1] == 1:   # assuming (B,1,W,H)
            img = img.repeat(1,3,1,1)
        elif img_shape[1] == 4:   # assuming (B,4,W,H)
            print('Image shape: {}, assuming the THIRD channel as MAGNITUDE.'.format(img_shape))
            img = img[:,2:3].repeat(1,3,1,1)
        else:   # assuming (B,C,W,H)
            print('Image shape: {}, assuming the FIRST channel as MAGNITUDE.'.format(img_shape))
            img = img[:,0:1].repeat(1,3,1,1)
    elif n_dim == 3:    # assuming (B,W,H)
        img = torch.unsqueeze(img, dim=1).repeat(1,3,1,1)
    else:
        raise NotImplementedError('Implementation is needed for the n_dim={} of input tensors. Or choose only magnitude channel for the input.'.format(n_dim))
    
    img_shape = img.size()
    if img_shape[0] % batch_size:
        print(('Warning: Image slice # should be divisable by batch size. '
               'Setting Image slice # to n*batch_size'))
        img = img[:img_shape[0]-(img_shape[0]%batch_size), ...]
        
    img = torch.reshape(img, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))
    pred_arr = np.empty((img_shape[0], dims))
    start_idx = 0
    for batch in tqdm(img):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device)
    # print('Shape of activation:', act.shape)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(file, model, batch_size, dims, device):
    m, s = calculate_activation_statistics(file, model, batch_size, dims, device)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, model=None, num_workers=1, wait=True, input_gt=False):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mymodel = False
    
    if wait:
        # while not os.path.exists(os.path.join(paths, 'input0.npy')):
        print('Waiting for input0.npy...')
        time.sleep(10)
        # while not os.path.exists(os.path.join(paths, 'output0.npy')):
        print('Waiting for output0.npy...')
        time.sleep(10)
        # while not os.path.exists(os.path.join(paths, 'gt0.npy')):
        print('Waiting for gt0.npy...')
        time.sleep(10)
    
    if not os.path.exists(paths):
        raise RuntimeError('Invalid path: %s' % paths)
    
    if input_gt:
        filename = 'input0.npy'
    else:
        filename = 'output0.npy'
    
    file = []
    if '_rimp/' in paths:
        raw = np.squeeze(np.load(os.path.join(paths, filename)))
        mag = np.abs(raw[:,0] + 1j*raw[:,1])
        file.append(mag)
        raw = np.squeeze(np.load(os.path.join(paths,'gt0.npy')))
        mag = np.abs(raw[:,0] + 1j*raw[:,1])
        file.append(mag)
    elif '_mp/' in paths:
        file.append(np.squeeze(np.load(os.path.join(paths, filename)))[:,0])     # [:,0] for mag, [:,1] for phase
        file.append(np.squeeze(np.load(os.path.join(paths, 'gt0.npy')))[:,0])
    elif '_m/' in paths:
        file.append(np.squeeze(np.load(os.path.join(paths, filename))))     # [:,0] for mag
        file.append(np.squeeze(np.load(os.path.join(paths, 'gt0.npy'))))
    elif '_ri/' in paths:
        raw = np.squeeze(np.load(os.path.join(paths, filename)))
        mag = np.abs(raw[:,0] + 1j*raw[:,1])
        file.append(mag)
        raw = np.squeeze(np.load(os.path.join(paths,'gt0.npy')))
        mag = np.abs(raw[:,0] + 1j*raw[:,1])
        file.append(mag)
    else:
        file.append(np.squeeze(np.load(os.path.join(paths, filename))))
        file.append(np.squeeze(np.load(os.path.join(paths, 'gt0.npy'))))
    
    m1, s1 = compute_statistics_of_path(file[0], model, batch_size,
                                        dims, device, num_workers, mymodel)
    m2, s2 = compute_statistics_of_path(file[1], model, batch_size,
                                        dims, device, num_workers, mymodel)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_tensors(tensor1, tensor2, batch_size, device, dims, model=None):
    """Calculates the FID of two tensors
    tensor1: Output Magnitude Tensor, stack of 2D images (ex. (900, 240, 240))
    tensor2: Ground Truth Magnitude Tensor, stack of 2D images (ex. (900, 240, 240))
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    m1, s1 = calculate_activation_statistics(tensor1, model, batch_size,
                                        dims, device)
    m2, s2 = calculate_activation_statistics(tensor2, model, batch_size,
                                        dims, device)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def _affine(sample, num_iter=3, batch_size=45, forward=True, fill_type='mean'):
    # forward=True: Translate image in x,y,xy direction
    # forward=False: Translate image back in minus x,y,xy direction
    # considering repeated maxpooling in VGG, it can repeat translation (1,3,5,7,...)
    samples = []
    if forward: # add 3*num_iter translated samples to the original sample
        samples.append(sample)
        for n in range(num_iter):
            samples.append(TTF.affine(sample, 0, (2*n+1,0), 1, 0))
            samples.append(TTF.affine(sample, 0, (0,2*n+1), 1, 0))
            samples.append(TTF.affine(sample, 0, (2*n+1,2*n+1), 1, 0))
        sample = torch.stack(samples, dim=0)
    else:       # translate back 3*num_iter translated samples and average them to get the original sample
        samples.append(sample[0])
        for n in range(num_iter):
            samples.append(TTF.affine(sample[(3*n+1)], 0, (-(2*n+1),0), 1, 0))
            samples.append(TTF.affine(sample[(3*n+2)], 0, (0,-(2*n+1)), 1, 0))
            samples.append(TTF.affine(sample[(3*n+3)], 0, (-(2*n+1),-(2*n+1)), 1, 0))
        if fill_type == 'mean':
            sample = torch.stack(samples, dim=0).mean(dim=0, keepdim=True)
        elif fill_type == 'median':
            sample = torch.stack(samples, dim=0).median(dim=0, keepdim=True).values
        elif fill_type == 'max':
            sample = torch.stack(samples, dim=0).max(dim=0, keepdim=True).values
    return sample


def calculate_oversampled_fid(tensor1, tensor2, batch_size, device, dims, patch_size=64, min_sample=20000, random_crop=True, stride=None, translation_iter=0, translation_fill_type='median'):
    """Calculates the FID
    """
    torch.cuda.empty_cache()
    # batch_size = batch_size // (3*translation_iter+1)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1).to(torch.float32).to(device)
    else:
        tensor1 = tensor1.to(torch.float32).to(device)
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.from_numpy(tensor2).to(torch.float32).to(device)
    else:
        tensor2 = tensor2.to(torch.tensor2).to(device)
    n_dim = tensor1.dim()
    img_shape = tensor1.size()
    if n_dim == 4:
        if img_shape[1] == 1:   # assuming (B,1,W,H)
            tensor1 = tensor1.repeat(1,3,1,1)
            tensor2 = tensor2.repeat(1,3,1,1)
        elif img_shape[1] == 4:   # assuming (B,4,W,H)
            print('Image shape: {}, assuming the THIRD channel as MAGNITUDE.'.format(img_shape))
            tensor1 = tensor1[:,2:3].repeat(1,3,1,1)
            tensor2 = tensor2[:,2:3].repeat(1,3,1,1)
        else:   # assuming (B,C,W,H)
            print('Image shape: {}, assuming the FIRST channel as MAGNITUDE.'.format(img_shape))
            tensor1 = tensor1[:,0:1].repeat(1,3,1,1)
            tensor2 = tensor2[:,0:1].repeat(1,3,1,1)
    elif n_dim == 3:    # assuming (B,W,H)
        tensor1 = torch.unsqueeze(tensor1, dim=1).repeat(1,3,1,1)
        tensor2 = torch.unsqueeze(tensor2, dim=1).repeat(1,3,1,1)
    else:
        raise NotImplementedError('Implementation is needed for the n_dim={} of input tensors. Or choose only magnitude channel for the input.'.format(n_dim))
        
    if translation_iter > 0:
        tensor1s = _affine(tensor1, num_iter=translation_iter, forward=True, fill_type=translation_fill_type)
        tensor2s = _affine(tensor2, num_iter=translation_iter, forward=True, fill_type=translation_fill_type)
        pred_arr1s = []
        pred_arr2s = []
    
    if translation_iter > 0:
        for i in range(len(tensor1s)):
            torch.cuda.empty_cache()
            tensor1 = tensor1s[i]
            tensor2 = tensor2s[i]
            if random_crop:
                set1 = []
                set2 = []
                for _ in range(min_sample//img_shape[0]+1):
                    i, j, h, w = TF.RandomCrop.get_params(tensor1, output_size=(patch_size, patch_size))
                    tensor1 = TTF.crop(tensor1, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                    tensor2 = TTF.crop(tensor2, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                    set1.append(tensor1)
                    set2.append(tensor2)
                set1 = torch.cat(set1, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
                set2 = torch.cat(set2, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
            else:
                if stride is not None:
                    stride = stride # int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
                else:
                    stride = int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
                set1 = tensor1.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
                set2 = tensor2.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
            img_shape = set1.size()
            print(img_shape)
            if img_shape[0] % batch_size:
                print(('Warning: Image slice # should be divisable by batch size. '
                    'Setting Image slice # to n*batch_size'))
                set1 = set1[:img_shape[0]-(img_shape[0]%batch_size), ...]
                set2 = set2[:img_shape[0]-(img_shape[0]%batch_size), ...]
                
            img_shape = set1.size()
            set1 = torch.reshape(set1, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
            set2 = torch.reshape(set2, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
            pred_arr1 = np.empty((img_shape[0], dims))
            pred_arr2 = np.empty((img_shape[0], dims))
            start_idx = 0
            for batch in tqdm(zip(set1, set2), total=len(set1)):
                batch1 = batch[0].to(device)
                batch2 = batch[1].to(device)
                    
                with torch.no_grad():
                    pred1 = model(batch1)[0]
                    pred2 = model(batch2)[0]
                    
                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred1.size(2) != 1 or pred1.size(3) != 1:
                    pred1 = adaptive_avg_pool2d(pred1, output_size=(1, 1))
                    pred2 = adaptive_avg_pool2d(pred2, output_size=(1, 1))
                pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy()
                pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
                pred_arr1[start_idx:start_idx + pred1.shape[0]] = pred1
                pred_arr2[start_idx:start_idx + pred2.shape[0]] = pred2
                start_idx = start_idx + pred1.shape[0]
            pred_arr1s.append(pred_arr1)
            pred_arr2s.append(pred_arr2)
            del pred_arr1
            del pred_arr2
            del set1
            del set2
            del tensor1
            del tensor2
            torch.cuda.empty_cache()
        if translation_fill_type == 'mean':
            pred_arr1 = np.mean(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.mean(np.stack(pred_arr2s, axis=0), axis=0)
        elif translation_fill_type == 'median':
            pred_arr1 = np.median(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.median(np.stack(pred_arr2s, axis=0), axis=0)
        elif translation_fill_type == 'max':
            pred_arr1 = np.max(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.max(np.stack(pred_arr2s, axis=0), axis=0)
        del tensor1s
        del tensor2s
    else:
        if random_crop:
            set1 = []
            set2 = []
            for _ in range(min_sample//img_shape[0]+1):
                i, j, h, w = TF.RandomCrop.get_params(tensor1, output_size=(patch_size, patch_size))
                tensor1 = TTF.crop(tensor1, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                tensor2 = TTF.crop(tensor2, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                set1.append(tensor1)
                set2.append(tensor2)
            set1 = torch.cat(set1, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
            set2 = torch.cat(set2, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
        else:
            if stride is not None:
                stride = stride # int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
            else:
                stride = int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
            set1 = tensor1.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
            set2 = tensor2.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
        
        img_shape = set1.size()
        print(img_shape)
        if img_shape[0] % batch_size:
            print(('Warning: Image slice # should be divisable by batch size. '
                'Setting Image slice # to n*batch_size'))
            set1 = set1[:img_shape[0]-(img_shape[0]%batch_size), ...]
            set2 = set2[:img_shape[0]-(img_shape[0]%batch_size), ...]
            
        img_shape = set1.size()
        set1 = torch.reshape(set1, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
        set2 = torch.reshape(set2, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
        pred_arr1 = np.empty((img_shape[0], dims))
        pred_arr2 = np.empty((img_shape[0], dims))
        start_idx = 0
        for batch in tqdm(zip(set1, set2), total=len(set1)):
            batch1 = batch[0].to(device)
            batch2 = batch[1].to(device)
                
            with torch.no_grad():
                pred1 = model(batch1)[0]
                pred2 = model(batch2)[0]
            
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred1.size(2) != 1 or pred1.size(3) != 1:
                pred1 = adaptive_avg_pool2d(pred1, output_size=(1, 1))
                pred2 = adaptive_avg_pool2d(pred2, output_size=(1, 1))
            pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy()
            pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr1[start_idx:start_idx + pred1.shape[0]] = pred1
            pred_arr2[start_idx:start_idx + pred2.shape[0]] = pred2
            start_idx = start_idx + pred1.shape[0]
    del model
    torch.cuda.empty_cache()
    m1 = np.mean(pred_arr1, axis=0)
    s1 = np.cov(pred_arr1, rowvar=False)
    m2 = np.mean(pred_arr2, axis=0)
    s2 = np.cov(pred_arr2, rowvar=False)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_oversampled_med_fid(tensor1, tensor2, batch_size, device, dims, patch_size=64, min_sample=20000, random_crop=True, stride=None, translation_iter=0, translation_fill_type='median'):
    """Calculates the FID using brain MRI pre-trained EfficientNet-B0
    """
    torch.cuda.empty_cache()

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(in_features=dims, out_features=4)
    checkpoint = torch.load('MRI_EfficientNetB0/outputs/model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1).to(torch.float32).to(device)
    else:
        tensor1 = tensor1.to(torch.float32).to(device)
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.from_numpy(tensor2).to(torch.float32).to(device)
    else:
        tensor2 = tensor2.to(torch.tensor2).to(device)
    n_dim = tensor1.dim()
    img_shape = tensor1.size()
    if n_dim == 4:
        if img_shape[1] == 1:   # assuming (B,1,W,H)
            tensor1 = tensor1.repeat(1,3,1,1)
            tensor2 = tensor2.repeat(1,3,1,1)
        elif img_shape[1] == 4:   # assuming (B,4,W,H)
            print('Image shape: {}, assuming the THIRD channel as MAGNITUDE.'.format(img_shape))
            tensor1 = tensor1[:,2:3].repeat(1,3,1,1)
            tensor2 = tensor2[:,2:3].repeat(1,3,1,1)
        else:   # assuming (B,C,W,H)
            print('Image shape: {}, assuming the FIRST channel as MAGNITUDE.'.format(img_shape))
            tensor1 = tensor1[:,0:1].repeat(1,3,1,1)
            tensor2 = tensor2[:,0:1].repeat(1,3,1,1)
    elif n_dim == 3:    # assuming (B,W,H)
        tensor1 = torch.unsqueeze(tensor1, dim=1).repeat(1,3,1,1)
        tensor2 = torch.unsqueeze(tensor2, dim=1).repeat(1,3,1,1)
    else:
        raise NotImplementedError('Implementation is needed for the n_dim={} of input tensors. Or choose only magnitude channel for the input.'.format(n_dim))
        
    if translation_iter > 0:
        tensor1s = _affine(tensor1, num_iter=translation_iter, forward=True, fill_type=translation_fill_type)
        tensor2s = _affine(tensor2, num_iter=translation_iter, forward=True, fill_type=translation_fill_type)
        pred_arr1s = []
        pred_arr2s = []
    
    if translation_iter > 0:
        for i in range(len(tensor1s)):
            torch.cuda.empty_cache()
            tensor1 = tensor1s[i]
            tensor2 = tensor2s[i]
            if random_crop:
                set1 = []
                set2 = []
                for _ in range(min_sample//img_shape[0]+1):
                    i, j, h, w = TF.RandomCrop.get_params(tensor1, output_size=(patch_size, patch_size))
                    tensor1 = TTF.crop(tensor1, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                    tensor2 = TTF.crop(tensor2, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                    set1.append(tensor1)
                    set2.append(tensor2)
                set1 = torch.cat(set1, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
                set2 = torch.cat(set2, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
            else:
                if stride is not None:
                    stride = stride # int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
                else:
                    stride = int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
                set1 = tensor1.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
                set2 = tensor2.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
            img_shape = set1.size()
            print(img_shape)
            if img_shape[0] % batch_size:
                print(('Warning: Image slice # should be divisable by batch size. '
                    'Setting Image slice # to n*batch_size'))
                set1 = set1[:img_shape[0]-(img_shape[0]%batch_size), ...]
                set2 = set2[:img_shape[0]-(img_shape[0]%batch_size), ...]
                
            img_shape = set1.size()
            set1 = torch.reshape(set1, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
            set2 = torch.reshape(set2, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
            pred_arr1 = np.empty((img_shape[0], dims))
            pred_arr2 = np.empty((img_shape[0], dims))
            start_idx = 0
            for batch in tqdm(zip(set1, set2), total=len(set1)):
                batch1 = batch[0].to(device)
                batch2 = batch[1].to(device)
                    
                with torch.no_grad():
                    pred1 = model(batch1)
                    pred2 = model(batch2)
                    
                pred1 = pred1.cpu().numpy()
                pred2 = pred2.cpu().numpy()
                pred_arr1[start_idx:start_idx + pred1.shape[0]] = pred1
                pred_arr2[start_idx:start_idx + pred2.shape[0]] = pred2
                start_idx = start_idx + pred1.shape[0]
            pred_arr1s.append(pred_arr1)
            pred_arr2s.append(pred_arr2)
            del pred_arr1
            del pred_arr2
            del set1
            del set2
            del tensor1
            del tensor2
            torch.cuda.empty_cache()
        if translation_fill_type == 'mean':
            pred_arr1 = np.mean(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.mean(np.stack(pred_arr2s, axis=0), axis=0)
        elif translation_fill_type == 'median':
            pred_arr1 = np.median(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.median(np.stack(pred_arr2s, axis=0), axis=0)
        elif translation_fill_type == 'max':
            pred_arr1 = np.max(np.stack(pred_arr1s, axis=0), axis=0)
            pred_arr2 = np.max(np.stack(pred_arr2s, axis=0), axis=0)
        del tensor1s
        del tensor2s
    else:
        if random_crop:
            set1 = []
            set2 = []
            for _ in range(min_sample//img_shape[0]+1):
                i, j, h, w = TF.RandomCrop.get_params(tensor1, output_size=(patch_size, patch_size))
                tensor1 = TTF.crop(tensor1, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                tensor2 = TTF.crop(tensor2, i, j, h, w) # shape: (B, 3, patch_size, patch_size)
                set1.append(tensor1)
                set2.append(tensor2)
            set1 = torch.cat(set1, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
            set2 = torch.cat(set2, dim=0)   # shape: ((min_sample//img_shape[0]+1)*B, 3, patch_size, patch_size)
        else:
            # int(math.sqrt(min_sample//img_shape[0]+1)) + 1
            if stride is not None:
                stride = stride # int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
            else:
                stride = int((img_shape[-2] - patch_size + 1) // int(math.sqrt(min_sample//img_shape[0]+1)))
            set1 = tensor1.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
            set2 = tensor2.unfold(2,patch_size,stride).unfold(3,patch_size,stride).permute(0,2,3,1,4,5).reshape(-1,3,patch_size,patch_size)
        
        img_shape = set1.size()
        print(img_shape)
        if img_shape[0] % batch_size:
            print(('Warning: Image slice # should be divisable by batch size. '
                'Setting Image slice # to n*batch_size'))
            set1 = set1[:img_shape[0]-(img_shape[0]%batch_size), ...]
            set2 = set2[:img_shape[0]-(img_shape[0]%batch_size), ...]
            
        img_shape = set1.size()
        set1 = torch.reshape(set1, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
        set2 = torch.reshape(set2, (img_shape[0]//batch_size, batch_size, img_shape[1], img_shape[2], img_shape[3]))    # shape: (iter, B, 3, patch_size, patch_size)
        pred_arr1 = np.empty((img_shape[0], dims))
        pred_arr2 = np.empty((img_shape[0], dims))
        start_idx = 0
        for batch in tqdm(zip(set1, set2), total=len(set1)):
            batch1 = batch[0].to(device)
            batch2 = batch[1].to(device)
                
            with torch.no_grad():
                pred1 = model(batch1)
                pred2 = model(batch2)
                
            pred1 = pred1.cpu().numpy()
            pred2 = pred2.cpu().numpy()
            pred_arr1[start_idx:start_idx + pred1.shape[0]] = pred1
            pred_arr2[start_idx:start_idx + pred2.shape[0]] = pred2
            start_idx = start_idx + pred1.shape[0]
    del model
    torch.cuda.empty_cache()
    m1 = np.mean(pred_arr1, axis=0)
    s1 = np.cov(pred_arr1, rowvar=False)
    m2 = np.mean(pred_arr2, axis=0)
    s2 = np.cov(pred_arr2, rowvar=False)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

