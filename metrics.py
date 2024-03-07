import inspect
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn
import torchvision.models as models

def spatial_average(in_tens, keepdim=True):
    in_tens /= in_tens.max()
    return in_tens.mean([2,3],keepdim=keepdim)

def upsam(in_tens, out_hw=(64, 64)):
    """Upsample input with bilinear interpolation."""
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)(in_tens)

def normalize_tensor(in_feat, eps=1e-10):
    """Normalize tensors."""
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff = models.efficientnet_b0()

    def forward(self, x):
        outputs = []
        for features in self.eff.features:
            x = features(x)
            outputs.append(x)
        return outputs

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=False, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] tune the base/trunk network
            [True] keep base/trunk frozen
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type=='efficientnet_b0'):
            net_type = CustomEfficientNet
            self.chns = [32,16,24,40,80,112,192,320]
        else:
            assert NotImplementedError('Use LPIPS package instead!')
        self.L = len(self.chns)

        self.net = net_type()

        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(
                    os.path.join(inspect.getfile(self.__init__), "..", f"lpips_models/{net}.pth")  # type: ignore[misc]
                )

            if(verbose):
                print('Loading model from: %s'%model_path)
            new_state_dict = OrderedDict()
            checkpoint = torch.load(model_path, map_location='cpu')

            for k, v in checkpoint['model_state_dict'].items():
                name = k.replace("features", "net.eff.features")  # replace the keys
                new_state_dict[name] = v
            
            self.load_state_dict(new_state_dict, strict=False)

        self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            diffs[kk] /= (diffs[kk].max() + 1e-10)
            
        if self.spatial:
            res = [upsam(diffs[kk].sum(dim=1, keepdim=True), out_hw=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True)) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        val /= self.L
        
        if(retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
    
    
class Hook():
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        
    def close(self):
        self.hook.remove()