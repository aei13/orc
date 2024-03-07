import torch
import torch.nn as nn
import numbers

from timm.models.registry import register_model
from einops import rearrange


##########################################################################
## Activation Function (Swish)
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, act_type):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        
        if act_type == 'swish':
            self.act = Swish()
        elif act_type == 'mish':
            self.act = nn.Mish()
        else:
            self.act = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
    
class Attention_FQ_IGateQKV(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.gqkv = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.act = nn.GELU()
        self.fq = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        g = self.act(self.gqkv(x))
        
        q = torch.view_as_complex(rearrange(x, 'b (c ri) h w -> b c h w ri', ri=2).contiguous().to(torch.float32))    # fft needs .contiguous()
        q = torch.fft.fftn(q, dim=(2, 3), norm='forward')
        q = torch.fft.fftshift(q, dim=(2, 3))
        q = rearrange(torch.view_as_real(q), 'b c h w ri -> b (c ri) h w', ri=2)
        q = self.fq(q)
        q = torch.view_as_complex(rearrange(q, 'b (c ri) h w -> b c h w ri', ri=2).contiguous().to(torch.float32))    # fft needs .contiguous()
        q = torch.fft.ifftn(q, dim=(2, 3), norm='backward')
        q = rearrange(torch.view_as_real(q), 'b c h w ri -> b (c ri) h w', ri=2)
        
        kv = self.kv(x)
        
        qkv = self.qkv_dwconv(torch.cat([q, kv], 1))
        qkv = g * qkv
        q,k,v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, act_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, act_type)

    def forward(self, x, return_attention=False):
        if return_attention:
            out, vis = self.attn(self.norm1(x))
            return vis
        else:
            out = self.attn(self.norm1(x))
        out = x + out
        out = out + self.ffn(self.norm2(out))
        return out
    
class TransformerBlock_FQ_IGateQKV(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, act_type):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_FQ_IGateQKV(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, act_type)

    def forward(self, x, return_attention=False):
        if return_attention:
            out, vis = self.attn(self.norm1(x))
            return vis
        else:
            out = self.attn(self.norm1(x))
        out = x + out
        out = out + self.ffn(self.norm2(out))
        return out
    
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        # inp_channels=4, 
        # out_channels=2, 
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        **kwargs
    ):
        '''
        Deep model config:
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        
        Wide model config:
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        '''

        super().__init__()
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])

        self.patch_embed = OverlapPatchEmbed(self.inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), self.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2), torch.atan2(inp_img[:,1], inp_img[:,0])], 1)
        elif self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2)], 1)
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'RI':
            # Pass real, imag channel (excluding magnitude, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,:self.inp_channels//2]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (excluding real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,self.inp_channels//2:]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding real, imag, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,2:3]
        elif self.args_dict['input_type'] == 'MP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,0:1]
        elif self.inp_channels == self.out_channels:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        else:
            raise NotImplementedError('Check in/out channel type or implement new cases.')

        return out_dec_level1
    
class Restormer_FQ_IGateQKV(nn.Module):
    def __init__(self, 
        # inp_channels=4, 
        # out_channels=2, 
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        **kwargs
    ):
        '''
        Deep model config:
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        
        Wide model config:
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        '''

        super().__init__()
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])

        self.patch_embed = OverlapPatchEmbed(self.inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), self.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2), torch.atan2(inp_img[:,1], inp_img[:,0])], 1)
        elif self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2)], 1)
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'RI':
            # Pass real, imag channel (excluding magnitude, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,:self.inp_channels//2]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (excluding real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,self.inp_channels//2:]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding real, imag, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,2:3]
        elif self.args_dict['input_type'] == 'MP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,0:1]
        elif self.inp_channels == self.out_channels:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        else:
            raise NotImplementedError('Check in/out channel type or implement new cases.')

        return out_dec_level1
    
class Restormer_FQ_E_IGateQKV(nn.Module):
    def __init__(self, 
        # inp_channels=4, 
        # out_channels=2, 
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        **kwargs
    ):
        '''
        Deep model config:
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        
        Wide model config:
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        '''

        super().__init__()
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])

        self.patch_embed = OverlapPatchEmbed(self.inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), self.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2), torch.atan2(inp_img[:,1], inp_img[:,0])], 1)
        elif self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2)], 1)
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'RI':
            # Pass real, imag channel (excluding magnitude, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,:self.inp_channels//2]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (excluding real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,self.inp_channels//2:]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding real, imag, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,2:3]
        elif self.args_dict['input_type'] == 'MP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,0:1]
        elif self.inp_channels == self.out_channels:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        else:
            raise NotImplementedError('Check in/out channel type or implement new cases.')

        return out_dec_level1
    
class Restormer_FQ_D_IGateQKV(nn.Module):
    def __init__(self, 
        # inp_channels=4, 
        # out_channels=2, 
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        **kwargs
    ):
        '''
        Deep model config:
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        
        Wide model config:
        dim = 80,
        num_blocks = [1,2,2,2], 
        num_refinement_blocks = 3,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        '''

        super().__init__()
        self.args_dict = kwargs['args']
        self.inp_channels = len(self.args_dict['input_type'])
        self.out_channels = len(self.args_dict['output_type'])

        self.patch_embed = OverlapPatchEmbed(self.inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock_FQ_IGateQKV(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, act_type=self.args_dict['act_type']) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), self.out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.output2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2), torch.atan2(inp_img[:,1], inp_img[:,0])], 1)
        elif self.args_dict['input_type'] == 'RI' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (from real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + torch.stack([torch.sqrt(inp_img[:,0]**2 + inp_img[:,1]**2)], 1)
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'RI':
            # Pass real, imag channel (excluding magnitude, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,:self.inp_channels//2]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'MP':
            # Pass magnitude, phase channel (excluding real, imag channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,self.inp_channels//2:]
        elif self.args_dict['input_type'] == 'RIMP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding real, imag, phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,2:3]
        elif self.args_dict['input_type'] == 'MP' and self.args_dict['output_type'] == 'M':
            # Pass magnitude channel (excluding phase channel)
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,0:1]
        elif self.inp_channels == self.out_channels:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        else:
            raise NotImplementedError('Check in/out channel type or implement new cases.')

        return out_dec_level1



@register_model
def restormer2d(pretrained=False, **kwargs):
    '''Wide model config:
    dim = 80, num_blocks = [1,2,2,2], num_refinement_blocks = 3, ffn_expansion_factor = 2,
    '''
    model = Restormer(args=kwargs['args'])
    return model

@register_model
def restormer2d_fq_igateqkv(pretrained=False, **kwargs):
    '''Wide model config:
    dim = 80, num_blocks = [1,2,2,2], num_refinement_blocks = 3, ffn_expansion_factor = 2,
    '''
    model = Restormer_FQ_IGateQKV(args=kwargs['args'])
    return model

@register_model
def restormer2d_fq_e_igateqkv(pretrained=False, **kwargs):
    '''Wide model config:
    dim = 80, num_blocks = [1,2,2,2], num_refinement_blocks = 3, ffn_expansion_factor = 2,
    '''
    model = Restormer_FQ_E_IGateQKV(args=kwargs['args'])
    return model

@register_model
def restormer2d_fq_d_igateqkv(pretrained=False, **kwargs):
    '''Wide model config:
    dim = 80, num_blocks = [1,2,2,2], num_refinement_blocks = 3, ffn_expansion_factor = 2,
    '''
    model = Restormer_FQ_D_IGateQKV(args=kwargs['args'])
    return model