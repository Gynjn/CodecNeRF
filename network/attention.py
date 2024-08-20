import torch
import einops
import numpy as np

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, N_views_xa=1):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x.to(memory_format=torch.channels_last)

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk
    

class attentionblock(torch.nn.Module):
    def __init__(self, in_ch=256, num_heads=1,
                 skip_scale=np.sqrt(0.5), eps=1e-5):
        
        super().__init__()
        self.in_ch = in_ch
        self.num_heads = num_heads
        self.skip_scale = skip_scale

        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))

        self.norm2 = GroupNorm(num_channels=in_ch, eps=eps)
        self.qkv = Conv2d(in_channels=in_ch, out_channels=in_ch*3, kernel=1, **(init_attn))
        self.proj = Conv2d(in_channels=in_ch, out_channels=in_ch, kernel=1, **init_zero)
    
    def forward(self, x):

        h_xy, h_xz, h_yz = x # (b, c, h, w)
        B, C, H, W = h_xy.shape
        in_x = torch.stack((h_xy, h_xz, h_yz), axis=1) # (b, 3, c, h, w)
        in_x = in_x.permute(0, 1, 3, 4, 2) # (b, 3, h, w, c)
        # (b, 3, h, w, c) -> (b, 3*h, w, c) -> (b, c, 3*h, w)
        in_x = in_x.reshape(B, 3 * in_x.shape[2], *in_x.shape[3:]).permute(0, 3, 1, 2)
        q, k, v = self.qkv(self.norm2(in_x)).reshape(in_x.shape[0] * self.num_heads, in_x.shape[1] // self.num_heads, 3, -1).unbind(2)
        w = AttentionOp.apply(q, k)
        a = torch.einsum('nqk,nck->ncq', w, v)
        in_x = self.proj(a.reshape(*in_x.shape)).add_(in_x)
        in_x = in_x * self.skip_scale

        # (b, c, 3*h, w) -> (b, 3*h, w, c)
        in_x = in_x.permute(0, 2, 3, 1)
        # (b, 3*h, w, c) -> (b, 3, h, w, c) -> (b, 3, c, h, w)
        in_x = in_x.reshape(B, 3, H, W, C).permute(0, 1, 4, 2, 3)
        # (b, 3, c, h, w) -> # (b, c, h, w) 

        h_xy_out = in_x[:, 0, :, :, :]
        h_xz_out = in_x[:, 1, :, :, :]
        h_yz_out = in_x[:, 2, :, :, :]

        return h_xy_out, h_xz_out, h_yz_out
