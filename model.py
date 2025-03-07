
from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from timm.models.layers import DropPath

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
        
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_out = torch.nn.Dropout1d(0.2)
    
    def forward(self, x):
        input = x
        #x = self.drop_out(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)

        x = input + self.drop_path(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, dims:list[int], depths:list[int]):
        super().__init__()
        self.conv1 = Conv1d(dims[0], dims[0], kernel_size=3, padding=1)
        self.conv2 = Conv1d(dims[0], dims[0], kernel_size=3, stride=2, padding=1)

        blocks = []
        for idx in range(len(dims)):
            for depth in depths:
                blocks.append(Block(dims[idx]))
            
            if idx < len(dims) - 1:
                blocks.append(Conv1d(dims[idx], dims[idx + 1], kernel_size=3, stride=2, padding=1))
                    
        self.blocks: Iterable[Block] = nn.ModuleList(
            blocks
        )
        self.ln_post = LayerNorm(dims[-1])

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        for block in self.blocks:
            x = block(x)

        return x
        x = x.mean([-1])
        return self.ln_post(x)
    
class Whisper(nn.Module):
    def __init__(self, emb_dim=128, dims=[80, 160, 320, 640], depths=[2, 2, 8, 2]):
        super().__init__()
        self.output_dim = emb_dim
        self.encoder = AudioEncoder(dims, depths)
        self.head = nn.Linear(dims[-1], emb_dim)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.head(self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device