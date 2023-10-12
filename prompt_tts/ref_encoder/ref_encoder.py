import torch.nn as nn
from beartype import beartype
from beartype.typing import Tuple
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from prompt_tts.ref_encoder.utils import RMSNorm, Attention, FeedForward

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        use_flash = False,
        dropout = 0.,
        ff_mult = 4,
        final_norm = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RMSNorm(dim),
                Attention(
                    dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = dropout,
                    use_flash = use_flash
                ),
                RMSNorm(dim),
                FeedForward(
                    dim,
                    mult = ff_mult
                )
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask = None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask = mask) + x
            x = ff(ff_norm(x)) + x

        return self.norm(x)

class SpeechPromptEncoder(nn.Module):

    @beartype
    def __init__(
        self,
        dim_codebook,
        dims: Tuple[int] = (256, 2048, 2048, 2048, 2048, 512, 512, 512),
        *,
        depth = 6,
        heads = 8,
        dim_head = 64,
        dropout = 0.2,
        kernel_size = 9,
        padding = 4,
        use_flash_attn = True

    ):
        super().__init__()

        dims = [dim_codebook, *dims]

        self.dim, self.dim_out = dims[0], dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        modules = []
        for dim_in, dim_out in dim_pairs:
            modules.extend([
                nn.Conv1d(dim_in, dim_out, kernel_size, padding = padding),
                nn.SiLU()
            ])

        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            *modules,
            Rearrange('b c n -> b n c')
        )

        self.transformer = Transformer(
            dim = dims[-1],
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,
            use_flash = use_flash_attn
        )

    def forward(self, x):
        assert x.shape[-1] == self.dim

        x = self.conv(x)
        x = self.transformer(x)
        return x
