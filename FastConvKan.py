import torch
import torch.nn as nn
import torch.nn.functional as F

from Fastkan import Fast_KANLinear


class FastConvKAN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        num_grids=8,
        spline_weight_init_scale=0.1,
        base_activation=F.silu,
        grid_min=-2.0,
        grid_max=2.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.linear = Fast_KANLinear(
            input_dim=(in_channels* kernel_size* kernel_size),
            output_dim=out_channels,
            num_grids=num_grids,
            spline_weight_init_scale=(
                spline_weight_init_scale
            ),
            base_activation=base_activation,
            grid_min=grid_min,
            grid_max=grid_max
        )

    def forward(self, x):

        assert x.dim() == 4

        batch_size, in_channels, height, width = x.shape

        assert in_channels == self.in_channels
        patches = self.unfold(x)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(-1, self.in_channels* self.kernel_size * self.kernel_size
        )
        out = self.linear(patches)
        out = out.view( batch_size,-1,self.out_channels)
        out = out.transpose(1, 2)

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride +1 

        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride +1 

        out = out.reshape(batch_size,self.out_channels,out_height,out_width)

        return out

    