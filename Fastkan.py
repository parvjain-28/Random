import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable


class SplineLinear(nn.Linear):
   

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale: float = 0.1,
        **kwargs
    ) -> None:

        self.init_scale = init_scale

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            **kwargs
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=self.init_scale
        )

class RadialBasisFunction(nn.Module):
 
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 2,
        denominator: Optional[float] = None
    ) -> None:

        super().__init__()

        if num_grids < 2:
            raise ValueError(
                "num_grids must be at least 2 because the default "
                "denominator uses (num_grids - 1)."
            )

        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids

        grid = torch.linspace(
            grid_min,
            grid_max,
            num_grids
        )
        self.register_buffer(
            "grid",
            grid
        )

        if denominator is None:
            denominator = (
                grid_max - grid_min
            ) / (num_grids - 1)

        if denominator <= 0:
            raise ValueError(
                "denominator must be greater than zero."
            )

        self.denominator = float(denominator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.exp(
            -torch.square(
                (
                    x.unsqueeze(-1) - self.grid
                ) / self.denominator
            )
        )

class Fast_KANLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 2,
        use_base_update: bool = True,
        base_activation: Callable = F.silu,
        spline_weight_init_scale: float = 0.1
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.layernorm = nn.LayerNorm(input_dim)

        self.rbf = RadialBasisFunction(
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids
        )
        self.spline_linear = SplineLinear(
            in_features=input_dim * num_grids,
            out_features=output_dim,
            init_scale=spline_weight_init_scale
        )
        if use_base_update:
            self.base_linear = nn.Linear(
                in_features=input_dim,
                out_features=output_dim
            )
        else:
            self.base_linear = None

    def forward(
        self,
        x: torch.Tensor,
        time_benchmark: bool = False
    ) -> torch.Tensor:

        if not time_benchmark:
            normalized_x = self.layernorm(x)
        else:
            normalized_x = x
        spline_basis = self.rbf(normalized_x)
        spline_basis = spline_basis.flatten(
            start_dim=-2
        )

        output = self.spline_linear(
            spline_basis
        )
        if self.use_base_update:
            base_output = self.base_linear(
                self.base_activation(x)
            )

            output = output + base_output

        return output
class FastKAN(nn.Module):
  

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 2,
        use_base_update: bool = True,
        base_activation: Callable = F.silu,
        spline_weight_init_scale: float = 0.1
    ) -> None:

        super().__init__()

        if len(layers_hidden) < 2:
            raise ValueError(
                "layers_hidden must contain at least an input "
                "dimension and an output dimension."
            )

        self.layers_hidden = list(layers_hidden)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.use_base_update = use_base_update
        self.spline_weight_init_scale = (
            spline_weight_init_scale
        )

        self.layers = nn.ModuleList(
            [
                Fast_KANLinear(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=(
                        spline_weight_init_scale
                    )
                )
                for input_dim, output_dim in zip(
                    layers_hidden[:-1],
                    layers_hidden[1:]
                )
            ]
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)

        return x