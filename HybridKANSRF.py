import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable


# ============================================================
# COMMON GAUSSIAN RBF BASIS
# ============================================================

class GaussianRBF(nn.Module):

    def __init__(
        self,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        denominator: Optional[float] = None
    ):
        super().__init__()

        if num_grids < 2:
            raise ValueError(
                "num_grids must be at least 2."
            )

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
            ) / (
                num_grids - 1
            )

        if denominator <= 0:

            raise ValueError(
                "denominator must be positive."
            )

        self.denominator = denominator


    def forward(
        self,
        x: torch.Tensor
    ):

        return torch.exp(
            -(
                (
                    x.unsqueeze(-1)
                    - self.grid
                )
                / self.denominator
            ) ** 2
        )


# ============================================================
# SPLINE-LINEAR MODULE FOR INTENSITY FASTKAN PATH
# ============================================================

class SplineLinear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale: float = 0.1,
        **kwargs
    ):

        self.init_scale = init_scale

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            **kwargs
        )


    def reset_parameters(self):

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=self.init_scale
        )


# ============================================================
# WAVELENGTH-CONDITIONED KAN/RBF SRF GENERATOR
#
# Input:
#     wavelength positions lambda
#
# Output:
#     one smooth SRF with no_of_bands values
# ============================================================

class WavelengthSRFGenerator(nn.Module):

    def __init__(
        self,
        no_of_bands: int,
        wavelength_num_grids: int = 12,
        nonnegative: bool = False
    ):
        super().__init__()

        self.no_of_bands = no_of_bands

        self.wavelength_num_grids = (
            wavelength_num_grids
        )

        self.nonnegative = nonnegative

        # Normalized wavelength or band positions.
        # First band = -1
        # Last band = +1
        wavelength_positions = torch.linspace(
            -1.0,
            1.0,
            no_of_bands
        )

        self.register_buffer(
            "wavelength_positions",
            wavelength_positions
        )

        # Gaussian basis over wavelength positions.
        self.wavelength_rbf = GaussianRBF(
            grid_min=-1.0,
            grid_max=1.0,
            num_grids=wavelength_num_grids
        )

        # Learned coefficients for wavelength RBF bases.
        self.rbf_coefficients = nn.Parameter(
            torch.empty(
                wavelength_num_grids
            )
        )

        # FastKAN-style wavelength base path:
        # coefficient * SiLU(lambda)
        self.base_coefficient = nn.Parameter(
            torch.empty(1)
        )

        # Constant offset of the generated SRF.
        self.srf_offset = nn.Parameter(
            torch.zeros(1)
        )

        self.reset_parameters()


    def reset_parameters(self):

        nn.init.trunc_normal_(
            self.rbf_coefficients,
            mean=0.0,
            std=0.1
        )

        nn.init.normal_(
            self.base_coefficient,
            mean=0.0,
            std=0.1
        )

        nn.init.zeros_(
            self.srf_offset
        )


    def get_raw_srf(self):

        # Shape:
        # no_of_bands x wavelength_num_grids
        wavelength_basis = (
            self.wavelength_rbf(
                self.wavelength_positions
            )
        )

        # RBF component of wavelength function.
        # Shape:
        # no_of_bands
        wavelength_rbf_response = (
            wavelength_basis
            @ self.rbf_coefficients
        )

        # Residual base function over wavelength.
        wavelength_base_response = (
            self.base_coefficient
            * F.silu(
                self.wavelength_positions
            )
        )

        raw_srf = (
            wavelength_rbf_response
            + wavelength_base_response
            + self.srf_offset
        )

        return raw_srf


    def get_srf(self):

        srf = self.get_raw_srf()

        if self.nonnegative:

            # Positive wavelength envelope.
            srf = F.softplus(srf)

            # Keep positive response scale controlled.
            srf = (
                srf
                / (
                    srf.mean()
                    + 1e-8
                )
            )

        else:

            # Signed wavelength envelope.
            # Suitable for comparison with signed CNN filters.
            srf = (
                srf
                / (
                    torch.norm(
                        srf,
                        p=2
                    )
                    + 1e-8
                )
            )

        return srf


# ============================================================
# HYBRID SPECTRAL PRIMITIVE
#
# Wavelength KAN:
#     lambda_b -> w_k(lambda_b)
#
# Intensity FastKAN:
#     x_b -> nonlinear band contribution phi_k,b(x)
#
# Final:
#     P_k(x) = sum_b w_k(lambda_b) phi_k,b(x) + bias
# ============================================================

class HybridSpectralPrimitive(nn.Module):

    def __init__(
        self,
        no_of_bands: int,

        wavelength_num_grids: int = 12,

        intensity_num_grids: int = 8,

        intensity_grid_min: float = -2.0,

        intensity_grid_max: float = 2.0,

        use_base_update: bool = True,

        base_activation: Callable = F.silu,

        spline_weight_init_scale: float = 0.1,

        nonnegative_srf: bool = False
    ):
        super().__init__()

        self.input_dim = no_of_bands
        self.output_dim = 1

        self.use_base_update = use_base_update
        self.base_activation = base_activation

        # ----------------------------------------------------
        # 1. Smooth wavelength-conditioned SRF generator
        # ----------------------------------------------------

        self.srf_generator = (
            WavelengthSRFGenerator(
                no_of_bands=no_of_bands,
                wavelength_num_grids=(
                    wavelength_num_grids
                ),
                nonnegative=nonnegative_srf
            )
        )

        # ----------------------------------------------------
        # 2. Original intensity FastKAN components
        # ----------------------------------------------------

        self.layernorm = nn.LayerNorm(
            no_of_bands
        )

        self.intensity_rbf = GaussianRBF(
            grid_min=intensity_grid_min,
            grid_max=intensity_grid_max,
            num_grids=intensity_num_grids
        )

        self.spline_linear = SplineLinear(
            in_features=(
                no_of_bands
                * intensity_num_grids
            ),
            out_features=1,
            init_scale=(
                spline_weight_init_scale
            )
        )

        if use_base_update:

            self.base_linear = nn.Linear(
                in_features=no_of_bands,
                out_features=1
            )

        else:

            self.base_linear = None


    def get_srf(self):

        return self.srf_generator.get_srf()


    def get_band_contributions(
        self,
        x: torch.Tensor
    ):

        # ----------------------------------------------------
        # Intensity RBF/spline contribution
        # ----------------------------------------------------

        normalized = self.layernorm(x)

        # Shape:
        # batch x bands x intensity_num_grids
        intensity_basis = (
            self.intensity_rbf(
                normalized
            )
        )

        intensity_num_grids = (
            self.intensity_rbf.grid.numel()
        )

        # Shape:
        # output_dim x bands x intensity_num_grids
        spline_weights = (
            self.spline_linear.weight
            .reshape(
                self.output_dim,
                self.input_dim,
                intensity_num_grids
            )
        )

        # Shape:
        # batch x output_dim x bands
        spline_contribution = (
            intensity_basis.unsqueeze(1)
            * spline_weights.unsqueeze(0)
        ).sum(dim=-1)

        complete_contribution = (
            spline_contribution
        )

        # ----------------------------------------------------
        # Intensity base-path contribution
        # ----------------------------------------------------

        if self.use_base_update:

            base_input = self.base_activation(
                x
            )

            # Shape:
            # batch x output_dim x bands
            base_contribution = (
                base_input.unsqueeze(1)
                * self.base_linear.weight.unsqueeze(0)
            )

            complete_contribution = (
                complete_contribution
                + base_contribution
            )

        return complete_contribution


    def forward(
        self,
        x: torch.Tensor
    ):

        # Complete nonlinear FastKAN band contributions.
        # Shape:
        # batch x 1 x 66
        band_contributions = (
            self.get_band_contributions(x)
        )

        # Smooth KAN/RBF-generated wavelength envelope.
        # Shape:
        # 66
        smooth_srf = self.get_srf()

        # Weight complete nonlinear contributions
        # with smooth wavelength SRF.
        weighted_contributions = (
            band_contributions
            * smooth_srf.view(
                1,
                1,
                -1
            )
        )

        # Sum all 66 weighted nonlinear contributions.
        # Shape:
        # batch x 1
        output = weighted_contributions.sum(
            dim=-1
        )

        # Restore original FastKAN base bias.
        if self.use_base_update:

            output = (
                output
                + self.base_linear.bias
            )

        return output


# ============================================================
# COMPLETE THREE-PRIMITIVE HYBRID COLOR MODEL
# ============================================================

class HybridColorKAN(nn.Module):

    def __init__(
        self,
        no_of_bands: int,
        number_of_primitives: int,
        output_classes: int,

        wavelength_num_grids: int = 12,

        intensity_num_grids: int = 8,

        nonnegative_srf: bool = False,

        use_base_update: bool = True,

        base_activation: Callable = F.silu,

        spline_weight_init_scale: float = 0.1
    ):
        super().__init__()

        self.no_of_bands = no_of_bands

        self.number_of_primitives = (
            number_of_primitives
        )

        self.primitives = nn.ModuleList([

            HybridSpectralPrimitive(
                no_of_bands=no_of_bands,

                wavelength_num_grids=(
                    wavelength_num_grids
                ),

                intensity_num_grids=(
                    intensity_num_grids
                ),

                nonnegative_srf=(
                    nonnegative_srf
                ),

                use_base_update=(
                    use_base_update
                ),

                base_activation=(
                    base_activation
                ),

                spline_weight_init_scale=(
                    spline_weight_init_scale
                )
            )

            for _ in range(
                number_of_primitives
            )
        ])

        # Same Dense classifier used in your previous models.
        self.classifier_dense = nn.Linear(
            in_features=number_of_primitives,
            out_features=output_classes
        )


    def get_primitive_outputs(
        self,
        x: torch.Tensor
    ):

        primitive_outputs = [

            primitive(x)

            for primitive in self.primitives
        ]

        merged = torch.cat(
            primitive_outputs,
            dim=1
        )

        return merged


    def forward(
        self,
        x: torch.Tensor
    ):

        merged = self.get_primitive_outputs(
            x
        )

        logits = self.classifier_dense(
            merged
        )

        return logits