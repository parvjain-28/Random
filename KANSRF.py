import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRBF(nn.Module):

    def __init__(
        self,
        grid_min=-1.0,
        grid_max=1.0,
        num_grids=12,
        denominator=None
    ):
        super().__init__()

        if num_grids < 2:
            raise ValueError("num_grids must be at least 2.")

        grid = torch.linspace(grid_min,grid_max,num_grids)

        self.register_buffer("grid",grid)

        if denominator is None:

            denominator = (grid_max - grid_min) / (num_grids - 1)

        if denominator <= 0:

            raise ValueError("denominator must be positive.")

        self.denominator = denominator


    def forward(self, wavelength_positions):

        return torch.exp(-((wavelength_positions.unsqueeze(-1) - self.grid)/ self.denominator) ** 2)


class KANSpectralPrimitive(nn.Module):

    def __init__( self ,no_of_bands,num_grids=12,nonnegative=False):
        super().__init__()

        self.no_of_bands = no_of_bands
        self.num_grids = num_grids
        self.nonnegative = nonnegative

        # Normalized spectral positions.
        # Band 1 corresponds to -1.
        # Last band corresponds to +1.
        wavelength_positions = torch.linspace(-1.0,1.0,no_of_bands)

        self.register_buffer("wavelength_positions",wavelength_positions)

        # Gaussian RBF basis over wavelength.
        self.rbf = GaussianRBF(grid_min=-1.0,grid_max=1.0,num_grids=num_grids)

        # One coefficient for each RBF basis.
        self.rbf_coefficients = nn.Parameter(
            torch.empty(num_grids)
        )

        # FastKAN-style residual base path:
        # coefficient × SiLU(wavelength).
        self.base_coefficient = nn.Parameter(
            torch.empty(1)
        )

        # Constant offset in the generated SRF.
        self.srf_offset = nn.Parameter(
            torch.zeros(1)
        )

        # Primitive output bias.
        self.output_bias = nn.Parameter(
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

        nn.init.zeros_(
            self.output_bias
        )


    def get_raw_srf(self):

        # Shape:
        # no_of_bands × num_grids
        rbf_basis = self.rbf(
            self.wavelength_positions
        )

        # RBF/spline-style contribution.
        # Shape: no_of_bands
        rbf_response = (
            rbf_basis
            @ self.rbf_coefficients
        )

        # Base-path contribution.
        # Shape: no_of_bands
        base_response = (
            self.base_coefficient
            * F.silu(
                self.wavelength_positions
            )
        )

        raw_srf = (
            rbf_response
            + base_response
            + self.srf_offset
        )

        return raw_srf


    def get_srf(self):

        srf = self.get_raw_srf()

        if self.nonnegative:

            # Physical/CIE-like positive sensitivity.
            srf = F.softplus(srf)

            # Mean-normalize positive response.
            srf = srf / (
                srf.mean()
                + 1e-8
            )

        else:

            # Signed SRF for CNN-like comparison.
            # L2 normalization prevents uncontrolled scale.
            srf = srf / (
                torch.norm(srf, p=2)
                + 1e-8
            )

        return srf


    def forward(self, x):

        # x shape:
        # batch_size × no_of_bands

        srf = self.get_srf()

        # Actual primitive calculation:
        # weighted sum across 66 spectral bands.
        output = (
            x
            * srf.unsqueeze(0)
        ).sum(
            dim=1,
            keepdim=True
        )

        output = (
            output
            + self.output_bias
        )

        return output


class ColorKANSRF(nn.Module):

    def __init__(
        self,
        no_of_bands,
        number_of_primitives,
        output_classes,
        num_grids=12,
        nonnegative=False
    ):
        super().__init__()

        self.no_of_bands = no_of_bands
        self.number_of_primitives = (
            number_of_primitives
        )

        self.primitives = nn.ModuleList([

            KANSpectralPrimitive(
                no_of_bands=no_of_bands,
                num_grids=num_grids,
                nonnegative=nonnegative
            )

            for _ in range(
                number_of_primitives
            )
        ])

        # Same Dense classifier as your current model.
        self.classifier_dense = nn.Linear(
            in_features=number_of_primitives,
            out_features=output_classes
        )


    def get_primitive_outputs(self, x):

        primitive_outputs = [

            primitive(x)

            for primitive in self.primitives
        ]

        merged = torch.cat(
            primitive_outputs,
            dim=1
        )

        return merged


    def forward(self, x):

        merged = self.get_primitive_outputs(
            x
        )

        logits = self.classifier_dense(
            merged
        )

        return logits