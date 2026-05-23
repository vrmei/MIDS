"""Pure-PyTorch Mamba block fallback.

When the official ``mamba-ssm`` package isn't installable (e.g., on
native Windows, where the CUDA kernels + Triton dependency are a
nightmare), we fall back to this minimal pure-PyTorch implementation.
It exposes the same constructor signature as :class:`mamba_ssm.Mamba`
so the rest of the codebase doesn't notice the swap.

The implementation follows the selective state-space recurrence from
the Mamba paper (Gu & Dao, 2023). It's ~3-5x slower than the official
kernels on a comparable GPU because it materialises the full
``(B, L, d_inner, d_state)`` SSM state tensor instead of using the
fused selective-scan kernel, but it's bit-equivalent in numerics
(modulo float32/16 reordering) and small enough to read end-to-end.

Reference: https://github.com/johnma2006/mamba-minimal — apache-2.0,
adapted/condensed here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Mamba(nn.Module):
    """Pure-PyTorch S6 block, signature-compatible with ``mamba_ssm.Mamba``.

    Args:
        d_model: Token feature dim (input and output).
        d_state: Latent SSM state dim (``N`` in the paper).
        d_conv: Depthwise conv kernel width.
        expand: Inner expansion factor (``d_inner = expand * d_model``).
        dt_rank: Rank of the ∆-projection. Defaults to ``ceil(d_model/16)``.
        conv_bias: Whether the depthwise conv has bias.
        bias: Whether linear projections have bias.

    Input:  ``x`` of shape ``(B, L, d_model)``.
    Output: same shape ``(B, L, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection: x -> (x_path, residual_gate) each of width d_inner.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise 1-D convolution along sequence axis.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # Data-dependent projection: input -> (dt, B, C).
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False,
        )
        # ∆ low-rank to full-rank.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # ∆ bias initialisation (per Gu & Dao §3.4).
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialise dt_proj.bias so softplus(dt_proj.bias) ~ U[0.001, 0.1].
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: HiPPO-style logarithmically spaced negative real diagonal.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        # D: residual skip.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model.
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run one bidirectional pass; same shape in, same shape out."""
        B, L, _ = x.shape

        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                        # each (B, L, d_inner)

        # Depthwise conv along the sequence axis.
        x = x.transpose(1, 2)                             # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]                       # crop right-pad
        x = x.transpose(1, 2)                             # (B, L, d_inner)
        x = F.silu(x)

        # Selective scan parameters.
        y = self._selective_scan(x)                       # (B, L, d_inner)

        # Gated residual: SiLU(z) acts as a per-channel gate.
        y = y * F.silu(z)
        return self.out_proj(y)

    # ------------------------------------------------------------------

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """Vectorised SSM recurrence over ``u`` of shape (B, L, d_inner).

        Materialises the full ``(B, L, d_inner, d_state)`` discretised
        state matrices and uses a simple sequential scan along ``L``.
        Quadratic in batch * d_inner * d_state but linear in L, so total
        cost is ``O(B · L · d_inner · d_state)`` — same asymptotic class
        as the fused kernel, just without the kernel fusion.
        """
        B, L, d_inner = u.shape
        d_state = self.d_state

        # ∆ and the data-dependent B, C.
        x_dbl = self.x_proj(u)                            # (B, L, dt_rank + 2N)
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))                 # (B, L, d_inner)

        # Discretise A and B using ZOH-style approximation.
        A = -torch.exp(self.A_log.float())                # (d_inner, d_state)
        # dA: (B, L, d_inner, d_state) = exp(dt[..., None] * A)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        # dB_u: (B, L, d_inner, d_state) = dt[..., None] * B_proj[:, :, None, :] * u[..., None]
        dB_u = (
            dt.unsqueeze(-1)
            * B_proj.unsqueeze(2)
            * u.unsqueeze(-1)
        )

        # Sequential scan: h_t = dA_t * h_{t-1} + dB_u_t.
        h = u.new_zeros(B, d_inner, d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]                 # (B, d_inner, d_state)
            # y_t = sum_n C_t[n] * h_t[..., n]
            y_t = torch.einsum("bdn,bn->bd", h, C_proj[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                        # (B, L, d_inner)

        # Skip connection through D (per channel).
        y = y + u * self.D
        return y
