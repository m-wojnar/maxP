"""Tests for maxp.alignment — per-layer alignment computation."""

import math

import torch

from maxp.alignment import compute_alignment


class TestComputeAlignment:
    """Unit tests for compute_alignment()."""

    def test_zero_weight_change(self):
        """When dW = 0, align_z0_dW should be 0 (and align_dZ_w0/align_dZ_dW too)."""
        torch.manual_seed(0)
        z0 = torch.randn(8, 16)
        w0 = torch.randn(16, 16)
        # No change at all
        align_z0_dW, align_dZ_w0, align_dZ_dW = compute_alignment(z0, w0, z0, w0, fan_in=16)
        assert align_z0_dW == 0.0
        assert align_dZ_w0 == 0.0
        assert align_dZ_dW == 0.0

    def test_zero_activation_change(self):
        """When dZ = 0 but dW != 0, align_z0_dW is nonzero but align_dZ_w0 and align_dZ_dW are 0."""
        torch.manual_seed(1)
        z0 = torch.randn(8, 16)
        w0 = torch.randn(16, 16)
        w = w0 + 0.1 * torch.randn(16, 16)
        # z unchanged, w changed
        align_z0_dW, align_dZ_w0, align_dZ_dW = compute_alignment(z0, w0, z0, w, fan_in=16)
        assert align_z0_dW != 0.0  # should be nonzero
        assert align_dZ_w0 == 0.0  # no dZ
        assert align_dZ_dW == 0.0  # no dZ

    def test_known_alignment_identity(self):
        """Verify alignment with structured inputs where we can reason about values."""
        torch.manual_seed(42)
        n = 64
        # z0 and w0 are random, z and w have small perturbations
        z0 = torch.randn(16, n)
        w0 = torch.randn(n, n)
        z = z0 + 0.01 * torch.randn(16, n)
        w = w0 + 0.01 * torch.randn(n, n)

        align_z0_dW, align_dZ_w0, align_dZ_dW = compute_alignment(z0, w0, z, w, fan_in=n)

        # All should be finite floats
        assert math.isfinite(align_z0_dW)
        assert math.isfinite(align_dZ_w0)
        assert math.isfinite(align_dZ_dW)

    def test_sanitize_inf_nan(self):
        """Extreme inputs (zeros) should not produce inf or nan."""
        # z0 = 0 could cause log(0) issues
        z0 = torch.zeros(4, 8)
        w0 = torch.zeros(8, 8)
        z = torch.randn(4, 8) * 1e-20
        w = torch.randn(8, 8) * 1e-20

        align_z0_dW, align_dZ_w0, align_dZ_dW = compute_alignment(z0, w0, z, w, fan_in=8)

        assert math.isfinite(align_z0_dW)
        assert math.isfinite(align_dZ_w0)
        assert math.isfinite(align_dZ_dW)

    def test_fan_in_1_no_crash(self):
        """fan_in=1 should not crash (log(1)=0 is handled)."""
        z0 = torch.randn(4, 1)
        w0 = torch.randn(1, 1)
        z = z0 + 0.1 * torch.randn(4, 1)
        w = w0 + 0.1 * torch.randn(1, 1)

        align_z0_dW, align_dZ_w0, align_dZ_dW = compute_alignment(z0, w0, z, w, fan_in=1)
        assert math.isfinite(align_z0_dW)
        assert math.isfinite(align_dZ_w0)
        assert math.isfinite(align_dZ_dW)
