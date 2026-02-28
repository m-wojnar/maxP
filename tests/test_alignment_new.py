"""Tests for maxp_new.alignment — per-layer alignment computation."""

import math

import pytest
import torch

from maxp_new.alignment import compute_alignment, compute_alignments_for_pms


class TestComputeAlignment:
    """Unit tests for compute_alignment()."""

    def test_zero_weight_change(self):
        """When dw = 0, alpha should be 0 (and omega/u stay 0 too)."""
        torch.manual_seed(0)
        z0 = torch.randn(8, 16)
        w0 = torch.randn(16, 16)
        # No change at all
        alpha, omega, u = compute_alignment(z0, w0, z0, w0, fan_in=16)
        assert alpha == 0.0
        assert omega == 0.0
        assert u == 0.0

    def test_zero_activation_change(self):
        """When dz = 0 but dw != 0, alpha is nonzero but omega and u are 0."""
        torch.manual_seed(1)
        z0 = torch.randn(8, 16)
        w0 = torch.randn(16, 16)
        w = w0 + 0.1 * torch.randn(16, 16)
        # z unchanged, w changed
        alpha, omega, u = compute_alignment(z0, w0, z0, w, fan_in=16)
        assert alpha != 0.0  # should be nonzero
        assert omega == 0.0  # no dz
        assert u == 0.0      # no dz

    def test_known_alignment_identity(self):
        """Verify alignment with structured inputs where we can reason about values."""
        torch.manual_seed(42)
        n = 64
        # z0 and w0 are random, z and w have small perturbations
        z0 = torch.randn(16, n)
        w0 = torch.randn(n, n)
        z = z0 + 0.01 * torch.randn(16, n)
        w = w0 + 0.01 * torch.randn(n, n)

        alpha, omega, u = compute_alignment(z0, w0, z, w, fan_in=n)

        # All should be finite floats
        assert math.isfinite(alpha)
        assert math.isfinite(omega)
        assert math.isfinite(u)

    def test_sanitize_inf_nan(self):
        """Extreme inputs (zeros) should not produce inf or nan."""
        # z0 = 0 could cause log(0) issues
        z0 = torch.zeros(4, 8)
        w0 = torch.zeros(8, 8)
        z = torch.randn(4, 8) * 1e-20
        w = torch.randn(8, 8) * 1e-20

        alpha, omega, u = compute_alignment(z0, w0, z, w, fan_in=8)

        assert math.isfinite(alpha)
        assert math.isfinite(omega)
        assert math.isfinite(u)

    def test_spectral_mode(self):
        """compute_alignment works in spectral norm mode."""
        torch.manual_seed(3)
        z0 = torch.randn(8, 16)
        w0 = torch.randn(16, 16)
        z = z0 + 0.1 * torch.randn(8, 16)
        w = w0 + 0.1 * torch.randn(16, 16)

        alpha, omega, u = compute_alignment(
            z0, w0, z, w, fan_in=16, norm_mode="spectral"
        )
        assert math.isfinite(alpha)
        assert math.isfinite(omega)
        assert math.isfinite(u)

    def test_invalid_norm_mode(self):
        """Invalid norm_mode raises ValueError."""
        z0 = torch.randn(4, 8)
        w0 = torch.randn(8, 8)
        with pytest.raises(ValueError, match="norm_mode"):
            compute_alignment(z0, w0, z0, w0, fan_in=8, norm_mode="l1")

    def test_fan_in_1_no_crash(self):
        """fan_in=1 should not crash (log(1)=0 is handled)."""
        z0 = torch.randn(4, 1)
        w0 = torch.randn(1, 1)
        z = z0 + 0.1 * torch.randn(4, 1)
        w = w0 + 0.1 * torch.randn(1, 1)

        alpha, omega, u = compute_alignment(z0, w0, z, w, fan_in=1)
        assert math.isfinite(alpha)
        assert math.isfinite(omega)
        assert math.isfinite(u)


class TestComputeAlignmentsForPMs:
    """Tests for the batch helper."""

    def test_none_snapshots_get_defaults(self):
        """None entries produce default full-alignment values."""
        alpha, omega, u = compute_alignments_for_pms(
            [None, None], fan_ins=[16, 16]
        )
        assert alpha == [1.0, 1.0]
        assert omega == [0.5, 0.5]
        assert u == [1.0, 1.0]

    def test_mixed_none_and_real(self):
        """Mix of None and real snapshots."""
        torch.manual_seed(5)
        z0 = torch.randn(4, 8)
        w0 = torch.randn(8, 8)
        z = z0 + 0.05 * torch.randn(4, 8)
        w = w0 + 0.05 * torch.randn(8, 8)

        snapshots = [
            None,
            ((z0, w0), (z, w)),
        ]
        alpha, omega, u = compute_alignments_for_pms(snapshots, fan_ins=[8, 8])
        # First is default
        assert alpha[0] == 1.0
        assert omega[0] == 0.5
        assert u[0] == 1.0
        # Second is computed
        assert math.isfinite(alpha[1])
        assert math.isfinite(omega[1])
        assert math.isfinite(u[1])
