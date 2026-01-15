"""Tests for ABC weight initialization utilities."""

import math

import pytest
import torch
import torch.nn as nn

from maxp import ScaledLinear, get_abc_parametrization, initialize_abc_weights


class SimpleMLP(nn.Module):
    """Simple 3-layer MLP for testing."""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


class NestedMLP(nn.Module):
    """MLP with nested structure for testing path-based replacement."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class TestScaledLinear:
    """Tests for ScaledLinear class."""
    
    def test_forward_applies_scale(self):
        linear = nn.Linear(4, 8, bias=False)
        nn.init.ones_(linear.weight)
        
        scaled = ScaledLinear(linear, scale=0.5)
        x = torch.ones(2, 4)
        
        # Linear output should be 4.0 (sum of 4 ones), scaled by 0.5 = 2.0
        output = scaled(x)
        assert output.shape == (2, 8)
        assert torch.allclose(output, torch.full((2, 8), 2.0))
    
    def test_weight_property(self):
        linear = nn.Linear(4, 8)
        scaled = ScaledLinear(linear, scale=0.5)
        
        assert scaled.weight is linear.weight
        assert scaled.bias is linear.bias
    
    def test_in_out_features_properties(self):
        linear = nn.Linear(4, 8)
        scaled = ScaledLinear(linear, scale=0.5)
        
        assert scaled.in_features == 4
        assert scaled.out_features == 8
    
    def test_repr(self):
        linear = nn.Linear(4, 8, bias=True)
        scaled = ScaledLinear(linear, scale=0.125)
        
        repr_str = repr(scaled)
        assert "in_features=4" in repr_str
        assert "out_features=8" in repr_str
        assert "bias=True" in repr_str
        assert "scale=0.125" in repr_str


class TestInitializeABCWeights:
    """Tests for initialize_abc_weights function."""
    
    def test_initialize_with_named_parametrization(self):
        torch.manual_seed(42)
        model = SimpleMLP()
        
        # Get expected al/bl for comparison
        abc = get_abc_parametrization(n_layers=3, parametrization="mup")
        
        initialize_abc_weights(model, parametrization="mup")
        
        # Check that layers with non-zero al are wrapped with ScaledLinear
        # muP has al = [-0.5, 0.0, 0.5] for 3 layers
        # So l1 (a=-0.5) and l3 (a=0.5) should be wrapped, l2 (a=0.0) should not
        assert isinstance(model.l1, ScaledLinear)
        assert isinstance(model.l2, nn.Linear) and not isinstance(model.l2, ScaledLinear)
        assert isinstance(model.l3, ScaledLinear)
    
    def test_initialize_with_custom_values(self):
        torch.manual_seed(42)
        model = SimpleMLP()
        
        al = [0.0, 0.5, 0.0]  # Only middle layer gets multiplier
        bl = [0.5, 0.5, 0.5]
        
        initialize_abc_weights(model, al=al, bl=bl)
        
        # Only l2 should be wrapped (a=0.5)
        assert isinstance(model.l1, nn.Linear) and not isinstance(model.l1, ScaledLinear)
        assert isinstance(model.l2, ScaledLinear)
        assert isinstance(model.l3, nn.Linear) and not isinstance(model.l3, ScaledLinear)
    
    def test_initialize_without_multipliers(self):
        torch.manual_seed(42)
        model = SimpleMLP()
        
        initialize_abc_weights(model, parametrization="mup", apply_multipliers=False)
        
        # No layers should be wrapped
        assert isinstance(model.l1, nn.Linear) and not isinstance(model.l1, ScaledLinear)
        assert isinstance(model.l2, nn.Linear) and not isinstance(model.l2, ScaledLinear)
        assert isinstance(model.l3, nn.Linear) and not isinstance(model.l3, ScaledLinear)
    
    def test_weight_initialization_variance(self):
        """Test that weights are initialized with correct variance."""
        torch.manual_seed(42)
        model = SimpleMLP()
        
        # Use bl = [0.5, 0.5, 0.5] which means std = sqrt(2) * n^{-0.5}
        bl = [0.5, 0.5, 0.5]
        al = [0.0, 0.0, 0.0]  # No multipliers
        std_prefactor = 2**0.5
        
        initialize_abc_weights(model, al=al, bl=bl, std_prefactor=std_prefactor)
        
        # Check l1: fan_in=64, expected std = sqrt(2) * 64^{-0.5} = sqrt(2)/8
        expected_std_l1 = std_prefactor * (64 ** -0.5)
        actual_std_l1 = model.l1.weight.std().item()
        assert abs(actual_std_l1 - expected_std_l1) < 0.05  # Allow some variance
        
        # Check l2: fan_in=128, expected std = sqrt(2) * 128^{-0.5}
        expected_std_l2 = std_prefactor * (128 ** -0.5)
        actual_std_l2 = model.l2.weight.std().item()
        assert abs(actual_std_l2 - expected_std_l2) < 0.05
    
    def test_nested_model_replacement(self):
        """Test that nested layers are correctly replaced."""
        torch.manual_seed(42)
        model = NestedMLP()
        
        # 3 linear layers total
        al = [0.5, 0.5, 0.5]  # All get multipliers
        bl = [0.5, 0.5, 0.5]
        
        initialize_abc_weights(model, al=al, bl=bl)
        
        # Check nested layers are wrapped
        assert isinstance(model.encoder[0], ScaledLinear)
        assert isinstance(model.decoder[0], ScaledLinear)
        assert isinstance(model.decoder[2], ScaledLinear)
    
    def test_forward_still_works_after_initialization(self):
        """Test that model forward pass works after initialization."""
        torch.manual_seed(42)
        model = SimpleMLP()
        
        initialize_abc_weights(model, parametrization="mup")
        
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
    
    def test_returns_same_model(self):
        model = SimpleMLP()
        result = initialize_abc_weights(model, parametrization="sp")
        assert result is model
    
    def test_error_when_both_parametrization_and_al_provided(self):
        model = SimpleMLP()
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            initialize_abc_weights(model, parametrization="mup", al=[0.0, 0.0, 0.0])
    
    def test_error_when_neither_provided(self):
        model = SimpleMLP()
        
        with pytest.raises(ValueError, match="Must provide either"):
            initialize_abc_weights(model)
    
    def test_error_when_length_mismatch(self):
        model = SimpleMLP()
        
        with pytest.raises(ValueError, match="must match"):
            initialize_abc_weights(model, al=[0.0, 0.0], bl=[0.0, 0.0])  # Only 2, need 3
    
    def test_scale_values_are_correct(self):
        """Test that ScaledLinear has correct scale values."""
        torch.manual_seed(42)
        model = SimpleMLP()
        
        # muP has al = [-0.5, 0.0, 0.5]
        # l1: fan_in=64, scale = 64^{-(-0.5)} = 64^{0.5} = 8
        # l3: fan_in=128, scale = 128^{-0.5} = 1/sqrt(128)
        initialize_abc_weights(model, parametrization="mup")
        
        expected_scale_l1 = 64 ** 0.5  # 8.0
        expected_scale_l3 = 128 ** -0.5  # ~0.0884
        
        assert abs(model.l1.scale - expected_scale_l1) < 1e-6
        assert abs(model.l3.scale - expected_scale_l3) < 1e-6
    
    def test_bias_initialized_to_zero(self):
        """Test that biases are initialized to zero."""
        torch.manual_seed(42)
        model = SimpleMLP()
        
        initialize_abc_weights(model, parametrization="sp", apply_multipliers=False)
        
        assert torch.allclose(model.l1.bias, torch.zeros_like(model.l1.bias))
        assert torch.allclose(model.l2.bias, torch.zeros_like(model.l2.bias))
        assert torch.allclose(model.l3.bias, torch.zeros_like(model.l3.bias))
