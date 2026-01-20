"""Tests for ABC parametrization utilities."""

import pytest
import torch
import torch.nn as nn

from maxp import MaxPScheduler, create_param_groups, get_abc_parametrization
from maxp.utils import SemanticRole, get_semantic_roles, get_managed_layers


class TestGetABCParametrization:
    """Tests for get_abc_parametrization function with semantic roles."""

    def test_mup_adam_full_4_layers(self):
        """Test muP with 4 layers: EMBEDDING, HIDDEN, HIDDEN, READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="mup", optimizer="adam", alignment="full")
        
        assert abc.al == [-0.5, 0.0, 0.0, 0.5]
        assert abc.bl == [0.5, 0.5, 0.5, 0.5]
        assert abc.cl == [0.5, 1.0, 1.0, 0.5]
        assert abc.name == "mup-adam-full"

    def test_mup_sgd_no_alignment(self):
        """Test muP with 3 layers: EMBEDDING, HIDDEN, READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="mup", optimizer="sgd", alignment="no")
        
        assert abc.al == [-0.5, 0.0, 0.5]
        assert abc.bl == [0.5, 0.5, 0.5]
        assert abc.cl == [0.0, -0.5, 0.0]
        assert abc.name == "mup-sgd-no"

    def test_sp_parametrization(self):
        """Test SP with 3 layers: EMBEDDING, HIDDEN, READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="sp", optimizer="adam", alignment="full")
        
        assert abc.al == [0.0, 0.0, 0.0]
        assert abc.bl == [0.0, 0.5, 0.5]
        assert abc.cl == [0.0, 1.0, 1.0]
        assert abc.name == "sp-adam-full"

    def test_ntk_parametrization(self):
        """Test NTK with 4 layers: EMBEDDING, HIDDEN, HIDDEN, READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="ntk", optimizer="adam", alignment="full")
        
        assert abc.al == [0.0, 0.5, 0.5, 0.5]
        assert abc.bl == [0.0, 0.0, 0.0, 0.0]
        assert abc.cl == [0.0, 0.5, 0.5, 0.5]
        assert abc.name == "ntk-adam-full"

    def test_mfp_parametrization(self):
        """Test MFP with 4 layers: EMBEDDING, HIDDEN, HIDDEN, READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="mfp", optimizer="sgd", alignment="full")
        
        assert abc.al == [0.0, 0.5, 0.5, 1.0]
        assert abc.bl == [0.0, 0.0, 0.0, 0.0]
        assert abc.cl == [-1.0, -1.0, -1.0, -1.0]
        assert abc.name == "mfp-sgd-full"

    def test_case_insensitive(self):
        """Test that parametrization options are case insensitive."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]
        abc1 = get_abc_parametrization(semantic_roles=roles, parametrization="MUP", optimizer="ADAM", alignment="FULL")
        abc2 = get_abc_parametrization(semantic_roles=roles, parametrization="mup", optimizer="adam", alignment="full")
        
        assert abc1.al == abc2.al
        assert abc1.bl == abc2.bl
        assert abc1.cl == abc2.cl

    def test_two_layers(self):
        """Test minimal network with just EMBEDDING and READOUT."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.READOUT]
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="mup", optimizer="adam", alignment="full")
        
        # Just embedding + readout
        assert abc.al == [-0.5, 0.5]
        assert abc.bl == [0.5, 0.5]
        assert abc.cl == [0.5, 0.5]

    def test_error_on_empty_roles(self):
        """Test error when semantic_roles is empty."""
        with pytest.raises(ValueError, match="semantic_roles cannot be empty"):
            get_abc_parametrization(semantic_roles=[], parametrization="mup")

    def test_error_on_no_readout(self):
        """Test error when no READOUT role is present."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.HIDDEN]
        with pytest.raises(ValueError, match="READOUT"):
            get_abc_parametrization(semantic_roles=roles, parametrization="mup")

    def test_error_on_invalid_parametrization(self):
        """Test error on invalid parametrization name."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.READOUT]
        with pytest.raises(ValueError, match="Unknown parametrization"):
            get_abc_parametrization(semantic_roles=roles, parametrization="unknown")  # type: ignore

    def test_error_on_invalid_optimizer(self):
        """Test error on invalid optimizer name."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.READOUT]
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_abc_parametrization(semantic_roles=roles, optimizer="rmsprop")  # type: ignore

    def test_error_on_invalid_alignment(self):
        """Test error on invalid alignment name."""
        roles = [SemanticRole.EMBEDDING, SemanticRole.READOUT]
        with pytest.raises(ValueError, match="Unknown alignment"):
            get_abc_parametrization(semantic_roles=roles, alignment="partial")  # type: ignore
    
    def test_get_semantic_roles_for_mlp(self):
        """Test get_semantic_roles for a simple MLP (no embedding layers)."""
        model = SmallMLP()
        layers = get_managed_layers(model)
        roles = get_semantic_roles(layers)
        
        # MLP has 3 LINEAR layers and no EMBEDDING layers:
        # - First LINEAR becomes EMBEDDING (input projection)
        # - Middle LINEAR is HIDDEN
        # - Last LINEAR is READOUT
        assert len(roles) == 3
        assert roles == [SemanticRole.EMBEDDING, SemanticRole.HIDDEN, SemanticRole.READOUT]


class SmallMLP(nn.Module):
    """Simple 3-layer MLP for testing."""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=False)
        self.l2 = nn.Linear(8, 8, bias=False)
        self.l3 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)


class TestSchedulerWithParametrization:
    """Tests for MaxPScheduler with named parametrization."""

    def test_scheduler_with_named_parametrization(self):
        """Test the simplest API: just provide parametrization name everywhere."""
        torch.manual_seed(0)
        model = SmallMLP()
        
        # Simplest API: no need to manually get cl values!
        groups = create_param_groups(model, lr_prefactor=0.1, parametrization="mup")
        opt = torch.optim.AdamW(groups)
        
        sched = MaxPScheduler(
            opt, model, 
            parametrization="mup",
            lr_prefactor=0.1, 
            solver_warmup_steps=3
        )
        
        # For MLP, all LINEAR layers → last one is READOUT, others are HIDDEN
        # SmallMLP: l1 (HIDDEN), l2 (HIDDEN), l3 (READOUT) - no EMBEDDING since all are nn.Linear
        layers = get_managed_layers(model)
        roles = get_semantic_roles(layers)
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="mup", optimizer="adam", alignment="full")
        
        # Verify internal state uses muP values
        assert sched.al == abc.al
        assert sched.bl == abc.bl
        assert sched._param_name == "mup-adam-full"
        
        X0 = torch.randn(16, 4)
        sched.capture_initial(X0)
        
        # Run a few warmup steps
        for _ in range(3):
            opt.zero_grad()
            y = model(X0).sum()
            y.backward()
            opt.step()
            sched.step(X0)
        
        assert len(sched.get_last_lr()) == 3

    def test_scheduler_with_sgd_and_named_parametrization(self):
        torch.manual_seed(0)
        model = SmallMLP()
        
        # Use SGD-specific parametrization for both groups and scheduler
        groups = create_param_groups(
            model, lr_prefactor=0.1, 
            parametrization="ntk", optimizer="sgd", alignment="no"
        )
        opt = torch.optim.SGD(groups, lr=0.1)
        
        sched = MaxPScheduler(
            opt, model,
            parametrization="ntk",
            alignment_assumption="no",
            lr_prefactor=0.1,
            solver_warmup_steps=0
        )
        
        layers = get_managed_layers(model)
        roles = get_semantic_roles(layers)
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="ntk", optimizer="sgd", alignment="no")
        
        # Verify it picked up SGD-specific values
        assert sched.optimizer_type == "sgd"
        assert sched._param_name == "ntk-sgd-no"
        assert sched.al == abc.al
        assert sched.bl == abc.bl

    def test_create_param_groups_with_parametrization(self):
        """Test create_param_groups with named parametrization."""
        model = SmallMLP()
        
        # Get expected cl values for comparison
        layers = get_managed_layers(model)
        roles = get_semantic_roles(layers)
        abc = get_abc_parametrization(semantic_roles=roles, parametrization="sp", optimizer="adam", alignment="full")
        
        groups = create_param_groups(model, lr_prefactor=0.1, parametrization="sp")
        
        # Should have 3 managed groups (Linear layers) + 0 other params (no biases)
        assert len(groups) == 3
        for i, group in enumerate(groups):
            assert group["maxp_managed"] is True
            assert group["c"] == abc.cl[i]

    def test_create_param_groups_error_when_both_cl_and_parametrization(self):
        model = SmallMLP()
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_param_groups(
                model, lr_prefactor=0.1,
                cl=[0.0, 0.5, 0.5],
                parametrization="mup"
            )

    def test_create_param_groups_error_when_neither_cl_nor_parametrization(self):
        model = SmallMLP()
        
        with pytest.raises(ValueError, match="Must provide either"):
            create_param_groups(model, lr_prefactor=0.1)

    def test_error_when_both_parametrization_and_al_provided(self):
        model = SmallMLP()
        groups = create_param_groups(model, lr_prefactor=0.1, cl=[0.0, 0.5, 0.5])
        opt = torch.optim.AdamW(groups)
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            MaxPScheduler(
                opt, model,
                parametrization="mup",
                al=[0.0, 0.5, 0.5],
                lr_prefactor=0.1
            )

    def test_error_when_neither_parametrization_nor_al_provided(self):
        model = SmallMLP()
        groups = create_param_groups(model, lr_prefactor=0.1, cl=[0.0, 0.5, 0.5])
        opt = torch.optim.AdamW(groups)
        
        with pytest.raises(ValueError, match="Must provide either"):
            MaxPScheduler(opt, model, lr_prefactor=0.1)

    def test_custom_al_bl_still_works(self):
        torch.manual_seed(0)
        model = SmallMLP()
        al = [0.0, 0.5, 0.5]
        bl = [0.0, 0.0, 0.0]
        cl = [0.0, 0.5, 0.5]

        groups = create_param_groups(model, lr_prefactor=0.1, cl=cl)
        opt = torch.optim.AdamW(groups)

        sched = MaxPScheduler(
            opt, model,
            al=al, bl=bl,
            lr_prefactor=0.1,
            solver_warmup_steps=0
        )

        assert sched.al == al
        assert sched.bl == bl
        assert sched._param_name is None  # Custom values, no name


class TinyNetWithLayerNorm(nn.Module):
    """Simple model with Linear and LayerNorm layers for testing."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 8, bias=False)
        self.ln = nn.LayerNorm(8)
        self.l2 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.ln(self.l1(x)))
        return self.l2(x)


class TestLayerNormSemanticRole:
    """Tests for LayerNorm semantic role assignment."""

    def test_layernorm_gets_embedding_semantic_role(self):
        """Test that LayerNorm scale parameter is assigned EMBEDDING semantic role.

        According to the paper: 'the embedding layers include the embeddings,
        positional embeddings and the Layernorm scale parameter'.
        """
        model = TinyNetWithLayerNorm()
        layers = get_managed_layers(model)
        roles = get_semantic_roles(layers)

        # Should have 3 managed layers: l1 (LINEAR), ln.weight (EMBEDDING), l2 (LINEAR)
        assert len(layers) == 3
        assert len(roles) == 3

        # Find the LayerNorm layer and verify its role
        ln_idx = next(i for i, l in enumerate(layers) if 'ln.weight' in l.name)
        assert layers[ln_idx].semantic_role == SemanticRole.EMBEDDING
        assert roles[ln_idx] == SemanticRole.EMBEDDING

        # l1 should be HIDDEN (has_embedding_layer is True due to ln.weight)
        # l2 should be READOUT (last LINEAR)
        l1_idx = next(i for i, l in enumerate(layers) if 'l1' in l.name)
        l2_idx = next(i for i, l in enumerate(layers) if 'l2' in l.name)

        assert roles[l1_idx] == SemanticRole.HIDDEN
        assert roles[l2_idx] == SemanticRole.READOUT

    def test_layernorm_fan_in_is_correct(self):
        """Test that LayerNorm layer info has correct fan_in (normalized_shape)."""
        model = TinyNetWithLayerNorm()
        layers = get_managed_layers(model)

        # Find the LayerNorm layer
        ln_layer = next(l for l in layers if 'ln.weight' in l.name)

        # fan_in should be 8 (the normalized_shape of LayerNorm(8))
        assert ln_layer.fan_in == 8
        assert ln_layer.fan_out == 1
