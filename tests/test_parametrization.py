"""Tests for ABC parametrization utilities."""

import pytest
import torch
import torch.nn as nn

from maxp import MaxPScheduler, create_param_groups, get_abc_parametrization


class TestGetABCParametrization:
    """Tests for get_abc_parametrization function."""

    def test_mup_adam_full_4_layers(self):
        abc = get_abc_parametrization(n_layers=4, parametrization="mup", optimizer="adam", alignment="full")
        
        assert abc.al == [-0.5, 0.0, 0.0, 0.5]
        assert abc.bl == [0.5, 0.5, 0.5, 0.5]
        assert abc.cl == [0.5, 1.0, 1.0, 0.5]
        assert abc.name == "mup-adam-full"

    def test_mup_sgd_no_alignment(self):
        abc = get_abc_parametrization(n_layers=3, parametrization="mup", optimizer="sgd", alignment="no")
        
        assert abc.al == [-0.5, 0.0, 0.5]
        assert abc.bl == [0.5, 0.5, 0.5]
        assert abc.cl == [0.0, -0.5, 0.0]
        assert abc.name == "mup-sgd-no"

    def test_sp_parametrization(self):
        abc = get_abc_parametrization(n_layers=3, parametrization="sp", optimizer="adam", alignment="full")
        
        assert abc.al == [0.0, 0.0, 0.0]
        assert abc.bl == [0.0, 0.5, 0.5]
        assert abc.cl == [0.0, 1.0, 1.0]
        assert abc.name == "sp-adam-full"

    def test_ntk_parametrization(self):
        abc = get_abc_parametrization(n_layers=4, parametrization="ntk", optimizer="adam", alignment="full")
        
        assert abc.al == [0.0, 0.5, 0.5, 0.5]
        assert abc.bl == [0.0, 0.0, 0.0, 0.0]
        assert abc.cl == [0.0, 0.5, 0.5, 0.5]
        assert abc.name == "ntk-adam-full"

    def test_mfp_parametrization(self):
        abc = get_abc_parametrization(n_layers=4, parametrization="mfp", optimizer="sgd", alignment="full")
        
        assert abc.al == [0.0, 0.5, 0.5, 1.0]
        assert abc.bl == [0.0, 0.0, 0.0, 0.0]
        assert abc.cl == [-1.0, -1.0, -1.0, -1.0]
        assert abc.name == "mfp-sgd-full"

    def test_case_insensitive(self):
        abc1 = get_abc_parametrization(n_layers=3, parametrization="MUP", optimizer="ADAM", alignment="FULL")
        abc2 = get_abc_parametrization(n_layers=3, parametrization="mup", optimizer="adam", alignment="full")
        
        assert abc1.al == abc2.al
        assert abc1.bl == abc2.bl
        assert abc1.cl == abc2.cl

    def test_two_layers(self):
        abc = get_abc_parametrization(n_layers=2, parametrization="mup", optimizer="adam", alignment="full")
        
        # First layer + last layer only, no hidden layers
        assert abc.al == [-0.5, 0.5]
        assert abc.bl == [0.5, 0.5]
        assert abc.cl == [0.5, 0.5]

    def test_error_on_single_layer(self):
        with pytest.raises(ValueError, match="n_layers must be >= 2"):
            get_abc_parametrization(n_layers=1, parametrization="mup")

    def test_error_on_invalid_parametrization(self):
        with pytest.raises(ValueError, match="Unknown parametrization"):
            get_abc_parametrization(n_layers=3, parametrization="unknown")  # type: ignore

    def test_error_on_invalid_optimizer(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_abc_parametrization(n_layers=3, optimizer="rmsprop")  # type: ignore

    def test_error_on_invalid_alignment(self):
        with pytest.raises(ValueError, match="Unknown alignment"):
            get_abc_parametrization(n_layers=3, alignment="partial")  # type: ignore


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
        
        abc = get_abc_parametrization(n_layers=3, parametrization="mup", optimizer="adam", alignment="full")
        
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
        
        abc = get_abc_parametrization(n_layers=3, parametrization="ntk", optimizer="sgd", alignment="no")
        
        # Verify it picked up SGD-specific values
        assert sched.optimizer_type == "sgd"
        assert sched._param_name == "ntk-sgd-no"
        assert sched.al == abc.al
        assert sched.bl == abc.bl

    def test_create_param_groups_with_parametrization(self):
        """Test create_param_groups with named parametrization."""
        model = SmallMLP()
        
        # Get expected cl values for comparison
        abc = get_abc_parametrization(n_layers=3, parametrization="sp", optimizer="adam", alignment="full")
        
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
