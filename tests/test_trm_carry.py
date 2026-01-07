"""Tests for TRM carry state management."""
import torch
import pytest

from src.model.trm import TRM, TRMInnerCarry, TRMCarry


class TestTRMCarry:
    """Test carry dataclasses and TRM carry methods."""

    @pytest.fixture
    def model(self):
        return TRM(seq_len=82, input_dim=512)

    def test_empty_carry_shape(self, model):
        """empty_carry creates tensors with correct shape."""
        carry = model.empty_carry(batch_size=4)

        assert carry.z_H.shape == (4, 82, 512)
        assert carry.z_L.shape == (4, 82, 512)

    def test_reset_carry_initializes_flagged_sequences(self, model):
        """reset_carry initializes z_H/z_L to H_init/L_init where reset_flag=True."""
        carry = model.empty_carry(batch_size=4)
        # Fill with zeros so we can detect the reset
        carry.z_H.zero_()
        carry.z_L.zero_()

        reset_flag = torch.tensor([True, False, True, False])
        new_carry = model.reset_carry(reset_flag, carry)

        # Sequences 0 and 2 should be initialized to H_init/L_init
        assert torch.allclose(new_carry.z_H[0, 0], model.H_init)
        assert torch.allclose(new_carry.z_H[2, 0], model.H_init)
        assert torch.allclose(new_carry.z_L[0, 0], model.L_init)
        assert torch.allclose(new_carry.z_L[2, 0], model.L_init)

    def test_reset_carry_preserves_non_flagged_sequences(self, model):
        """reset_carry preserves z_H/z_L where reset_flag=False."""
        carry = model.empty_carry(batch_size=4)
        # Fill with a known value
        carry.z_H.fill_(42.0)
        carry.z_L.fill_(43.0)

        reset_flag = torch.tensor([True, False, True, False])
        new_carry = model.reset_carry(reset_flag, carry)

        # Sequences 1 and 3 should preserve original values
        assert torch.all(new_carry.z_H[1] == 42.0)
        assert torch.all(new_carry.z_H[3] == 42.0)
        assert torch.all(new_carry.z_L[1] == 43.0)
        assert torch.all(new_carry.z_L[3] == 43.0)

    def test_trm_carry_dataclass(self):
        """TRMCarry wraps TRMInnerCarry with steps and halted."""
        inner = TRMInnerCarry(
            z_H=torch.randn(2, 82, 512),
            z_L=torch.randn(2, 82, 512),
        )
        carry = TRMCarry(
            inner_carry=inner,
            steps=torch.tensor([3, 5]),
            halted=torch.tensor([False, True]),
        )

        assert carry.inner_carry.z_H.shape == (2, 82, 512)
        assert carry.steps.tolist() == [3, 5]
        assert carry.halted.tolist() == [False, True]


class TestTRMForward:
    """Test TRM.forward() method for single ACT iteration."""

    @pytest.fixture
    def model(self):
        return TRM(seq_len=82, input_dim=512, context_len=1)

    def test_forward_output_shapes(self, model):
        """forward() returns correct output shapes."""
        input_emb = torch.randn(2, 81, 512)  # No context, forward adds it
        carry = model.initial_carry(input_emb)

        new_carry, outputs = model.forward(carry, input_emb)

        assert new_carry.inner_carry.z_H.shape == (2, 82, 512)
        assert new_carry.inner_carry.z_L.shape == (2, 82, 512)
        assert outputs["logits"].shape == (2, 81, 11)
        assert outputs["q_halt_logits"].shape == (2,)
        assert new_carry.steps.shape == (2,)
        assert new_carry.halted.shape == (2,)

    def test_forward_carry_is_detached(self, model):
        """forward() returns carry with detached tensors."""
        input_emb = torch.randn(2, 81, 512)
        carry = model.initial_carry(input_emb)

        new_carry, _ = model.forward(carry, input_emb)

        assert not new_carry.inner_carry.z_H.requires_grad
        assert not new_carry.inner_carry.z_L.requires_grad

    def test_forward_q_halt_initial_value(self, model):
        """forward() q_halt starts near -5 due to bias initialization."""
        input_emb = torch.randn(2, 81, 512)
        carry = model.initial_carry(input_emb)

        _, outputs = model.forward(carry, input_emb)

        # q_head bias is -5, so initial q_halt should be around -5
        assert torch.all(outputs["q_halt_logits"] < 0), "Initial q_halt should be negative"

    def test_initial_carry_halted_true(self, model):
        """initial_carry() starts with halted=True to trigger reset."""
        input_emb = torch.randn(2, 81, 512)
        carry = model.initial_carry(input_emb)

        assert torch.all(carry.halted), "Initial carry should have halted=True"
        assert torch.all(carry.steps == 0), "Initial carry should have steps=0"

    def test_forward_increments_steps(self, model):
        """forward() increments step count."""
        model.eval()  # Disable exploration
        input_emb = torch.randn(2, 81, 512)
        carry = model.initial_carry(input_emb)

        carry, _ = model.forward(carry, input_emb)
        assert torch.all(carry.steps == 1)

        carry, _ = model.forward(carry, input_emb)
        assert torch.all(carry.steps == 2)

    def test_forward_halts_at_max_steps(self, model):
        """forward() halts when max_steps reached."""
        model.eval()  # Disable exploration
        input_emb = torch.randn(2, 81, 512)
        carry = model.initial_carry(input_emb)

        # Run until max_steps (16)
        for _ in range(16):
            carry, _ = model.forward(carry, input_emb)

        assert torch.all(carry.halted), "All sequences should halt at max_steps"
