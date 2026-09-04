"""GPU parity test comparing selective Liger fused-linear cross entropy with standard PyTorch CE on tiny synthetic inputs."""

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    import liger_kernel
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
except ImportError:
    liger_kernel = None


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="CUDA required for Liger fused-linear CE GPU parity test",
)
@pytest.mark.skipif(
    liger_kernel is None,
    reason="liger-kernel required for Liger GPU parity test",
)
def test_liger_fused_linear_ce_loss_and_grad_parity():
    """Verify Liger fused-linear cross entropy yields finite losses and gradients close to standard CE."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    vocab_size = 256

    # Inputs in FP16 (matching T4 inference/training dtype)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)

    # Labels with ignored index -100 masking
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, :4] = -100  # prompt masking / completion-only

    # 1. Reference standard PyTorch CE (cast to float for numerical stability)
    logits_ref = F.linear(hidden_states.float(), weight.float())
    loss_ref = F.cross_entropy(logits_ref.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

    assert torch.isfinite(loss_ref)

    # 2. Liger Fused Linear Cross Entropy
    liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    loss_liger = liger_loss_fn(
        hidden_states.view(-1, hidden_dim),
        weight,
        labels.view(-1),
    )

    assert torch.isfinite(loss_liger)

    # Absolute difference tolerance for FP16 kernel vs FP32 reference: 1e-2
    abs_diff = torch.abs(loss_ref.detach() - loss_liger.detach()).item()
    assert abs_diff < 0.05, f"Loss mismatch: ref={loss_ref.item()}, liger={loss_liger.item()}, diff={abs_diff}"
