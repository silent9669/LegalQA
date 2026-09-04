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
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
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
def test_liger_fused_linear_ce_forward_backward_parity():
    """Verify Liger fused-linear cross entropy yields finite losses and gradients with high cosine similarity to reference CE."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    vocab_size = 256

    # 1. Base tensors in FP16 (matching T4 compute dtype)
    hidden_base = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
    weight_base = torch.randn(vocab_size, hidden_dim, device=device, dtype=torch.float16)

    # Independent cloned tensors with requires_grad=True
    hidden_ref = hidden_base.clone().detach().requires_grad_(True)
    weight_ref = weight_base.clone().detach().requires_grad_(True)

    hidden_liger = hidden_base.clone().detach().requires_grad_(True)
    weight_liger = weight_base.clone().detach().requires_grad_(True)

    # Labels with ignored index -100 masking (completion-only simulation)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, :4] = -100

    # 2. Reference branch: standard linear projection (FP32 cast) + cross_entropy
    logits_ref = F.linear(hidden_ref.float(), weight_ref.float())
    loss_ref = F.cross_entropy(
        logits_ref.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss_ref), "Reference loss must be finite"
    loss_ref.backward()

    # 3. Liger branch: fused-linear cross-entropy
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    loss_liger = loss_fn(
        hidden_liger.view(-1, hidden_dim),
        weight_liger,
        labels.view(-1),
    )
    assert torch.isfinite(loss_liger), "Liger loss must be finite"
    loss_liger.backward()

    # 4. Forward loss assertions
    abs_loss_diff = torch.abs(loss_ref.detach() - loss_liger.detach()).item()
    assert abs_loss_diff < 0.05, f"Loss mismatch: ref={loss_ref.item()}, liger={loss_liger.item()}, diff={abs_loss_diff}"

    # 5. Backward gradient assertions
    assert hidden_ref.grad is not None, "Reference hidden grad must exist"
    assert hidden_liger.grad is not None, "Liger hidden grad must exist"
    assert weight_ref.grad is not None, "Reference weight grad must exist"
    assert weight_liger.grad is not None, "Liger weight grad must exist"

    assert torch.isfinite(hidden_liger.grad).all(), "Liger hidden grad must be finite"
    assert torch.isfinite(weight_liger.grad).all(), "Liger weight grad must be finite"

    # Shapes must match exactly
    assert hidden_liger.grad.shape == hidden_ref.grad.shape, "Hidden grad shapes must match"
    assert weight_liger.grad.shape == weight_ref.grad.shape, "Weight grad shapes must match"

    # Cosine similarity between gradients (flat vectors)
    cos_hidden = F.cosine_similarity(
        hidden_liger.grad.float().view(-1),
        hidden_ref.grad.float().view(-1),
        dim=0,
    ).item()
    assert cos_hidden >= 0.99, f"Hidden gradient cosine similarity too low: {cos_hidden}"

    cos_weight = F.cosine_similarity(
        weight_liger.grad.float().view(-1),
        weight_ref.grad.float().view(-1),
        dim=0,
    ).item()
    assert cos_weight >= 0.99, f"Weight gradient cosine similarity too low: {cos_weight}"

    # Maximum absolute gradient difference within FP16 tolerance
    max_grad_diff_h = torch.max(torch.abs(hidden_liger.grad.float() - hidden_ref.grad.float())).item()
    max_grad_diff_w = torch.max(torch.abs(weight_liger.grad.float() - weight_ref.grad.float())).item()
    assert max_grad_diff_h < 0.05, f"Hidden grad max diff {max_grad_diff_h} exceeds tolerance"
    assert max_grad_diff_w < 0.05, f"Weight grad max diff {max_grad_diff_w} exceeds tolerance"
