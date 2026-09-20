"""Task 10: thin gate entrypoints share one runner/config (no algorithm drift)."""

from __future__ import annotations

from pathlib import Path

LAUNCHERS = [
    "scripts/modal_app.py",
    "scripts/run_gpu_gate.py",
    "scripts/run_pipeline.py",
]

# Algorithm defaults that must live in configs/task2/algorithm.yaml + the
# shared recipe adapter, never inline in a launcher.
FORBIDDEN_INLINE_ALGORITHM = ("LoraConfig(", "SYSTEM_PROMPT =", "peft_config =")


def test_launchers_route_through_shared_runner():
    for launcher in LAUNCHERS:
        source = Path(launcher).read_text(encoding="utf-8")
        assert ("run_gpu_gate" in source or "run_pipeline" in source or "run_platform_stage" in source), \
            f"{launcher} must call the shared runner, not fork the algorithm"


def test_launchers_carry_no_inline_algorithm_defaults():
    for launcher in LAUNCHERS:
        source = Path(launcher).read_text(encoding="utf-8")
        for token in FORBIDDEN_INLINE_ALGORITHM:
            assert token not in source, f"{launcher} must not inline algorithm: {token}"


def test_a100_stage_resolves_declared_runtime_profile():
    from scripts.run_gpu_gate import ALLOWED_A100_PROFILES, STAGE_RUNTIME_PROFILE

    assert STAGE_RUNTIME_PROFILE["a100_micro_probe"] == "colab_a100"
    assert set(ALLOWED_A100_PROFILES) == {"colab_a100", "modal_a100"}
