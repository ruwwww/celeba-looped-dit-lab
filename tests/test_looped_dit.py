from __future__ import annotations

import torch

from models import FlowMatching, LoopedDiT, euler_sample


def make_test_model() -> LoopedDiT:
    return LoopedDiT(
        image_size=16,
        in_channels=32,
        hidden_size=64,
        num_heads=4,
        mlp_ratio=2.0,
        loop_split=(1, 2, 1),
        num_classes=64,
        class_dropout_prob=0.0,
    )


def test_forward_preserves_latent_shape() -> None:
    model = make_test_model()
    x = torch.randn(2, 32, 16, 16)
    t = torch.tensor([0.25, 0.75])
    y = torch.tensor([1, 17])

    output, exits = model(x, t, y, num_loops=4)

    assert output.shape == x.shape
    assert exits == {}


def test_forward_returns_requested_deep_supervision_exits() -> None:
    model = make_test_model()
    x = torch.randn(2, 32, 16, 16)
    t = torch.rand(2)

    output, exits = model(x, t, num_loops=4, exit_loops=(1, 3))

    assert output.shape == x.shape
    assert set(exits) == {1, 3}
    assert all(exit_output.shape == x.shape for exit_output in exits.values())


def test_flow_matching_loss_supports_backward() -> None:
    torch.manual_seed(0)
    model = make_test_model()
    flow = FlowMatching(model, num_loops=4)
    x1 = torch.randn(2, 32, 16, 16)
    y = torch.tensor([2, 9])

    loss = flow.loss(x1, y=y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_five_step_euler_sampling_with_cfg() -> None:
    torch.manual_seed(0)
    model = make_test_model()
    labels = torch.tensor([4, 12])

    samples = euler_sample(
        model,
        shape=(2, 32, 16, 16),
        y=labels,
        cfg_scale=2.0,
        steps=5,
        num_loops=2,
    )

    assert samples.shape == (2, 32, 16, 16)
    assert torch.isfinite(samples).all()
