import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
@pytest.mark.parametrize(
    "shape",
    [
        # 2D matrices
        (5, 3),
        (3, 5),
        (8, 6),
        (6, 8),
        # 3D batched matrices
        (2, 3, 3),
        (3, 4, 4),
        (2, 5, 3),
        (2, 3, 5),
        # 4D batched matrices
        (2, 3, 4, 4),
        (1, 2, 5, 3),
        (2, 1, 3, 5),
        # Edge cases
        (1, 1),
        (10, 1),
        (1, 10),
        (2, 1, 1),
        (1, 1, 1),
    ],
)
def test_linalg_qr(dtype, mode, shape):
    A = torch.randn(shape, dtype=dtype)
    A_xpu = A.to("xpu")

    Q, R = torch.linalg.qr(A, mode=mode)
    Q_xpu, R_xpu = torch.linalg.qr(A_xpu, mode=mode)

    assert torch.allclose(Q, Q_xpu.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(R, R_xpu.cpu(), atol=1e-5, rtol=1e-5)
    assert Q_xpu.device.type == "xpu" and R_xpu.device.type == "xpu"
