import torch
import pytest


@pytest.mark.parametrize("mode", ['reduced', 'complete', 'r'])
def test_linalg_qr(mode):
    A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    A_xpu = A.to('xpu')
    
    Q, R = torch.linalg.qr(A, mode=mode)
    Q_xpu, R_xpu = torch.linalg.qr(A_xpu, mode=mode)


    print("==== CPU ====")
    print("A",A)
    print("Q",Q)
    print("R",R)
    print("\n==== XPU ====")
    print("A_xpu",A_xpu)
    print("Q_xpu",Q_xpu)
    print("R_xpu",R_xpu)

    assert torch.allclose(Q, Q_xpu.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(R, R_xpu.cpu(), atol=1e-5, rtol=1e-5)
