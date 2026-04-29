# Owner(s): ["module: dynamo"]

import sys

sys.path.append("../../../../test/dynamo")

import torch
import torch._dynamo.test_case

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class AutogradFunctionTests(torch._dynamo.test_case.TestCase):
    def test_inplace_op_with_side_effect_wrong_grad(self):
        # Repro for https://github.com/pytorch/pytorch/issues/180642
        # In-place op inside autograd.Function.forward combined with a side
        # effect (list append) causes collect_intermediate_outputs to add a
        # pre-mutation alias as an extra subgraph output. Because the alias
        # shares the same TensorImpl as the real return value,
        # set_gradient_edge overwrites output_nr, routing the backward
        # gradient to the wrong slot and producing zero gradients.
        captured = []

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                intermediate = torch.sin(x)
                captured.append(intermediate)
                # The in-place += is intentional: it creates an aliased
                # intermediate that triggers the wrong-gradient bug.
                loss = torch.tensor(0.0, device=x.device)
                loss += x.sum()
                return loss

            @staticmethod
            def backward(ctx, grad):
                return grad.expand(8)

        def fn(x):
            return Foo.apply(x)

        x = torch.randn(8, device=device_type, requires_grad=True)

        # Eager reference
        x_ref = x.detach().clone().requires_grad_(True)
        captured.clear()
        out_ref = fn(x_ref)
        out_ref.backward()

        # Compiled
        torch._dynamo.reset()
        captured.clear()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()

        self.assertEqual(x_ref.grad, x_c.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
