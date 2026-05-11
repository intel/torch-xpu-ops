# Owner(s): ["module: intel"]
"""
Regression test for:
    torch._dynamo.exc.InternalTorchDynamoError:
    TypeError: cannot create weak reference to 'torch.Event' object

This error occurred when torch.xpu.Event objects were encountered during
torch.compile tracing because XPU Event objects did not support weak
references.  The fix registers torch.xpu.current_stream() as a handled
stream function in dynamo (via XpuStreamVariable) so that Event objects
are properly traced without needing weak references.

See: https://github.com/pytorch/pytorch/pull/182792
"""

import pytest
import torch

xpu_available = torch.xpu.is_available()


@pytest.mark.skipif(not xpu_available, reason="XPU not available")
def test_xpu_event_in_compiled_function():
    """Verify that torch.xpu.Event objects can be created and used inside a
    torch.compile region without raising
    'cannot create weak reference to torch.Event object'."""

    def fn(x):
        cur_stream = torch.xpu.current_stream()
        new_stream = torch.xpu.Stream()

        x = torch.mul(x, 1)
        x = torch.add(x, 2)
        x = torch.add(x, 3)

        event = cur_stream.record_event()
        event.query()

        new_stream.wait_event(event)
        with torch.xpu.stream(new_stream):
            x = torch.add(x, 4)

        new_event = torch.xpu.Event()
        new_event.record(new_stream)
        new_event.wait(cur_stream)
        x = torch.add(x, 5)

        new_event.synchronize()

        x = torch.relu(x)
        x = torch.cos(x)
        return x

    x = torch.randn((2, 2), device="xpu")
    ref = fn(x)
    # Use 'eager' backend: the weak reference error occurs during dynamo's
    # tracing phase (VariableBuilder), not during backend compilation.
    # 'eager' backend still exercises the full dynamo tracing path where the
    # error was triggered, making it sufficient to catch this regression.
    opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
    res = opt_fn(x)
    assert torch.allclose(ref, res)


@pytest.mark.skipif(not xpu_available, reason="XPU not available")
def test_xpu_current_stream_attrs():
    """Verify that torch.xpu.current_stream().sycl_queue is accessible
    under torch.compile and matches eager behavior."""

    def fn(x):
        return torch.xpu.current_stream().sycl_queue

    x = torch.zeros(1, device="xpu")
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    assert compiled(x) == fn(x)


if __name__ == "__main__":
    raise RuntimeError(
        "Run this test file using pytest: pytest test/repro/test_xpu_dynamo_event.py"
    )
