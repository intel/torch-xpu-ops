import torch


prof_xpu = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU],
    record_shapes=True,
    with_stack=True,
)


def profile_model_eval(model, input_tensor):
    with prof_xpu:
        with torch.no_grad():
            model(input_tensor)
    return prof_xpu


def profile_model_train(model, input_tensor, loss_fn):
    with prof_xpu:
        loss = loss_fn(model(input_tensor))
        loss.backward()
    return prof_xpu


class OpLevelPerfSummary(object):
    def __init__(self, prof):
        self.events = []
        for event in prof.key_averages(group_by_input_shape=True):
            if event.input_shapes is not None and len(event.input_shapes) > 0:
                time_us = self._transform_time(event.self_device_time_total_str)
                if time_us > 0.1:
                    self.events.append({
                        'name': event.key,
                        'count': event.count,
                        'input_shapes': event.input_shapes,
                        'self_device_time_us': time_us,
                    })

    def _transform_time(self, time_str):
        time = None
        if time_str.endswith('us'):
            time = float(time_str[:-2].strip())
        elif time_str.endswith('ms'):
            time = float(time_str[:-2].strip()) * 1000.0
        elif time_str.endswith('s'):
            time = float(time_str[:-1].strip()) * 1000.0 * 1000.0
        return time
