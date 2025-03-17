import json
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


class OpLevelPerfSummary:
    def __init__(self, prof=None, fname=None):
        if prof is not None:
            self.events = {}
            for event in prof.key_averages(group_by_input_shape=True):
                if event.input_shapes is not None and len(event.input_shapes) > 0:
                    time_us = self._transform_time(event.self_device_time_total_str)
                    if time_us > 0.1:
                        _key = self._format_key(event)
                        assert _key not in self.events
                        self.events[_key] = time_us
        else:
            self.load(fname)

    def _format_key(self, event):
        return f"{event.key}|{event.count}|{event.input_shapes}"

    def _transform_time(self, time_str):
        time = None
        if time_str.endswith('us'):
            time = float(time_str[:-2].strip())
        elif time_str.endswith('ms'):
            time = float(time_str[:-2].strip()) * 1000.0
        elif time_str.endswith('s'):
            time = float(time_str[:-1].strip()) * 1000.0 * 1000.0
        return time

    def compare(self, summary2):
        gap = {}
        for key in self.events.keys():
            self_time_us = self.events[key]
            other_time_us = summary2.events[key]
            relative_gap = (self_time_us - other_time_us) / self_time_us
            gap[key] = relative_gap
        sorted_list = sorted(gap.items(), key=lambda x: (x[0], -x[1]))
        return sorted_list

    def store(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, ensure_ascii=False, indent=4)

    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            self.events = json.load(f)
