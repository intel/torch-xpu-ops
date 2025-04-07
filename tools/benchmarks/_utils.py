import os
import sys
import json
import torch
from pathlib import Path


def find_pytorch_root():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        print(parent.name)
        if parent.name == 'pytorch':
            return str(parent)
    raise FileNotFoundError('pytorch not found')


path = os.path.join(find_pytorch_root(), 'benchmarks/dynamo')
sys.path.append(path)
from torchbench import TorchBenchmarkRunner
from huggingface import HuggingfaceRunner


def apply_fake_validate_model(runner):
    def fake_validate_model(*args, **kwargs):
        return
    runner.validate_model = fake_validate_model


torchbench_runner = TorchBenchmarkRunner()
huggingface_runner = HuggingfaceRunner()
apply_fake_validate_model(torchbench_runner)
apply_fake_validate_model(huggingface_runner)


prof_xpu = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU],
    record_shapes=True,
    with_stack=True,
)


def dry_run_model_eval(model, inputs, niter):
    for i in range(niter):
        if isinstance(inputs, dict):
            model(**inputs)
        else:
            model(*inputs)


def profile_model_eval(model, inputs):
    with prof_xpu:
        if isinstance(inputs, dict):
            with torch.no_grad():
                model(**inputs)
        else:
            with torch.no_grad():
                model(*inputs)
    return prof_xpu


def profile_model_train(model, inputs, loss_fn):
    with prof_xpu:
        if isinstance(inputs, dict):
            loss = loss_fn(model(**inputs))
        else:
            loss = loss_fn(model(*inputs))
        loss.backward()
    return prof_xpu


class OpLevelPerfSummary:
    def __init__(self, saved_events=None):
        if saved_events is not None:
            self.load(saved_events)
        else:
            self.events = {}

    def append(self, name, dtype, prof):
        for event in prof.key_averages(group_by_input_shape=True):
            if event.input_shapes is not None and len(event.input_shapes) > 0:
                time_us = self._transform_time(event.self_device_time_total_str)
                if time_us > 0.1:
                    _key = self._format_key(name, dtype, event)
                    assert _key not in self.events
                    self.events[_key] = time_us

    def _format_key(self, name, dtype, event):
        return f"{name}|{dtype}|{event.key}|{event.count}|{event.input_shapes}"

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
