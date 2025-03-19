import os
import torch
from addict import Dict
from _utils import profile_model_eval, OpLevelPerfSummary, torchbench_runner


def parse_args(runner):
    args = Dict()
    args.enable_activation_checkpointing = False
    args.float32 = True
    runner.args = args
    return runner


def main():
    runner = parse_args(torchbench_runner)
    summary = OpLevelPerfSummary()
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    with open(os.path.join(current_dir, 'torchbench_list.txt'), encoding='utf-8') as f:
        for line in f:
            data = line.strip().split(',')
            model_name = data[0].strip()
            batch_size = data[1].strip()
            print(model_name)
            device, benchmark_name, model, example_inputs, batch_size = \
                runner.load_model(torch.device('xpu'), model_name, batch_size=int(batch_size))
            warm_up_iters = 3
            for i in range(warm_up_iters):
                output = model(*example_inputs)
            prof = profile_model_eval(model, example_inputs)
            summary.append(benchmark_name, 'fp32', prof)
    summary.store('test.json')


if __name__ == '__main__':
    main()
