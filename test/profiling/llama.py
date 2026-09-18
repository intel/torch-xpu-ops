# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

SORT_BY = "xpu_time_total"
DEFAULT_ITERS = 5
DTYPE = torch.float16

PROMPT = "If Alice is older than Bob, and Bob is older than Charlie, who is the youngest? Explain your reasoning."


def run_profile(iters=DEFAULT_ITERS, name=model_name):
    """Yield (iteration, prof). The ``datatype: ... ; i: N`` header is printed
    here because .github/scripts/llama_summary.py splits the log on it."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=DTYPE,
    )
    model.eval().to("xpu")

    inputs = tokenizer(PROMPT, return_tensors="pt").to("xpu")

    with torch.no_grad():
        for i in range(iters):
            print(
                "datatype:",
                DTYPE,
                "; i:",
                i,
            )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.XPU,
                ]
            ) as prof:
                model.generate(**inputs, max_new_tokens=1)
            yield i, prof


if __name__ == "__main__":
    for _, p in run_profile():
        print(p.key_averages().table(sort_by=SORT_BY, row_limit=-1))
