from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)
model.eval().to("xpu")

prompt = "If Alice is older than Bob, and Bob is older than Charlie, who is the youngest? Explain your reasoning."
inputs = tokenizer(prompt, return_tensors="pt").to("xpu")

with torch.no_grad():
    for i in range(5):
        print(
            "datatype:",
            torch.float16,
            "; i:",
            i,
        )
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]) as prof:
            outputs = model.generate(
                **inputs,
                max_new_tokens=1
            )
        print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=-1))
