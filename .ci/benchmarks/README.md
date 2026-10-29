# Benchmark Test Scope

This directory defines the models, backends, and precisions executed by the E2E benchmark on XPU. Scope is specified in YAML, with one file per priority tier (p0, p1, p2).

## Purpose

The full PyTorch benchmark matrix exceeds the XPU capacity available for every change. The tiers partition the matrix by priority: p0 provides a fast signal on every change, while p1 and p2 extend coverage as capacity allows, surfacing failures that p0 does not exercise.

## Basis of each tier

- **p0 (CUDA parity)**: the models and modes tracked for CUDA by the stock PyTorch performance dashboard (HUD), using the suite lists in `pytorch/benchmarks/dynamo/`. XPU results can be compared directly with the corresponding CUDA results.
- **p1 (current PyTorch coverage)**: the same models as p0, evaluated across the precisions exercised by current PyTorch benchmarking (float32, float16, bfloat16, amp_fp16, amp_bf16). Coverage is extended along the dtype axis without increasing the model set.
- **p2 (legacy PyTorch coverage)**: the models from the legacy PyTorch lists that p0 and p1 do not include (for example `detectron2_*`, `hf_*`, `timm_*`, and quantized QAT models), evaluated with all five precisions per mode.

The tiers are ordered by cost and diagnostic yield. p0 is the lowest-cost parity check. p1 isolates precision-specific failures on the p0 models. p2 isolates model-specific failures that are unlikely to surface within p0 or p1.

## Purpose of each tier

PyTorch component acceptance tests run the full scope (the union of p0, p1, and p2)

| Tier | Purpose | Schedule |
|---|---|---|
| p0 | Regression gate with CUDA parity | CI (every change) |
| p1 | Precision coverage on the p0 models | Nightly |
| p2 | Broad coverage of additional models and dtypes | Weekly |


## Tiers

- `p0`, `p1`, `p2` are independent selections. They differ in backend and precision; p0 and p1 share the same model lists. Each file defines the backends, precisions, suites, scenarios, and models.
- Only the `default` backend is active. Commented backends (`dynamic`, `cpp_wrapper`, `freezing`, `aot_inductor`, `inductor_deterministic_perf`) are not executed unless uncommented; the full backend set will be added later.

| Tier | Backends (active) | Precisions per mode | Models (inference / training) | Cases |
|---|---|---|---|---|
| p0 | 1 (default) | 1 (inference: bfloat16, training: amp_fp16) | 106 / 88 | 388 |
| p1 | 1 (default) | 4 per mode | 106 / 88 | 1552 |
| p2 | 1 (default) | 5 per mode (all dtypes) | 114 / 109 | 2230 |

A case is one (backend, scenario, precision, model) combination per mode. Scenarios are `accuracy` and `performance`; for example, p0 comprises $1 \times 2 \times 1 \times 106 + 1 \times 2 \times 1 \times 88 = 388$ cases.

## Time cost

Wall time scales with the number of cases:

$$
T_{\text{tier}} \approx N_{\text{cases}} \times \bar{t}_{\text{case}}
$$

where $\bar{t}_{\text{case}}$ is the mean time per case, including model load and compilation, and varies by suite and dtype.

| Tier | Cases | Relative cost | Measured wall time |
|---|---|---|---|
| p0 | 388 | 1× (baseline) | to measure |
| p1 | 1552 | 4× p0 | to measure |
| p2 | 2230 | about 5.7× p0 | to measure |

The E2E job is capped at 3600 minutes (`timeout-minutes` in `_linux_e2e.yml`). The cap applies to the whole job, so a tier must finish well within it. Fill the measured column from a p0 run first, then derive p1 and p2 from it.

## Config structure

- `models`: nested as `suite -> mode -> [model names]`. Modes are `inference` and `training`.
- A precision entry without `mode` runs in both modes. An entry with `mode` runs only in that mode.

## Case count

$$
\text{cases} = \sum_{\text{mode}} \left( |\text{backends}| \times |\text{scenario}| \times |\text{precision}_{\text{mode}}| \times |\text{models}_{\text{mode}}| \right)
$$

where $|\text{models}_{\text{mode}}|$ is the sum of model-list lengths across suites for that mode.

## Editing and verification

1. Edit the tier file and keep the `# N models` comment on each mode line consistent with the list length.
2. Use model names exactly as the suite defines them (for example, `hf_Bert` or `timm_vision_transformer`).
3. Update the case table whenever backends, precisions, or model counts change.
4. Before adding a model to a tier, confirm that the same suite and mode is not already listed in another tier, so that each tier remains distinct.

## Runner selection

The E2E workflow selects scope through the `test_scope` input:

- `p0`, `p1`, `p2`: read `./benchmarks/dynamo/<tier>/config.yaml`.
- `full`: the union of the p0, p1, and p2 selections. This directory has no root `config.yaml`.

The `.ci/benchmarks/` tree is copied to `./benchmarks/dynamo/` before the run.
