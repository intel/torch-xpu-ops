# Benchmark Test Scope

This directory defines the models, backends and precisions that the E2E benchmark runs. Scope lives in YAML so that the tiers and the full run are defined in one place.

## Tiers

- `p0`, `p1`, `p2`: priority tiers. Each is a separate selection, not a subset of another. Each file defines backends, precision, suite, scenario and model lists.
- `config.yaml`: the full scope. Its model lists must equal the union of p0, p1 and p2.

| Tier | Backends | Precisions per mode | Models (inference / training) | Cases |
|---|---|---|---|---|
| p0 | 6 | 1 (inference: bfloat16, training: amp_fp16) | 106 / 88 | 2328 |
| p1 | 3 | 4 per mode | 106 / 88 | 4656 |
| p2 | 1 | 5 per mode (all dtypes) | 114 / 109 | 2230 |
| full | per tier | per tier | 220 / 197 | 9214 |

The full case count is the sum of the p0, p1 and p2 case counts (2328 + 4656 + 2230 = 9214), not a product over the full config.

## Config structure

- `models`: nested as `suite -> mode -> [model names]`. Modes are `inference` and `training`.
- A precision entry without `mode` runs in both modes. An entry with `mode` runs only in that mode.

## Case count

$$
\text{cases} = \sum_{\text{mode}} \left( |\text{backends}| \times |\text{scenario}| \times |\text{precision}_{\text{mode}}| \times |\text{models}_{\text{mode}}| \right)
$$

where $|\text{models}_{\text{mode}}|$ is the sum of model-list lengths across suites for that mode.

## Editing and verification

1. Edit the tier file, then update `config.yaml` so it still equals p0 + p1 + p2.
2. Use model names exactly as the suite expects them (for example `hf_Bert`, `timm_vision_transformer`).
3. Keep the `# N models` comment on each mode line and the total case comment in sync with the lists.
4. Verify the union: each tier model must appear in `config.yaml` for the same suite and mode, and nothing extra may appear there.

## Runner selection

The E2E workflow selects scope with the `model_list` input:

- `p0`, `p1`, `p2`: reads `./benchmarks/dynamo/<tier>/config.yaml`.
- `full`: reads the union of the p0, p1 and p2 config files. The root `config.yaml` is not used by the runner.

The `.ci/benchmarks/` tree is copied to `./benchmarks/dynamo/` before the run.
