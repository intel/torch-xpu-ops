"""Constants and column definitions for benchmark comparison."""

KNOWN_SUITES = {"huggingface", "timm_models", "torchbench", "pt2e"}
KNOWN_DATA_TYPES = {"float32", "bfloat16", "float16", "amp_bf16", "amp_fp16", "int8"}
KNOWN_MODES = {"inference", "training"}
DEFAULT_PERF_THRESHOLD = 0.1  # 10% change = regression / improvement

# PT2E accuracy thresholds (max allowed top1 drop as fraction)
PT2E_ACC_THRESHOLDS = {
    "float32": 0.0,   # 0% tolerance
    "int8": 0.05,     # 5% tolerance
}

MERGE_KEYS = ["suite", "data_type", "mode", "model"]

ACC_CSV_COLS = ["dev", "name", "batch_size", "accuracy"]
PERF_CSV_COLS = ["dev", "name", "batch_size", "speedup", "abs_latency"]

ACC_OUTPUT_COLS = [
    "suite", "data_type", "mode", "model",
    "batch_size_target", "accuracy_target",
    "batch_size_baseline", "accuracy_baseline",
    "comparison",
]
PERF_OUTPUT_COLS = [
    "suite", "data_type", "mode", "model",
    "batch_size_target", "inductor_target", "eager_target",
    "batch_size_baseline", "inductor_baseline", "eager_baseline",
    "inductor_ratio", "eager_ratio", "comparison",
]

PT2E_ACC_OUTPUT_COLS = [
    "suite", "mode", "model", "category",
    "fp32_target", "int8_target", "int8/fp32_target",
    "fp32_baseline", "int8_baseline", "int8/fp32_baseline",
    "fp32_comparison", "int8_comparison",
]
PT2E_PERF_OUTPUT_COLS = [
    "suite", "mode", "model",
    "fp32_target", "symm_target", "asymm_target",
    "symm/fp32_target", "asymm/fp32_target",
    "fp32_baseline", "symm_baseline", "asymm_baseline",
    "symm/fp32_baseline", "asymm/fp32_baseline",
    "comparison",
]

SUMMARY_LEVELS = [
    ("Overall", []),
    ("By Suite", ["suite"]),
    ("By Suite+DataType+Mode", ["suite", "data_type", "mode"]),
]

FAIL_LABELS = {"new_failed", "new_dropped"}
PASS_LABELS = {"new_passed", "new_improved"}
