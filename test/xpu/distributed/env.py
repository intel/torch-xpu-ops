"""Centralized environment-variable configuration for the MoE fusion modules.

Every environment variable read by symm_buffer.py, ring_collectives.py and
their perf/benchmark tests is declared here, together with its default value.
Import this module and use the accessors below instead of reading os.environ
directly, so the full set of knobs (and their defaults) lives in one place.

Naming convention:
    <VAR>_ENV      -> the environment-variable name (str)
    <VAR>_DEFAULT  -> the default value used when the variable is unset
    <getter>()     -> reads os.environ and applies the default/parsing rules

The getters read os.environ lazily on each call so that tests can set the
variables before importing the modules that consume them.
"""

import os

# ---------------------------------------------------------------------------
# Distributed bootstrap (RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT)
# ---------------------------------------------------------------------------
# Under mpirun the launcher exports PMI_RANK / PMI_SIZE; we mirror those into
# the RANK / WORLD_SIZE variables that torch.distributed expects and pin the
# rendezvous endpoint.
PMI_RANK_ENV = "PMI_RANK"
PMI_SIZE_ENV = "PMI_SIZE"
RANK_ENV = "RANK"
WORLD_SIZE_ENV = "WORLD_SIZE"
MASTER_ADDR_ENV = "MASTER_ADDR"
MASTER_PORT_ENV = "MASTER_PORT"

# Defaults for the rendezvous endpoint.  Individual callers may pass their own
# master_addr / master_port to keep the historical per-file port assignments.
MASTER_ADDR_DEFAULT = "127.0.0.1"
MASTER_PORT_DEFAULT = "29500"


def setup_distributed_env(
    master_addr=MASTER_ADDR_DEFAULT,
    master_port=MASTER_PORT_DEFAULT,
    overwrite=True,
):
    """Populate RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.

    RANK / WORLD_SIZE are derived from PMI_RANK / PMI_SIZE (default 0 / 1).

    When ``overwrite`` is True (the default, matching the perf tests) existing
    values are replaced; when False the values are only filled in if unset
    (matching the benchmark scripts that used os.environ.setdefault).
    """
    set_var = os.environ.__setitem__ if overwrite else os.environ.setdefault
    set_var(RANK_ENV, str(os.environ.get(PMI_RANK_ENV, 0)))
    set_var(WORLD_SIZE_ENV, str(os.environ.get(PMI_SIZE_ENV, 1)))
    set_var(MASTER_ADDR_ENV, str(master_addr))
    set_var(MASTER_PORT_ENV, str(master_port))


# ---------------------------------------------------------------------------
# symm_buffer.py
# ---------------------------------------------------------------------------
SYMM_BUFFER_DEBUG_ENV = "SYMM_BUFFER_DEBUG"
SYMM_BUFFER_DEBUG_DEFAULT = "0"

# Dispatch (allgather + permute) and combine (unpermute + reduce-scatter) are
# controlled independently.
#   FUSION_RING_DISPATCH=auto -> staged for ws<=4, ring for ws>4
#   FUSION_RING_DISPATCH=1    -> always use ring dispatch when available
#   FUSION_RING_DISPATCH=0    -> never use ring dispatch
FUSION_RING_DISPATCH_ENV = "FUSION_RING_DISPATCH"
FUSION_RING_DISPATCH_DEFAULT = "auto"

FUSION_RING_COMBINE_ENV = "FUSION_RING_COMBINE"
FUSION_RING_COMBINE_DEFAULT = "1"

# PUSH is faster than PULL at every measured scale (ws=4: ring dispatch
# ~0.78 ms PUSH vs ~0.88 ms PULL), so the default is PUSH.  The module still
# gates this on native-op availability (hasattr on torch.ops.symm_mem).
FUSION_RING_PUSH_ENV = "FUSION_RING_PUSH"
FUSION_RING_PUSH_DEFAULT = True


def symm_buffer_debug():
    """SYMM_BUFFER_DEBUG=1 enables detailed input/output logging."""
    return os.environ.get(SYMM_BUFFER_DEBUG_ENV, SYMM_BUFFER_DEBUG_DEFAULT) == "1"


def fusion_ring_dispatch():
    """Dispatch policy for allgather+permute: auto / 1 / 0."""
    return os.environ.get(
        FUSION_RING_DISPATCH_ENV, FUSION_RING_DISPATCH_DEFAULT
    ).lower()


def fusion_ring_combine():
    """Whether combine should use the ring kernel (default enabled)."""
    return os.environ.get(
        FUSION_RING_COMBINE_ENV, FUSION_RING_COMBINE_DEFAULT
    ) == "1"


def fusion_ring_push():
    """Whether the ring dispatch should use the PUSH kernel (default True).

    Unset -> FUSION_RING_PUSH_DEFAULT (PUSH); "1" -> PUSH; anything else -> PULL.
    The caller must still AND this with native-op availability.
    """
    value = os.environ.get(FUSION_RING_PUSH_ENV)
    if value is None:
        return FUSION_RING_PUSH_DEFAULT
    return value == "1"


# ---------------------------------------------------------------------------
# ring_collectives.py
# ---------------------------------------------------------------------------
RING_WG_AUTOTUNE_ENV = "RING_WG_AUTOTUNE"
RING_WG_AUTOTUNE_DEFAULT = "1"

RING_WG_TUNE_WARMUP_ENV = "RING_WG_TUNE_WARMUP"
RING_WG_TUNE_WARMUP_DEFAULT = 5

RING_WG_TUNE_ITERS_ENV = "RING_WG_TUNE_ITERS"
RING_WG_TUNE_ITERS_DEFAULT = 20

RING_WG_CACHE_PATH_ENV = "RING_WG_CACHE_PATH"
RING_WG_CACHE_PATH_DEFAULT = os.path.join(
    os.path.expanduser("~"), ".cache", "torch_xpu_ops", "ring_wg_tune.json"
)

# Same semantics as FUSION_RING_PUSH: PUSH is faster at every measured scale,
# so the default is PUSH (still gated on native-op availability by the caller).
RING_AGP_PUSH_ENV = "RING_AGP_PUSH"
RING_AGP_PUSH_DEFAULT = True


def ring_wg_autotune():
    """RING_WG_AUTOTUNE: disabled when set to 0/false/no, enabled otherwise."""
    return os.environ.get(RING_WG_AUTOTUNE_ENV, RING_WG_AUTOTUNE_DEFAULT).lower() not in (
        "0",
        "false",
        "no",
    )


def ring_wg_tune_warmup():
    return int(os.environ.get(RING_WG_TUNE_WARMUP_ENV, str(RING_WG_TUNE_WARMUP_DEFAULT)))


def ring_wg_tune_iters():
    return int(os.environ.get(RING_WG_TUNE_ITERS_ENV, str(RING_WG_TUNE_ITERS_DEFAULT)))


def ring_wg_cache_path():
    return os.environ.get(RING_WG_CACHE_PATH_ENV, RING_WG_CACHE_PATH_DEFAULT)


def ring_agp_push():
    """Whether ring_allgather_permute should use the PUSH kernel (default True).

    Unset -> RING_AGP_PUSH_DEFAULT (PUSH); "1" -> PUSH; anything else -> PULL.
    The caller must still AND this with native-op availability.
    """
    value = os.environ.get(RING_AGP_PUSH_ENV)
    if value is None:
        return RING_AGP_PUSH_DEFAULT
    return value == "1"


# ---------------------------------------------------------------------------
# perf / benchmark tests
# ---------------------------------------------------------------------------
TOKENS_PER_RANK_ENV = "TOKENS_PER_RANK"
TOKENS_PER_RANK_DEFAULT = 2048


def tokens_per_rank(default=TOKENS_PER_RANK_DEFAULT):
    return int(os.environ.get(TOKENS_PER_RANK_ENV, default))
