skip_dict = {
    # failed on MTL windows, skip first for Preci
    "test_xpu.py": (
        "test_lazy_init_xpu",
        "test_mem_get_info_xpu",
        "test_wrong_xpu_fork_xpu",
    ),
}
