"""
Window specific skip list for unit tests
Using pytest -k filtering syntax
"""

skip_dict = {
    # Windows: Skip entire files using *.py:: pattern
    "test_decomp": [
        "test_decomp.py::",  # Skip entire file on Windows
    ],

    # Files where Windows only needs to skip specific tests (will merge with Linux defaults)
    # "test_linalg": [
    #     "test_cholesky_windows_bug",  # Only skip specific Windows issues
    #     "test_qr_windows_memory",     # Will be merged with Linux skip list
    # ],
    
    # New test groups only needed on Windows
    # "windows_specific_issues": [
    #     "test_dll_loading",
    #     "test_path_length",
    # ],
}