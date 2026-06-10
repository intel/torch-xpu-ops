# Advanced Nightly CI Debugging

## AOT Inductor C++ Compile Errors

When a test fails with `CppCompileError`, read the generated `.wrapper.cpp` error message carefully.
The root cause is usually **codegen ordering** in `cpp_wrapper_cpu.py` — a function used before
its definition is emitted. Check `write_wrapper_decl()` and `generate_input_output_runtime_checks()`
ordering.

