# Advanced Nightly CI Debugging

## AOT Inductor C++ Compile Errors

When a test fails with `CppCompileError`, read the generated `.wrapper.cpp` error message carefully.
The root cause is usually **codegen ordering** in `cpp_wrapper_cpu.py` — a function used before
its definition is emitted. Check `write_wrapper_decl()` and `generate_input_output_runtime_checks()`
ordering.

## Editable Install Header Caveats

Editable installs resolve Python from source but C++ headers from the installed path (`torch/include/`).
After editing a C++ header, manually copy it to the installed include path.

After modifying any header under `torch/csrc/inductor/cpp_wrapper/`, delete the PCH cache or the
fix will be masked by stale precompiled headers:

```bash
rm -rf /tmp/torchinductor_$USER/precompiled_headers/
```
