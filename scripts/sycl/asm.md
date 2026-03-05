
IGC： https://github.com/intel/intel-graphics-compiler


VISA：https://github.com/intel/intel-graphics-compiler/tree/master/visa 

VISA documentation: https://github.com/intel/intel-graphics-compiler/tree/master/documentation/visa


## 如何 dump asm 指令？

### 方法 1：使用 `-fsycl-dump-spirv` 输出 SPIR-V

```bash
icpx -fsycl -fsycl-dump-spirv -c test.cpp -o test.o
```

这会生成 `.spv` 文件，然后用 `spirv-dis` 反汇编：

```bash
spirv-dis *.spv
```

### 方法 2：使用 IGC 环境变量输出 Gen ISA

```bash
# 编译时输出最终的 GPU 汇编
export IGC_ShaderDumpEnable=1
export IGC_DumpToCurrentDir=1
./your_program
```

会在当前目录生成 `.asm` 文件，里面包含 Gen ISA。

### 方法 3：使用 `-fsycl-device-only` + `-S`

```bash
icpx -fsycl -fsycl-device-only -S test.cpp -o test.ll
```

输出 LLVM IR，可以看到中间表示。

### 方法 4：使用 `ocloc` 离线编译查看

```bash
# 先生成 SPIR-V
icpx -fsycl -fsycl-dump-spirv -c test.cpp

# 用 ocloc 编译成 Gen ISA
ocloc compile -file *.spv -spirv_input -device pvc -options "-igc_opts 'PrintToConsole=1'"
```
