#!/bin/bash

for (( SHARD_ID=0; SHARD_ID<8; SHARD_ID++ ))
do
    export ZE_AFFINITY_MASK=$SHARD_ID
    python test/run_test.py \
    --include inductor/test_torchinductor inductor/test_torchinductor_opinfo inductor/test_aot_inductor \
                inductor/test_codecache.py inductor/test_kernel_benchmark.py inductor/test_max_autotune.py \
                inductor/test_mkldnn_pattern_matcher.py inductor/test_triton_kernels.py \
                inductor/test_compile_subprocess.py inductor/test_compiled_optimizers.py inductor/test_compiled_autograd.py \
    --shard $SHARD_ID 8 \
    --verbose 2>$GITHUB_WORKSPACE/ut_log/torch_xpu/torch_xpu_test_error.$SHARD_ID.log > \
    $GITHUB_WORKSPACE/ut_log/torch_xpu/torch_xpu_test.$SHARD_ID.log &
done

wait
cp -r test/test-reports $GITHUB_WORKSPACE/ut_log/torch_xpu/
function read_dir(){
for file in `ls $1`
do
    if [ -d $1"/"$file ]
    then
    cp $1"/"$file"/"*.xml $GITHUB_WORKSPACE/ut_log/
    else
    echo "[Warning] $file has no xml"
    fi
done
}
read_dir test/test-reports/python-pytest