#! /bin/bash
# This script work for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
DT=${2:-float32}            # float32 / float16 / amp (amp_bf16) / amp_fp16
MODE=${3:-inference}        # inference / training
SCENARIO=${4:-accuracy}     # accuracy / performance
DEVICE=${5:-xpu}            # xpu / cuda
CARD=${6:-0}                # 0 / 1 / 2 / 3 ...
SHAPE=${7:-static}          # static / dynamic
NUM_SHARDS=${8}             # num test shards
SHARD_ID=${9}               # shard id
MODEL_ONLY=${10}            # GoogleFnet / T5Small / ...

WORKSPACE=`pwd`
LOG_DIR=$WORKSPACE/inductor_log/${SUITE}/${DT}
mkdir -p ${LOG_DIR}
LOG_NAME=inductor_${SUITE}_${DT}_${MODE}_${DEVICE}_${SCENARIO}

Model_only_extra=""
if [[ -n "$MODEL_ONLY" ]]; then
    echo "Testing model ${MODEL_ONLY}"
    Model_only_extra="--only ${MODEL_ONLY}"
fi

Cur_Ver=`pip list | grep "^torch " | awk '{print $2}' | cut -d"+" -f 1`
if [ $(printf "${Cur_Ver}\n2.0.2"|sort|head -1) = "${Cur_Ver}" ]; then
    Mode_extra="";
else
    # For PT >= 2.1
    # Remove --freezing cause feature not ready
    Mode_extra="--inference ";
fi

if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

Real_DT=$DT
DT_extra=''
if [[ "$DT" == "amp_bf16" ]]; then
    Real_DT=amp
    DT_extra="--amp-dtype bfloat16 "
elif [[ "$DT" == "amp_fp16" ]]; then
    Real_DT=amp
    DT_extra="--amp-dtype float16 "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

partition_flags=""
if [[ -n "$NUM_SHARDS" && -n "$SHARD_ID" ]]; then
  partition_flags="--total-partitions $NUM_SHARDS --partition-id $SHARD_ID "
fi

ulimit -n 1048576
ZE_AFFINITY_MASK=${CARD} \
    python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${Real_DT} -d ${DEVICE} -n10 ${DT_extra} ${Mode_extra} \
    ${Shape_extra} ${partition_flags} ${Model_only_extra} --backend=inductor --cold-start-latency --timeout=10800 \
    --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}_card${CARD}.log
