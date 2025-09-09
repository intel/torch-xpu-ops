#!/bin/bash
set -xe

# init test
test_mode="--$1"
export ZE_AFFINITY_MASK=$2
test_file="`realpath $3`"
rm -rf ${GITHUB_WORKSPACE}/logs
mkdir -p ${GITHUB_WORKSPACE}/logs
output_file="${GITHUB_WORKSPACE}/logs/tmp.csv"

cd pytorch

while read line
do

	# skip #
	if [ $(echo ${line} |grep -c '^ *# ') -ne 0 ];then
		continue
	fi

	# get test case
	category=`	echo $line |awk '{print $1}'`
	model=`		echo $line |awk '{print $2}'`
	suite=`		echo $line |awk '{print $1}' |sed -E 's/_(float|bfloat|amp_).*//'`
	scenario=`	echo $line |awk '{print $1}' |sed -E 's/.*_/--/'`
	dtype=`		echo $line |awk '{print $1}' |sed -E 's/.*(face|bench|models)_/--/;s/_(tra|inf).*//;s/amp_fp16/amp --amp-dtype float16/;s/amp_bf16/amp --amp-dtype bfloat16/'`

	# test
	numactl -C 0-11 -l python benchmarks/dynamo/${suite}.py ${scenario} ${dtype} -d xpu -n10 ${test_mode}  --only ${model} --cold-start-latency --backend=inductor --output=${output_file}

	# result
	sed -i "s/^xpu,/${category},xpu,/;s/^dev,/category,dev,/" ${output_file}

done < ${test_file}
