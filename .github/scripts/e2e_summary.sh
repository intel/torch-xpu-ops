#!/bin/bash

results_dir="$1"
check_file="$(dirname "$0")/../ci_expected_accuracy/check_expected.py"

function get_model_result() {
    echo -e "\n<table><thead>
        <tr>
            <th rowspan=2> Suite </th><th rowspan=2> Model </th>
            <th colspan=5> Training </th><th colspan=5> Inference </th>
        </tr><tr>
            <th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>
            <th> float32 </th><th> bfloat16 </th><th> float16 </th><th> amp_bf16 </th><th> amp_fp16 </th>
        </tr>
    </thead><tbody>"
    suite_list=$(
        find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" |\
        sed "s/.*inductor_//;s/_[abf].*//" |sort |uniq
    )
    rm -rf /tmp/tmp-result.txt
    for suite in ${suite_list}
    do
        model_list=$(
            find "${results_dir}" -name "*.csv" |grep -E ".*${suite}.*_xpu_accuracy.csv" |\
            xargs cat |grep "^xpu," |cut -d, -f2 |sort |uniq
        )
        for model in ${model_list}
        do
            for dtype in float32 bfloat16 float16 amp_bf16 amp_fp16
            do
                for mode in training inference
                do
                    colorful=$(grep -w "${model}" "/tmp/tmp-${suite}-${mode}-${dtype}.txt" 2>&1 |awk 'BEGIN{
                        color = "black";
                        exit_label = 0;
                    }{
                        if ($0 ~/Real failed/){
                            color="red";
                            exit_label++;
                        }else if ($0 ~/Expected failed/){
                            color="blue";
                        }else if ($0 ~/Warning timeout/){
                            color="orange";
                        }else if ($0 ~/New models/){
                            color="blue";
                        }else if ($0 ~/Failed to passed/){
                            color="green";
                            exit_label++;
                        }
                    }END{print color, exit_label}')
                    echo "${colorful}" >> /tmp/tmp-result.txt
                    context=$(find "${results_dir}" -name "*.csv" |\
                        grep -E ".*${suite}_${dtype}_${mode}_xpu_accuracy.csv" |xargs grep ",${model}," |cut -d, -f4 |\
                        awk -v c="${colorful/ *}" '{if(c=="black") {print $0}else {printf("\\$\\${__color__{%s}%s}\\$\\$", c, $0)}}')
                    eval "export ${mode}_${dtype}=${context}"
                done
            done
            echo -e "<tr>
                    <td>${suite}</td>
                    <td>${model}</td>
                    <td>${training_float32}</td>
                    <td>${training_bfloat16}</td>
                    <td>${training_float16}</td>
                    <td>${training_amp_bf16}</td>
                    <td>${training_amp_fp16}</td>
                    <td>${inference_float32}</td>
                    <td>${inference_bfloat16}</td>
                    <td>${inference_float16}</td>
                    <td>${inference_amp_bf16}</td>
                    <td>${inference_amp_fp16}</td>
                </tr>" |sed '/__color__/{s/__color__/\\color/g;s/_/\\_/g}'
        done
    done
    echo -e "</tbody></table>\n"
}

# Accuracy
accuracy=$(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" -c)
if [ "${accuracy}" -gt 0 ];then
    printf "#### Note:
\$\${\\color{red}Red}\$\$: the failed cases which need look into
\$\${\\color{green}Green}\$\$: the new passed cases which need update reference
\$\${\\color{blue}Blue}\$\$: the expected failed or new enabled cases
\$\${\\color{orange}Orange}\$\$: the warning cases
Empty means the cases NOT run\n\n"
    echo "### Accuracy"
    printf "| Category | Total | Passed | Pass Rate | \$\${\\color{red}Failed}\$\$ | "
    printf "\$\${\\color{blue}Xfailed}\$\$ | \$\${\\color{orange}Timeout}\$\$ | "
    printf "\$\${\\color{green}New Passed}\$\$ | \$\${\\color{blue}New Enabled}\$\$ | Not Run |\n"
    printf "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    echo > /tmp/tmp-summary.txt
    echo > /tmp/tmp-details.txt
    for csv in $(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" |sort)
    do
        category="$(echo "${csv}" |sed 's/.*inductor_//;s/_xpu_accuracy.*//')"
        suite="$(echo "${csv}" |sed 's/.*inductor_//;s/_.*//;s/timm/timm_models/')"
        mode="$(echo "${csv}" |sed 's/_xpu_accuracy.*//;s/.*_//')"
        dtype="$(echo "${csv}" |sed -E 's/.*inductor_[a-z]*_//;s/models_//;s/_infer.*|_train.*//')"
        python "${check_file}" --suite "${suite}" --mode "${mode}" --dtype "${dtype}" --csv_file "${csv}" > "/tmp/tmp-${suite}-${mode}-${dtype}.txt"
        test_result="$(sed 's/, /,/g' "/tmp/tmp-${suite}-${mode}-${dtype}.txt" |awk '{
            if($0 ~/Total/){
                total = $3;
            }
            if($0 ~/Passed/){
                passed = $3;
            }
            if($0 ~/Pass rate/){
                pass_rate = $3;
            }
            if($0 ~/Real failed/){
                failed = $4;
                failed_models = $5;
            }
            if($0 ~/Expected failed/){
                xfail = $4;
                xfail_models = $5;
            }
            if($0 ~/timeout/){
                timeout = $4;
                timeout_models = $5;
            }
            if($0 ~/Failed to passed/){
                new_passed = $5;
                new_passed_models = $6;
            }
            if($0 ~/Not run/){
                not_run = $4;
                not_run_models = $5;
            }
            if($0 ~/New models/){
                new_enabled = $3;
                new_enabled_models = $4;
            }
        }END {
            printf(" %d | %d | %s | %d | %d | %d | %d | %d | %d\n",
                total, passed, pass_rate, failed, xfail, timeout, new_passed, new_enabled, not_run);
        }')"
        echo "| ${category} | ${test_result} |" >> /tmp/tmp-summary.txt
    done
    cat /tmp/tmp-summary.txt
    get_model_result
fi

# Performance
performance=$(find "${results_dir}" -name "*.csv" |grep -E "_xpu_performance.csv" -c)
if [ "${performance}" -gt 0 ];then
    echo "### Performance"
    echo "| Category | Total | \$\${\\color{green}Passed}\$\$ | Pass Rate | Speedup |"
    echo "| --- | --- | --- | --- | --- |"
    for csv in $(find "${results_dir}" -name "*.csv" |grep -E "_xpu_performance.csv" |sort)
    do
        category="$(echo "${csv}" |sed 's/.*inductor_//;s/_xpu_performance.*//')"
        test_result="$(awk -M -v PREC=1024 -F ',' 'BEGIN{
            total = 0;
            pass = 0;
            fail = 0;
            speedup = 1;
        }{
            if ($1 == "xpu") {
                total++;
                if ($4 > 0) {
                    pass++;
                    speedup *= $4;
                }else {
                    fail++;
                }
            }
        }END{
            printf("%d | %d | %.2f% | %.3f\n", total, pass, pass/total*100, speedup^(1/pass))
        }' "${csv}")"
        echo "| ${category} | ${test_result} |"
    done
    echo
fi
