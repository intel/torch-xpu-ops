#!/bin/bash

results_dir="$1"

# Accuracy
accuracy=$(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" -c)
if [ "${accuracy}" -gt 0 ];then
    echo "### Accuracy"
    printf "| Category | Total | \$\${\\color{green}Passed}\$\$ | Pass Rate | \$\${\\color{red}Failed}\$\$ | "
    printf "\$\${\\color{blue}Xfailed}\$\$ | \$\${\\color{green}Timeout}\$\$ | New Passed | New Enabled | Not Run |\n"
    printf "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    echo > tmp-summary.txt
    echo > tmp-details.txt
    for csv in $(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" |sort)
    do
        category="$(echo "${csv}" |sed 's/.*inductor_//;s/_xpu_accuracy.*//')"
        suite="$(echo "${csv}" |sed 's/.*inductor_//;s/_.*//;s/timm/timm_models/')"
        mode="$(echo "${csv}" |sed 's/_xpu_accuracy.*//;s/.*_//')"
        dt="$(echo "${csv}" |sed -E 's/.*inductor_[a-z]*_//;s/models_//;s/_infer.*|_train.*//')"
        python .github/ci_expected_accuracy/check_expected.py \
            --suite "${suite}" --mode "${mode}" --dtype "${dt}" --csv_file "${csv}" > tmp-result.txt
        test_result="$(sed 's/, /,/g' tmp-result.txt |awk '{
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
        echo "| ${category} | ${test_result} |" >> tmp-summary.txt
        sed -i '
            s/Real failed models:/$${\\color{red}Real \\space failed \\space models}$$:/g;
            s/Expected failed models:/$${\\color{blue}Expected \\space failed \\space models}$$:/g;
            s/Warning timeout models:/$${\\color{orange}Warning \\space timeout \\space models}$$:/g;
            s/Failed to passed models:/$${\\color{green}Failed \\space to \\space passed \\space models}$$:/g;
        ' tmp-result.txt
        {
            echo "<table><thead><tr><th colspan=2>$(sed 's/=//g' tmp-result.txt |head -n 1)</th></tr></thead><tbody>"
            sed "1d" tmp-result.txt |awk -F: '{printf("<tr><td>%s</td><td>%s</td></tr>\n", $1, $2)}'
            echo "</tbody></table>"
        } >> tmp-details.txt
    done
    cat tmp-summary.txt
    grep -v "<td> 0 \[\]</td>" tmp-details.txt
    rm -rf tmp-*.txt
fi

# Performance
performance=$(find "${results_dir}" -name "*.csv" |grep -E "_xpu_performance.csv" -c)
if [ "${performance}" -gt 0 ];then
    echo "### Performance"
    echo "| Category | Passed | Total | Pass Rate | Speedup |"
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
