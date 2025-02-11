#!/bin/bash

results_dir="$1"

# Accuracy
accuracy=$(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" -c)
if [ "${accuracy}" -gt 0 ];then
    echo "### Accuracy"
    echo "| Category | Passed | Total | Pass Rate |"
    echo "| --- | --- | --- | --- |"
    for csv in $(find "${results_dir}" -name "*.csv" |grep -E "_xpu_accuracy.csv" |sort)
    do
        category="$(echo "${csv}" |sed 's/.*inductor_//;s/_xpu_accuracy.*//')"
        suite="$(echo "${csv}" |sed 's/.*inductor_//;s/_.*//;s/timm/timm_models/')"
        mode="$(echo "${csv}" |sed 's/_xpu_accuracy.*//;s/.*_//')"
        dt="$(echo "${csv}" |sed -E 's/.*inductor_[a-z]*_//;s/models_//;s/_infer.*|_train.*//')"
        test_result="$(awk -F ',' 'BEGIN{
            total = 0;
            pass = 0;
            fail = 0;
        }{
            if ($1 == "xpu") {
                total++;
                if ($4 ~/pass/) {
                    pass++;
                }else {
                    fail++;
                }
            }
        }END{
            printf("%d | %d | %.2f%\n", total, pass, pass/total*100)
        }' "${csv}")"
        echo "| ${category} | ${test_result} |"
        python .github/ci_expected_accuracy/check_expected.py \
            --suite "${suite}" \
            --mode "${mode}" \
            --dtype "${dt}" \
            --csv_file "${csv}"
        exit
    done
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
fi
