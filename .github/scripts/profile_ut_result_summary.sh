#!/bin/bash

# Unified Log Analysis Script
# Usage: ./result_check_for_profile_ut.sh <log_file> [analysis_type]

LOG_FILE=$1
ANALYSIS_TYPE=${2:-"all"}

# Check parameters
if [ $# -lt 1 ]; then
    echo "Usage: $0 <log_file> [analysis_type]"
    echo "Available analysis types:"
    echo "  all - Execute all analyses"
    echo "  1 - correlation_id_mixed"
    echo "  2 - reproducer_missing_gpu_kernel_time"
    echo "  3 - time_precision"
    echo "  4 - partial_runtime_ops"
    echo "  5 - triton_xpu_ops"
    echo "  6 - profiling_fp32_train_resnet50"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' does not exist"
    exit 1
fi

# Analysis functions
analyze_correlation_id_mixed() {
    echo "=== Analysis: correlation_id_mixed ==="
    echo "Check criteria:"
    echo "1. CPU time avg > 0 (all ops)"
    echo "2. aten:: ops: XPU time avg > 0"
    echo "3. runtime ops(begin with ur): XPU time avg = 0"
    echo "4. aten:: and corresponding ur ops Calls = 3"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 6;
            xpu_col = 10;
            calls_col = 11;
            total_errors = 0;
            aten_ops = 0;
            ur_ops = 0;
        }
        
        /^ *aten::/ {
            aten_ops++;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be > 0 (actual: " cpu_time ")";
            }
            if (xpu_time <= 0) {
                errors++;
                error_msg = error_msg " XPU time should be > 0 (actual: " xpu_time ")";
            }
            if (calls != 3) {
                errors++;
                error_msg = error_msg " Calls should be 3 (actual: " calls ")";
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ aten:: OP error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", error_msg, $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "✅ aten:: OP normal: %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        /^ *ur/ {
            ur_ops++;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be > 0 (actual: " cpu_time ")";
            }
            if (xpu_time != 0) {
                errors++;
                error_msg = error_msg " XPU time should be 0 (actual: " xpu_time ")";
            }
            if (calls != 3) {
                errors++;
                error_msg = error_msg " Calls should be 3 (actual: " calls ")";
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ ur OP error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", error_msg, $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "✅ ur OP normal: %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "aten:: OP count: %d\n", aten_ops;
            printf "ur OP count: %d\n", ur_ops;
            printf "Total errors: %d\n", total_errors;
            if (total_errors == 0 && (aten_ops > 0 || ur_ops > 0)) {
                printf "✅ All checks passed!\n";
            } else if (aten_ops == 0 && ur_ops == 0) {
                printf "⚠️  No aten:: or ur ops found in the log\n";
            } else {
                printf "❌ Found %d errors, please check the above issues\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

analyze_reproducer_missing_gpu_kernel_time() {
    echo "=== Analysis: reproducer_missing_gpu_kernel_time ==="
    echo "Check criteria:"
    echo "1. XPU time avg of aten::gather and backward ops (MaxUnpool2DBackward0) should be > 0"
    echo "2. CPU time avg should be > 0"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 6;
            xpu_col = 10;
            calls_col = 11;
            total_errors = 0;
            checked_ops = 0;
        }
        
        /^ *aten::gather/ || /MaxUnpool2DBackward0/ {
            checked_ops++;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be > 0 (actual: " cpu_time ")";
            }
            if (xpu_time <= 0) {
                errors++;
                error_msg = error_msg " XPU time should be > 0 (actual: " xpu_time ")";
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ OP error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", error_msg, $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "✅ OP normal: %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "Checked ops count: %d\n", checked_ops;
            printf "Total errors: %d\n", total_errors;
            if (total_errors == 0 && checked_ops > 0) {
                printf "✅ All checks passed! XPU time is > 0 for all aten::gather and MaxUnpool2DBackward0 ops\n";
            } else if (checked_ops == 0) {
                printf "⚠️  No aten::gather or MaxUnpool2DBackward0 ops found in the log\n";
            } else {
                printf "❌ Found %d errors, some ops have XPU time <= 0\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

analyze_time_precision() {
    echo "=== Analysis: time_precision_in_profile ==="
    echo "Check criteria:"
    echo "1. All ops Self CPU time should be > 0"
    echo "2. Print exception lines where Self CPU <= 0"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 3;
            xpu_col = 10;
            calls_col = 11;
            total_errors = 0;
            checked_ops = 0;
        }
        {
            # Skip empty lines
            if (NF == 0) next;
            
            checked_ops++;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            
            cpu_time = cpu_str + 0;
            
            if (cpu_time < 0) {
                total_errors++;
                printf "❌ CPU time error: %s\n", $0;
                printf "   Self CPU: %s | XPU time avg: %s | Calls: %s\n\n", $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "Total ops checked: %d\n", checked_ops;
            printf "Total errors: %d\n", total_errors;
            if (total_errors == 0 && checked_ops > 0) {
                printf "✅ All checks passed! All ops have Self CPU time > 0\n";
            } else if (checked_ops == 0) {
                printf "⚠️  No ops found in the log\n";
            } else {
                printf "❌ Found %d errors, some ops have Self CPU time <= 0\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

analyze_partial_runtime_ops() {
    echo "=== Analysis: partial_runtime_ops ==="
    echo "Check criteria:"
    echo "1. All runtime ops(begein with ur) should only be in [urEnqueueUSMMemcpy, urEnqueueKernelLaunch]"
    echo "2. Print all ur ops and mark invalid ones"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 6;
            xpu_col = 10;
            calls_col = 11;
            total_errors = 0;
            valid_ops = 0;
            invalid_ops = 0;
        }
        /^ *ur/ {
            op_name = $1;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            # Check if op is in allowed list
            if (op_name == "urEnqueueUSMMemcpy" || op_name == "urEnqueueKernelLaunch") {
                valid_ops++;
                printf "✅ Valid ur op: %s | CPU: %s | XPU: %s | Calls: %s\n", $0, $cpu_col, $xpu_col, $calls_col;
            } else {
                invalid_ops++;
                total_errors++;
                printf "❌ Invalid ur op: %s | CPU: %s | XPU: %s | Calls: %s\n", $0, $cpu_col, $xpu_col, $calls_col;
                printf "   Error: ur op should be urEnqueueUSMMemcpy or urEnqueueKernelLaunch\n\n";
            }
        }
        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "Valid ur ops (urEnqueueUSMMemcpy/urEnqueueKernelLaunch): %d\n", valid_ops;
            printf "Invalid ur ops: %d\n", invalid_ops;
            printf "Total errors: %d\n", total_errors;
            if (total_errors == 0 && (valid_ops > 0 || invalid_ops > 0)) {
                printf "✅ All checks passed! All ur ops are in allowed list\n";
            } else if (valid_ops == 0 && invalid_ops == 0) {
                printf "⚠️  No ur ops found in the log\n";
            } else {
                printf "❌ Found %d errors, some ur ops are not in allowed list\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

analyze_triton_xpu_ops() {
    echo "=== Analysis: triton_xpu_ops ==="
    echo "Check criteria:"
    echo "1. Triton kernels (begin with triton_) should have XPU time avg > 0"
    echo "2. Print all triton and ur ops"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 6;
            xpu_col = 10;
            calls_col = 11;
            total_errors = 0;
            triton_ops = 0;
            ur_ops = 0;
        }
        /^ *triton_/ {
            triton_ops++;
            op_name = $1;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be > 0 (actual: " cpu_time ")";
            }
            if (xpu_time <= 0) {
                errors++;
                error_msg = error_msg " XPU time should be > 0 (actual: " xpu_time ")";
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ Triton kernel error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", error_msg, $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "✅ Triton kernel normal: %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "Triton kernels checked: %d\n", triton_ops;
            printf "ur ops found: %d\n", ur_ops;
            printf "Total errors: %d\n", total_errors;
            if (total_errors == 0 && triton_ops > 0) {
                printf "✅ All checks passed! All Triton kernels have XPU time > 0\n";
            } else if (triton_ops == 0) {
                printf "⚠️  No Triton kernels (triton_*) found in the log\n";
            } else {
                printf "❌ Found %d errors, some Triton kernels have XPU time <= 0\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

analyze_profiling_fp32_train_resnet50() {
    echo "=== Analysis: profiling_fp32_train_resnet50 ==="
    echo "Check criteria:"
    echo "1. aten:: ops: XPU time avg should be >= 0 and not all zero"
    echo "2. ur ops: XPU time avg should all be 0"
    echo "----------------------------------------"
    
    awk '
        BEGIN { 
            cpu_col = 6; 
            xpu_col = 10; 
            calls_col = 11;
            total_errors = 0;
            aten_ops = 0;
            ur_ops = 0;
            aten_xpu_non_zero = 0;
            aten_xpu_zero = 0;
        }
        
        /^ *aten::/ {
            aten_ops++;
            op_name = $1;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be >= 0 (actual: " cpu_time ")";
            }
            if (xpu_time < 0) {
                errors++;
                error_msg = error_msg " XPU time should be >= 0 (actual: " xpu_time ")";
            }
            
            # Track XPU time status for aten ops
            if (xpu_time > 0) {
                aten_xpu_non_zero++;
            } else if (xpu_time == 0) {
                aten_xpu_zero++;
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ aten:: OP error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", error_msg, $cpu_col, $xpu_col, $calls_col;
            } else if (xpu_time > 0) {
                printf "✅ aten:: OP normal (XPU>0): %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "⚠️  aten:: OP warning (XPU=0): %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        /^ *ur/ {
            ur_ops++;
            op_name = $1;
            # Extract numeric values from the columns
            cpu_str = $cpu_col;
            xpu_str = $xpu_col;
            calls = $calls_col + 0;
            
            # Remove non-numeric characters except decimal point and minus sign
            gsub(/[^0-9.-]/, "", cpu_str);
            gsub(/[^0-9.-]/, "", xpu_str);
            
            cpu_time = cpu_str + 0;
            xpu_time = xpu_str + 0;
            
            errors = 0;
            error_msg = "";
            
            if (cpu_time < 0) {
                errors++;
                error_msg = error_msg " CPU time should be >= 0 (actual: " cpu_time ")";
            }
            if (xpu_time != 0) {
                errors++;
                error_msg = error_msg " XPU time should be 0 (actual: " xpu_time ")";
            }
            
            if (errors > 0) {
                total_errors++;
                printf "❌ ur OP error: %s\n", $0;
                printf "   Errors: %s | CPU: %s | XPU: %s | Calls: %s\n\n", $cpu_col, $xpu_col, $calls_col;
            } else {
                printf "✅ ur OP normal: %s | CPU: %s | XPU: %s | Calls: %s\n", $1, $cpu_col, $xpu_col, $calls_col;
            }
        }
        
        END {
            printf "\n=== Analysis Summary ===\n";
            printf "aten:: ops total: %d\n", aten_ops;
            printf "  - aten:: ops with XPU > 0: %d\n", aten_xpu_non_zero;
            printf "  - aten:: ops with XPU = 0: %d\n", aten_xpu_zero;
            printf "ur ops total: %d\n", ur_ops;
            printf "Total errors: %d\n", total_errors;
            
            # Additional check: not all aten ops have XPU time = 0
            if (aten_ops > 0 && aten_xpu_non_zero == 0) {
                total_errors++;
                printf "❌ Critical error: All aten:: ops have XPU time = 0 (should have at least some XPU time > 0)\n";
            }
            
            if (total_errors == 0 && (aten_ops > 0 || ur_ops > 0)) {
                printf "✅ All checks passed!\n";
                printf "   - aten:: ops: XPU time >= 0 and not all zero\n";
                printf "   - ur ops: XPU time all zero\n";
            } else if (aten_ops == 0 && ur_ops == 0) {
                printf "⚠️  No aten:: or ur ops found in the log\n";
            } else {
                printf "❌ Found %d errors, please check the above issues\n", total_errors;
            }
        }
    ' "$LOG_FILE"
    echo
}

# Execute corresponding analysis based on type
case $ANALYSIS_TYPE in
    all)
        analyze_correlation_id_mixed
        analyze_reproducer_missing_gpu_kernel_time
        analyze_time_precision
        analyze_partial_runtime_ops
        analyze_triton_xpu_ops
        analyze_profiling_fp32_train_resnet50
        ;;
    1|correlation_id_mixed) analyze_correlation_id_mixed ;;
    2|reproducer_missing_gpu_kernel_time) analyze_reproducer_missing_gpu_kernel_time ;;
    3|time_precision) analyze_time_precision ;;
    4|partial_runtime_ops) analyze_partial_runtime_ops ;;
    5|triton_xpu_ops) analyze_triton_xpu_ops ;;
    6|profiling_fp32_train_resnet50) analyze_profiling_fp32_train_resnet50 ;;
    *)
        echo "Error: Unknown analysis type '$ANALYSIS_TYPE'"
        exit 1
        ;;
esac

echo "Analysis completed!"