#! /bin/bash
# This script work op perf summary

results_dir="$1"
output_file="$2"
Get_backward=${3:-False}
cd "$results_dir" || exit

echo "case_name;datatype;op_name;shape;channels_last;dim;output_size;P;reduce;kernel_size;stride;replacement;num_samples;scale_factor;mode;shifts;affine;backward;time(us)" >> "$output_file"

function op_summary {
    while IFS= read -r line1 && IFS= read -r line2 <&3; do
        text=${line1}
        IFS=';' read -ra pairs <<< "$(echo "$text" | tr -d '\n' | tr -s ' ')"
        for pair in "${pairs[@]}"; do
            IFS=':' read -r key value <<< "$pair"
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            if [[ shape = "$key" ]] ; then
                shape=${value}
            fi
            if [[ datatype = "$key" ]] ; then
                datatype=${value}
            fi
            if [[ dim = "$key" ]] || [[ dims = "$key" ]] ; then
                dim=${value}
            fi
            if [[ output_size = "$key" ]] ; then
                output_size=${value}
            fi
            if [[ channels_last = "$key" ]] ; then
                channels_last=${value}
            fi
            if [[ backward = "$key" ]] ; then
                backward=${value}
            fi
            if [[ reduce = "$key" ]] ; then
                reduce=${value}
            fi
            if [[ kernel_size = "$key" ]] ; then
                kernel_size=${value}
            fi
            if [[ P = "$key" ]] ; then
                P=${value}
            fi
            if [[ stride = "$key" ]] ; then
                stride=${value}
            fi
            if [[ replacement = "$key" ]] ; then
                replacement=${value}
            fi
            if [[ num_samples = "$key" ]] ; then
                num_samples=${value}
            fi
            if [[ scale_factor = "$key" ]] ; then
                scale_factor=${value}
            fi
            if [[ mode = "$key" ]] ; then
                mode=${value}
            fi
            if [[ affine = "$key" ]] ; then
                affine=${value}
            fi
            if [[ shifts = "$key" ]] ; then
                shifts=${value}
            fi
        done
        number=""
        if [[ $line2 =~ ^([0-9.]+)([a-zA-Z]+)$ ]] ; then
            number="${BASH_REMATCH[1]}"
            unit="${BASH_REMATCH[2]}"
        fi
        # Align the time units
        if [[ $unit == "ms" ]] ;then
           number=$(echo "scale=3; $number * 1000" | bc)
        fi
        if [[ $Get_backward == "True" ]] && [[ $backward == "False" ]]; then
            echo "Only Forward"
        else
            echo "${i%.*};${datatype};${op_name};$shape;$channels_last;$dim;$output_size;$P;$reduce;$kernel_size;$stride;$replacement;$num_samples;$scale_factor;$mode;$shifts;$affine;$backward;$number" >> "$output_file"
        fi
    done < <(echo "$texts") 3< <(echo "$times")
}

filename=$(find -- *.log)

for i in $filename
do
    output_size=""
    P=""
    channels_last=""
    dim=""
    backward=""
    reduce=""
    kernel_size=""
    affine=""
    output_size=""
    stride=""
    replacement=""
    num_samples=""
    scale_factor=""
    mode=""
    shifts=""
    case_name="${i%.*}"
    op_name=$(echo "$case_name" | awk -F. '{print $NF}')
    if [[ $Get_backward == "False" ]] ; then
        if [[ $op_name =~ batch_norm ]] ; then
            op_name="aten::batch_norm"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        elif [[ $op_name =~ exponential ]] || [[ $op_name =~ geometric ]] || [[ $op_name =~ uniform ]] || [[ $op_name =~ random ]] || [[ $op_name =~ normal ]] || [[ $op_name =~ log_normal ]] || [[ $op_name =~ bernoulli ]] || [[ $op_name =~ cauchy ]] ;then
            op_name=$op_name"_"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        elif [[ $op_name == unique ]] ; then
            op_name="unique2"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        elif [[ $op_name == max_pool3d ]] || [[ $op_name == max_pool2d ]] ; then
            op_name=$op_name"_with_indices"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == softmax ]] ; then
            op_name="aten::softmax"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        elif [[ $op_name == group_norm ]] ; then
            op_name="aten::group_norm"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        else
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        fi
    else
        if [[ $op_name =~ batch_norm ]] ; then
            op_name="batch_norm_backward"
            times=$(grep -E "${op_name}" "${i}" | awk  '{print $10}')
        elif [[ $op_name == max_pool3d ]] || [[ $op_name == max_pool2d ]] ; then
            op_name=$op_name"_with_indices_backward"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == col2im ]] ; then
            op_name="Col2ImBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == im2col ]] ; then
            op_name="Im2ColBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == flip ]] ; then
            op_name="FlipBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == matmul ]] ; then
            op_name="MmBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == roll ]] ; then
            op_name="RollBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == softmax ]] ; then
            op_name=$op_name"_backward_data"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == remainder ]] ; then
            op_name="RemainderBackward0"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        elif [[ $op_name == l1_loss ]] ; then
            op_name="l1_loss"
        else
            op_name=$op_name"_backward"
            times=$(grep -E "${op_name} " "${i}" | awk  '{print $10}')
        fi
    fi

    texts=$(grep -E "shape :|shape:" "$i")
    number=""
    if [[ $op_name == l1_loss ]] && [[ $Get_backward == "True" ]] ; then
        op_name="AbsBackward0"
        times=$(grep -E "${op_name} " "${i}" | grep -v "autograd" | awk  '{print $10}' | head -n 6)
        texts=$(grep -E "shape :|shape:" "$i" | head -n 6)
        op_summary
        op_name="MeanBackward0"
        times=$(grep -E "${op_name} " "${i}" | grep -v "autograd" | awk  '{print $10}')
        texts=$(grep -E "shape :|shape:" "$i" | tail -n 6)
        op_summary
    else
        op_summary
    fi
done
