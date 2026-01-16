#!/bin/bash
# Enhanced Benchmark Utilities Library

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Logging functions with timestamps
log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S') INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S') SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S') WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S') ERROR]${NC} $*"
}

log_debug() {
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        echo -e "${CYAN}[$(date +'%H:%M:%S') DEBUG]${NC} $*"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate directory exists
validate_directory() {
    local dir="$1"
    local description="$2"

    if [[ ! -d "$dir" ]]; then
        log_error "$description directory does not exist: $dir"
        return 1
    fi
    log_debug "Validated directory: $dir"
    return 0
}

# Validate file exists
validate_file() {
    local file="$1"
    local description="$2"

    if [[ ! -f "$file" ]]; then
        log_error "$description file does not exist: $file"
        return 1
    fi
    log_debug "Validated file: $file"
    return 0
}

# Create directory with verbose output
create_directory() {
    local dir="$1"
    local description="$2"

    if mkdir -p "$dir"; then
        log_info "Created $description directory: $dir"
        return 0
    else
        log_error "Failed to create $description directory: $dir"
        return 1
    fi
}

# Get current timestamp
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# Get human readable duration
format_duration() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))

    if (( hours > 0 )); then
        printf "%dh %dm %ds" $hours $minutes $secs
    elif (( minutes > 0 )); then
        printf "%dm %ds" $minutes $secs
    else
        printf "%ds" $secs
    fi
}

# Calculate percentage change
calculate_percentage_change() {
    local old_value="$1"
    local new_value="$2"

    if [[ "$old_value" == "0" ]] || [[ "$old_value" == "0.0" ]]; then
        echo "N/A"
        return
    fi

    local change=$(echo "scale=2; (($new_value - $old_value) / $old_value) * 100" | bc)
    printf "%.2f%%" "$change"
}

# Parse CSV file and extract specific columns
extract_csv_columns() {
    local csv_file="$1"
    local columns="$2"  # Comma-separated column indices (1-based)
    local output_file="$3"

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    awk -F',' -v cols="$columns" '
    BEGIN {
        split(cols, col_arr, ",")
    }
    {
        output = ""
        for (i in col_arr) {
            col = col_arr[i]
            if (col <= NF) {
                output = output (output ? "," : "") $col
            }
        }
        print output
    }' "$csv_file" > "$output_file"

    log_debug "Extracted columns $columns from $csv_file to $output_file"
}

# Check if CSV has header
csv_has_header() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        return 1
    fi

    # Check if first line contains column names (no numbers)
    local first_line=$(head -1 "$csv_file")
    if [[ "$first_line" =~ ^[^0-9]*$ ]]; then
        return 0
    else
        return 1
    fi
}

# Get CSV column count
get_csv_column_count() {
    local csv_file="$1"

    if [[ ! -f "$csv_file" ]]; then
        echo "0"
        return
    fi

    head -1 "$csv_file" | awk -F',' '{print NF}'
}

# Validate CSV format
validate_csv_format() {
    local csv_file="$1"
    local expected_columns="$2"  # Expected column count

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    local actual_columns=$(get_csv_column_count "$csv_file")

    if [[ "$actual_columns" -lt "$expected_columns" ]]; then
        log_error "CSV file has $actual_columns columns, expected at least $expected_columns: $csv_file"
        return 1
    fi

    log_debug "CSV format validated: $csv_file ($actual_columns columns)"
    return 0
}

# Extract unique values from CSV column
extract_unique_values() {
    local csv_file="$1"
    local column_index="$2"
    local output_file="$3"

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    awk -F',' -v col="$column_index" '
    NR == 1 { next }  # Skip header
    {
        if ($col != "" && !seen[$col]++) {
            print $col
        }
    }' "$csv_file" | sort > "$output_file"

    local count=$(wc -l < "$output_file")
    log_debug "Extracted $count unique values from column $column_index of $csv_file"
}

# Filter CSV by column value
filter_csv_by_value() {
    local csv_file="$1"
    local column_index="$2"
    local filter_value="$3"
    local output_file="$4"

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return 1
    fi

    awk -F',' -v col="$column_index" -v value="$filter_value" '
    NR == 1 { print $0; next }
    $col == value { print $0 }
    ' "$csv_file" > "$output_file"

    local count=$(($(wc -l < "$output_file") - 1))
    log_debug "Filtered $csv_file by column $column_index=$filter_value: $count rows"
}

# Merge multiple CSV files
merge_csv_files() {
    local output_file="$1"
    shift
    local input_files=("$@")

    if [[ ${#input_files[@]} -eq 0 ]]; then
        log_error "No input files specified for merge"
        return 1
    fi

    # Check if all files exist
    for file in "${input_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Input file not found: $file"
            return 1
        fi
    done

    # Create header from first file
    head -1 "${input_files[0]}" > "$output_file"

    # Append data from all files (skip headers)
    for file in "${input_files[@]}"; do
        tail -n +2 "$file" >> "$output_file"
    done

    local total_rows=$(($(wc -l < "$output_file") - 1))
    log_info "Merged ${#input_files[@]} CSV files into $output_file ($total_rows total rows)"
}

# Calculate statistics from numeric CSV column
calculate_column_stats() {
    local csv_file="$1"
    local column_index="$2"

    if [[ ! -f "$csv_file" ]]; then
        log_error "CSV file not found: $csv_file"
        return
    fi

    awk -F',' -v col="$column_index" '
    BEGIN {
        sum = 0
        count = 0
        min = 999999999
        max = 0
        sumsq = 0
    }
    NR > 1 && $col != "" {
        value = $col + 0
        if (value != 0) {
            sum += value
            sumsq += value * value
            count++
            if (value < min) min = value
            if (value > max) max = value
        }
    }
    END {
        if (count > 0) {
            mean = sum / count
            variance = (sumsq / count) - (mean * mean)
            stddev = sqrt(variance)
            printf "Count: %d\n", count
            printf "Mean: %.4f\n", mean
            printf "Min: %.4f\n", min
            printf "Max: %.4f\n", max
            printf "StdDev: %.4f\n", stddev
            printf "Range: %.4f\n", max - min
        } else {
            print "No valid data found"
        }
    }' "$csv_file"
}

# Check if string contains substring
contains() {
    local string="$1"
    local substring="$2"

    if [[ "$string" == *"$substring"* ]]; then
        return 0
    else
        return 1
    fi
}

# Safe division with bc
safe_divide() {
    local numerator="$1"
    local denominator="$2"

    if [[ "$denominator" == "0" ]] || [[ "$denominator" == "0.0" ]]; then
        echo "0"
    else
        echo "scale=4; $numerator / $denominator" | bc
    fi
}

# Compare floating point numbers
float_compare() {
    local a="$1"
    local b="$2"
    local op="$3"

    case "$op" in
        lt) echo "$a < $b" | bc -l ;;
        le) echo "$a <= $b" | bc -l ;;
        gt) echo "$a > $b" | bc -l ;;
        ge) echo "$a >= $b" | bc -l ;;
        eq) echo "$a == $b" | bc -l ;;
        ne) echo "$a != $b" | bc -l ;;
        *) echo "0" ;;
    esac
}

# Get system information
get_system_info() {
    local output_file="$1"

    {
        echo "System Information"
        echo "=================="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "Kernel: $(uname -r)"
        echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '"' || uname -o)"
        echo "CPU: $(lscpu 2>/dev/null | grep "Model name" | cut -d: -f2 | sed 's/^[[:space:]]*//' || grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^[[:space:]]*//')"
        echo "CPU Cores: $(nproc)"
        echo "Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}')"

        if command_exists nvidia-smi; then
            echo ""
            echo "GPU Information:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 | \
            awk -F',' '{printf "  GPU: %s\n  Driver: %s\n  VRAM: %s\n", $1, $2, $3}'
        fi

        if command_exists clinfo; then
            echo ""
            echo "OpenCL Information:"
            clinfo 2>/dev/null | grep -E "Platform Name|Device Name" | head -4
        fi
    } > "$output_file"

    log_debug "System information saved to: $output_file"
}

# Check if running in CI environment
is_ci_environment() {
    [[ -n "${CI:-}" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]] || [[ -n "${GITLAB_CI:-}" ]] || [[ -n "${JENKINS_URL:-}" ]]
}

# Print banner
print_banner() {
    local message="$1"
    local color="${2:-BLUE}"

    local color_code
    case "$color" in
        RED) color_code="$RED" ;;
        GREEN) color_code="$GREEN" ;;
        YELLOW) color_code="$YELLOW" ;;
        BLUE) color_code="$BLUE" ;;
        MAGENTA) color_code="$MAGENTA" ;;
        CYAN) color_code="$CYAN" ;;
        *) color_code="$NC" ;;
    esac

    local length=${#message}
    local border=$(printf '%*s' "$((length + 4))" '' | tr ' ' '=')

    echo -e "${color_code}"
    echo "  $border"
    echo "  # $message #"
    echo "  $border"
    echo -e "${NC}"
}
