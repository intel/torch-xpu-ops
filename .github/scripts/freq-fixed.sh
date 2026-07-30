#!/bin/bash
# ==============================================================================
# Frequency Pinning Script
# Pin min frequency to max frequency for stable benchmarking results.
#
# Usage:
#   ./freq-fixed.sh status  - Show current frequency settings
#   ./freq-fixed.sh lock    - Pin min freq to max freq (requires sudo)
#   ./freq-fixed.sh unlock  - Revert to original min freq values (requires sudo)
# ==============================================================================

set -euo pipefail

# Configuration
BACKUP_DIR="/root/freq_backup"
LOG_PREFIX="[freq_fixed]"

# Color codes
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m' # No Color
else
    RED='' GREEN='' YELLOW='' NC=''
fi

# Logging helpers
info() { echo "$LOG_PREFIX [INFO]  $*"; }
warn() { echo "$LOG_PREFIX [WARN]  $*" >&2; }
error() { echo "$LOG_PREFIX [ERROR] $*" >&2; }
die() {
    error "$*"
    exit 1
}

# Colorize frequency value: green if min==max, red if min<max, yellow for N/A
color_freq() {
    local val="$1" min_val="$2" max_val="$3"
    if [[ "$val" == "N/A" || "$val" == "NOT FOUND" ]]; then
        printf "${YELLOW}%s${NC}" "$val"
    elif [[ "$min_val" == "$max_val" ]]; then
        printf "${GREEN}%s${NC}" "$val"
    else
        printf "${RED}%s${NC}" "$val"
    fi
}

# Discovery: locate frequency sysfs files
declare -a GT_MIN_FILES=()
declare -a MIN_FREQ_FILES=()

discover_freq_files() {
    info "Scanning /sys/ for frequency control files ..."

    # Intel GPU: gt_min_freq_mhz / gt_max_freq_mhz
    while IFS= read -r -d '' f; do
        GT_MIN_FILES+=("$f")
    done < <(find /sys/ -name "gt_min_freq_mhz" -print0 2>/dev/null)

    # CPU / other devices: min_freq / max_freq
    while IFS= read -r -d '' f; do
        MIN_FREQ_FILES+=("$f")
    done < <(find /sys/ -name "min_freq" -print0 2>/dev/null)

    local total=$((${#GT_MIN_FILES[@]} + ${#MIN_FREQ_FILES[@]}))
    if [[ $total -eq 0 ]]; then
        die "No frequency control files found under /sys/"
    fi
    info "Found ${#GT_MIN_FILES[@]} gt_min_freq_mhz file(s), ${#MIN_FREQ_FILES[@]} min_freq file(s)."
}

# Helper: get the corresponding max file for a given min file
get_max_file() {
    local min_file="$1"
    local max_file

    if [[ "$min_file" == *gt_min_freq_mhz* ]]; then
        max_file="${min_file/gt_min_freq_mhz/gt_max_freq_mhz}"
    else
        max_file="${min_file/min_freq/max_freq}"
    fi

    if [[ ! -f "$max_file" ]]; then
        warn "Max file not found for: $min_file (expected: $max_file)"
        return 1
    fi
    echo "$max_file"
}

# Helper: read a sysfs value safely
read_sysfs() {
    local file="$1"
    if [[ ! -r "$file" ]]; then
        warn "Cannot read: $file"
        return 1
    fi
    cat "$file" 2>/dev/null
}

# Helper: write a sysfs value with error checking
write_sysfs() {
    local file="$1"
    local value="$2"

    if [[ ! -w "$file" ]]; then
        if ! echo "$value" | sudo tee "$file" >/dev/null 2>&1; then
            error "Failed to write $value to $file"
            return 1
        fi
    else
        if ! echo "$value" >"$file" 2>/dev/null; then
            error "Failed to write $value to $file"
            return 1
        fi
    fi
    return 0
}

# Status: display current min/max values
show_status() {
    local found=0

    if [[ ${#GT_MIN_FILES[@]} -gt 0 ]]; then
        echo ""
        echo "=== Intel GPU Frequency (gt_min/max_freq_mhz) ==="
        printf "  %-6s  %10s  %10s  %s\n" "CARD" "MIN" "MAX" "PATH"
        printf "  %-6s  %10s  %10s  %s\n" "----" "---" "---" "----"
        # Sort files by card number
        local sorted_files
        sorted_files=$(for f in "${GT_MIN_FILES[@]}"; do
            local n
            n=$(dirname "$f" | grep -oP 'card\K[0-9]+' || echo 999)
            echo "$n $f"
        done | sort -n | awk '{print $2}')
        while IFS= read -r f; do
            [[ -n "$f" ]] || continue
            local card max_file min_val max_val
            card=$(dirname "$f" | grep -oP 'card[0-9]+' || basename "$(dirname "$f")")
            max_file="${f/gt_min_freq_mhz/gt_max_freq_mhz}"
            min_val=$(read_sysfs "$f") || min_val="N/A"
            if [[ -f "$max_file" ]]; then
                max_val=$(read_sysfs "$max_file") || max_val="N/A"
            else
                max_val="NOT FOUND"
            fi
            printf "  %-6s  %10b  %10b  %s\n" "$card" "$(color_freq "$min_val" "$min_val" "$max_val")" "$(color_freq "$max_val" "$min_val" "$max_val")" "$(dirname "$f")"
            ((found++)) || true
        done <<< "$sorted_files"
    fi

    if [[ ${#MIN_FREQ_FILES[@]} -gt 0 ]]; then
        echo ""
        echo "=== Device Frequency (min_freq/max_freq) ==="
        printf "  %-6s  %10s  %10s  %s\n" "CARD" "MIN" "MAX" "PATH"
        printf "  %-6s  %10s  %10s  %s\n" "----" "---" "---" "----"
        for f in "${MIN_FREQ_FILES[@]}"; do
            local card max_file min_val max_val
            card=$(dirname "$f" | grep -oP 'card[0-9]+|cpu[0-9]+' || basename "$(dirname "$f")")
            max_file="${f/min_freq/max_freq}"
            min_val=$(read_sysfs "$f") || min_val="N/A"
            if [[ -f "$max_file" ]]; then
                max_val=$(read_sysfs "$max_file") || max_val="N/A"
            else
                max_val="NOT FOUND"
            fi
            printf "  %-6s  %10b  %10b  %s\n" "$card" "$(color_freq "$min_val" "$min_val" "$max_val")" "$(color_freq "$max_val" "$min_val" "$max_val")" "$(dirname "$f")"
            ((found++)) || true
        done
    fi

    # CPU scaling governor
    echo ""
    echo "=== CPU Frequency ==="
    local gov
    gov=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq) || gov="N/A"
    echo "  Governor: $gov"

    echo ""
    if [[ $found -eq 0 ]]; then
        warn "No frequency files found."
    else
        info "$found frequency file(s) displayed."
    fi

    # Show backup status
    if [[ -d "$BACKUP_DIR" ]]; then
        info "Backup exists at $BACKUP_DIR (frequencies may be locked)."
    fi
}

# Backup: save original min values before modifying
backup_min_files() {
    if [[ -d "$BACKUP_DIR" ]]; then
        warn "Backup already exists at $BACKUP_DIR — skipping (use 'unlock' first to reset)."
        return 0
    fi

    mkdir -p "$BACKUP_DIR" || die "Failed to create backup directory: $BACKUP_DIR"
    info "Backing up original min freq values to $BACKUP_DIR ..."

    local count=0

    for f in "${GT_MIN_FILES[@]}"; do
        get_max_file "$f" >/dev/null || continue
        local val
        val=$(read_sysfs "$f") || continue
        local backup_name
        backup_name=$(echo "$f" | tr '/' '_')
        echo "$f $val" >"$BACKUP_DIR/$backup_name"
        ((count++)) || true
    done

    for f in "${MIN_FREQ_FILES[@]}"; do
        get_max_file "$f" >/dev/null || continue
        local val
        val=$(read_sysfs "$f") || continue
        local backup_name
        backup_name=$(echo "$f" | tr '/' '_')
        echo "$f $val" >"$BACKUP_DIR/$backup_name"
        ((count++)) || true
    done

    info "Backed up $count file(s)."
}

# CPU frequency governor: set to performance mode
lock_cpu_governor() {
    local scaling_governor
    scaling_governor=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq) || true

    if [[ -z "$scaling_governor" ]]; then
        warn "Cannot read CPU scaling governor."
        return 0
    fi

    if [[ "$scaling_governor" == "performance" ]]; then
        info "CPU governor already set to performance."
        return 0
    fi

    # Backup current governor
    echo "$scaling_governor" >"$BACKUP_DIR/cpu_governor"
    info "CPU governor is '${scaling_governor}', switching to performance ..."

    if ! command -v cpupower &>/dev/null; then
        info "Installing cpupower ..."
        if ! { sudo -E apt-get update -qq && sudo -E apt-get install -y -qq \
            linux-tools-common "linux-tools-$(uname -r)" "linux-cloud-tools-$(uname -r)"; }; then
            true
        fi
    fi
    if command -v cpupower &>/dev/null; then
        sudo cpupower set -b 0 || warn "cpupower set -b 0 failed"
        sudo cpupower frequency-set -g performance || warn "cpupower frequency-set failed"
    else
        warn "cpupower still not available after install attempt. Falling back to sysfs."
        for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "performance" | sudo tee "$gov_file" >/dev/null 2>&1 || true
        done
    fi

    # Drop caches
    sync
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" || true
    info "CPU governor set to performance, caches dropped."
}

# CPU frequency governor: restore original governor
unlock_cpu_governor() {
    local backup_file="$BACKUP_DIR/cpu_governor"
    if [[ ! -f "$backup_file" ]]; then
        info "No CPU governor backup found, skipping."
        return 0
    fi

    local orig_governor
    orig_governor=$(cat "$backup_file")
    local cur_governor
    cur_governor=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq) || cur_governor=""

    if [[ "$cur_governor" == "$orig_governor" ]]; then
        info "CPU governor already at original value '$orig_governor', skipping."
        return 0
    fi

    info "Restoring CPU governor to '$orig_governor' ..."
    if command -v cpupower &>/dev/null; then
        sudo cpupower frequency-set -g "$orig_governor" || warn "cpupower frequency-set -g $orig_governor failed"
    else
        for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo "$orig_governor" | sudo tee "$gov_file" >/dev/null 2>&1 || true
        done
    fi
    info "CPU governor restored to '$orig_governor'."
}

# Lock: set min freq = max freq
lock_freq() {
    info "Locking frequency: setting min freq = max freq ..."
    backup_min_files
    lock_cpu_governor

    local success=0 fail=0

    local skipped=0

    for f in "${GT_MIN_FILES[@]}"; do
        local max_file max_val min_val
        max_file=$(get_max_file "$f") || continue
        max_val=$(read_sysfs "$max_file") || continue
        min_val=$(read_sysfs "$f") || continue
        if [[ "$min_val" == "$max_val" ]]; then
            ((skipped++)) || true
            continue
        fi
        if write_sysfs "$f" "$max_val"; then
            info "  OK: $f <- $max_val"
            ((success++)) || true
        else
            ((fail++)) || true
        fi
    done

    for f in "${MIN_FREQ_FILES[@]}"; do
        local max_file max_val min_val
        max_file=$(get_max_file "$f") || continue
        max_val=$(read_sysfs "$max_file") || continue
        min_val=$(read_sysfs "$f") || continue
        if [[ "$min_val" == "$max_val" ]]; then
            ((skipped++)) || true
            continue
        fi
        if write_sysfs "$f" "$max_val"; then
            info "  OK: $f <- $max_val"
            ((success++)) || true
        else
            ((fail++)) || true
        fi
    done

    echo ""
    info "Lock complete: $success succeeded, $fail failed, $skipped already locked."
    [[ $fail -eq 0 ]] || warn "Some files could not be written. Check permissions."
}

# Unlock: restore original min freq values from backup
unlock_freq() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        info "No backup found at $BACKUP_DIR. Frequencies not locked, nothing to revert."
        return 0
    fi

    info "Reverting frequency: restoring original min freq values ..."
    unlock_cpu_governor

    local success=0 fail=0 skipped=0

    for backup in "$BACKUP_DIR"/*; do
        [[ -f "$backup" ]] || continue
        # Skip cpu_governor backup (handled by unlock_cpu_governor)
        [[ "$(basename "$backup")" == "cpu_governor" ]] && continue
        local file_path orig_val
        file_path=$(awk '{print $1}' "$backup")
        orig_val=$(awk '{print $2}' "$backup")

        if [[ -z "$file_path" || -z "$orig_val" ]]; then
            warn "Malformed backup entry: $backup"
            ((fail++)) || true
            continue
        fi

        if [[ ! -f "$file_path" ]]; then
            warn "Target file no longer exists: $file_path"
            ((fail++)) || true
            continue
        fi

        local cur_val
        cur_val=$(read_sysfs "$file_path") || cur_val=""
        if [[ "$cur_val" == "$orig_val" ]]; then
            ((skipped++)) || true
            continue
        fi

        if write_sysfs "$file_path" "$orig_val"; then
            info "  OK: $file_path <- $orig_val"
            ((success++)) || true
        else
            ((fail++)) || true
        fi
    done

    echo ""
    info "Unlock complete: $success succeeded, $fail failed, $skipped already at original."
    if [[ $fail -eq 0 ]]; then
        rm -rf "$BACKUP_DIR"
        info "Backup removed."
    else
        warn "Some files could not be restored. Backup kept at $BACKUP_DIR."
    fi
}

# Main
discover_freq_files

case "${1:-status}" in
    lock)
        lock_freq
        ;;
    unlock)
        unlock_freq
        ;;
    status)
        show_status
        ;;
    -h | --help)
        echo "Usage: $0 {lock|unlock|status}"
        echo ""
        echo "  status   Show current min/max frequency values (default)"
        echo "  lock     Pin min frequency to max frequency"
        echo "  unlock   Revert min frequency to original values"
        exit 0
        ;;
    *)
        error "Unknown command: $1"
        echo "Usage: $0 {lock|unlock|status}"
        exit 1
        ;;
esac
