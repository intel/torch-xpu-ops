#!/bin/bash
set -e

WGET_RETRY_DELAY_SECONDS=10
MAX_RETRIES=${MAX_RETRIES:-3}
download_with_retry() {
    local url="$1"
    local filename
    filename=$(basename "$url")
    local attempt=1
    echo "Starting download: $filename from $url"

    while [ $attempt -le $MAX_RETRIES ]; do
        if wget -q --timeout=60 "$url"; then
            echo "Successfully downloaded $filename."
            return 0
        else
            echo "Failed to download $filename. Retrying in $WGET_RETRY_DELAY_SECONDS seconds..."
            attempt=$((attempt + 1))
            sleep $WGET_RETRY_DELAY_SECONDS
        fi
    done

    echo "Error: Failed to download $filename after $MAX_RETRIES attempts."
    return 1
}

DEPENDENCY_DIR=${GITHUB_WORKSPACE}/../dependencies
rm -rf ${DEPENDENCY_DIR} || true
mkdir -p ${DEPENDENCY_DIR}
cd ${DEPENDENCY_DIR} && pwd || exit 1
download_with_retry https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/DPCPP_JGS/latest/DPCPP_JGS-master.tgz
download_with_retry https://artifactory-kfs.habana-labs.com/artifactory/bin-generic-dev-local/PROFILING_TOOLS_JGS/latest/PROFILING_TOOLS_JGS-master.tgz
for f in *.tgz; do filename=$f; woext="${filename%%-*}"; echo $woext ; mkdir $woext; tar -xzf $f -C $woext ; rm $f ; cat $woext/version.txt ; done
