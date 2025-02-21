#!/bin/bash

# Post-op the linux wheel to change the .so rpath to make the wheel work with XPU runtime pypi packages
# Usage: rpath.sh /path/to/torch-xxxx.whl

pkg=$1
PATCHELF_BIN=patchelf

make_wheel_record() {
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
        # if the RECORD file, then
        echo "$FPATH,,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

XPU_RPATHS=(
    '$ORIGIN/../../../..'
)
XPU_RPATHS=$(IFS=: ; echo "${XPU_RPATHS[*]}")
export C_SO_RPATH=$XPU_RPATHS':$ORIGIN:$ORIGIN/lib'
export LIB_SO_RPATH=$XPU_RPATHS':$ORIGIN'
export FORCE_RPATH="--force-rpath"

rm -rf tmp
mkdir -p tmp
cd tmp
cp $pkg .

unzip -q $(basename $pkg)
rm -f $(basename $pkg)

if [[ -d torch ]]; then
    PREFIX=torch
else
    PREFIX=libtorch
fi

# set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
find $PREFIX -maxdepth 1 -type f -name "*.so*" | while read sofile; do
    echo "Setting rpath of $sofile to ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'}"
    $PATCHELF_BIN --set-rpath ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'} ${FORCE_RPATH:-} $sofile
    $PATCHELF_BIN --print-rpath $sofile
done

# set RPATH of lib/ files to $ORIGIN
find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do    echo "Setting rpath of $sofile to ${LIB_SO_RPATH:-'$ORIGIN'}"
    $PATCHELF_BIN --set-rpath ${LIB_SO_RPATH:-'$ORIGIN'} ${FORCE_RPATH:-} $sofile
    $PATCHELF_BIN --print-rpath $sofile
done

# regenerate the RECORD file with new hashes
record_file=$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g')
if [[ -e $record_file ]]; then
    echo "Generating new record file $record_file"
    : > "$record_file"
    # generate records for folders in wheel
    find * -type f | while read fname; do
        make_wheel_record "$fname" >>"$record_file"
    done
fi

# zip up the wheel back
zip -rq $(basename $pkg) $PREIX*
