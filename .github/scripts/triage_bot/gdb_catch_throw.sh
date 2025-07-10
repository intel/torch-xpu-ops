testfile=$1
testcase=$2
if [ -e ${testfile} ]; then
    PYTORCH_ENABLE_XPU_FALLBACK=1 PYTORCH_TEST_WITH_SLOW=1 gdb -batch -ex "catch throw" -ex "run" -ex "bt" --args python -m pytest -v $testfile -k $testcase 
else
    echo "Cannot find the $testfile!"
fi
