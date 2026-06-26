# Build
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
     -lccl -lmpi \
     bench_ccl_allreduce_latency.cpp -o bench_ccl_allreduce_latency

# Run (4 GPUs)
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
mpirun -n 4 ./bench_ccl_allreduce_latency

# 自定义 size range
mpirun -n 4 ./bench_ccl_allreduce_latency --min 4 --max 14 --loop 200