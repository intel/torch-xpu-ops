icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
     -lccl -lmpi \
     bench_ccl_allgather_latency_c_api.cpp -o bench_ccl_allgather_latency