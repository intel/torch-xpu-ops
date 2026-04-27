
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cute/tensor.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

#include "symm.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

struct Options {
	bool help = false;
	bool error = false;
	int m = 8192;
	int n = 4096;
	int k = 3584;
	int l = 1;
	int iterations = 20;
	int debug_log = 1;
	int verify = 0;
	float alpha = 1.0f;
	float beta = 0.0f;

	void parse(int argc, char **args) {
		std::vector<char const*> cargs(argc);
		for (int i = 0; i < argc; ++i) {
			cargs[i] = args[i];
		}
		cutlass::CommandLine cmd(argc, cargs.data());
		if (cmd.check_cmd_line_flag("help")) {
			help = true;
			return;
		}
		cmd.get_cmd_line_argument("m", m, 8192);
		cmd.get_cmd_line_argument("n", n, 4096);
		cmd.get_cmd_line_argument("k", k, 3584);
		cmd.get_cmd_line_argument("l", l, 1);
		cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
		cmd.get_cmd_line_argument("beta", beta, 0.0f);
		cmd.get_cmd_line_argument("iterations", iterations, 20);
		cmd.get_cmd_line_argument("debug_log", debug_log, 1);
		cmd.get_cmd_line_argument("verify", verify, 0);
	}

	std::ostream& print_usage(std::ostream& out) const {
		out << "BMG GEMM Reduce-Scatter Example\n\n"
			<< "  --m=<int>         M extent (global, must be divisible by TP)\n"
			<< "  --n=<int>         N extent\n"
			<< "  --k=<int>         K extent\n"
			<< "  --l=<int>         batch count\n"
			<< "  --iterations=<int>\n"
			<< "  --debug_log=<int> 0/1 progress logs (default 1)\n"
			<< "  --verify=<int>    0/1 run accuracy verification (default 0)\n\n";
		return out;
	}
};

template <class Gemm>
struct ExampleRunner {
	using StrideA = typename Gemm::GemmKernel::StrideA;
	using StrideB = typename Gemm::GemmKernel::StrideB;
	using StrideC = typename Gemm::GemmKernel::StrideC;
	using StrideD = typename Gemm::GemmKernel::StrideD;

	using ElementA = typename Gemm::ElementA;
	using ElementB = typename Gemm::ElementB;
	using ElementAccumulator = typename Gemm::ElementAccumulator;

	using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
	using ElementC = typename Gemm::ElementC;
	using ElementOutput = typename CollectiveEpilogue::ElementOutput;
	using ElementCompute = typename CollectiveEpilogue::ElementCompute;

	using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

	StrideA stride_A;
	StrideB stride_B;
	StrideC stride_C;
	StrideD stride_D;
	ProblemShapeType shard_problem_{};
	StrideA shard_stride_A{};
	StrideC shard_stride_C{};
	StrideD shard_stride_D{};
	int local_rows_ = 0;
	std::unique_ptr<sycl::queue> current_q_;
	std::unique_ptr<sycl::queue> tmp_q_;
	std::unique_ptr<SymmMemory> symm_;
	Gemm gemm_op_;
	bool gemm_initialized_ = false;

	size_t chunk_elements_ = 0;
	size_t total_elements_ = 0;

	void initialize(
			ElementA* block_A,
			ElementB* block_B,
			ElementOutput* block_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::device const& device,
			int rank,
			int world_size) {
		auto log_init = [&](char const* msg) {
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] [init] " << msg << std::endl;
			}
		};

		log_init("initialize begin");
		int local_rows = options.m / world_size;

		if (!current_q_) {
			log_init("creating current_q");
			auto ctx = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
			log_init("current_q created");
		}

		auto ctx = current_q_->get_context();
		if (!tmp_q_) {
			log_init("creating tmp_q");
			tmp_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
			log_init("tmp_q created");
		}

		if (!symm_) {
			log_init("creating SymmMemory");
			// world_size slots of (local_rows * N * L) bf16 elements = M * N * L total
			size_t rs_data_elems = static_cast<size_t>(options.m) * options.n * options.l;
			symm_ = std::make_unique<SymmMemory>(options.m, options.n, options.k, rank, world_size, *current_q_, 8,
					rs_data_elems);
			log_init("SymmMemory created");
		}

		stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
		stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
		stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
		stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

		if (options.debug_log) {
			printf("[rank %d] GEMM shard per call: M=%d N=%d K=%d L=%d\n",
					rank, local_rows, options.n, options.k, options.l);
		}

		chunk_elements_ = static_cast<size_t>(local_rows) * options.n * options.l;
		total_elements_ = static_cast<size_t>(options.m) * options.n * options.l;

		// Pre-compute shard strides (same for every run_shard_gemm call)
		local_rows_ = local_rows;
		shard_problem_ = ProblemShapeType{local_rows, options.n, options.k, options.l};
		shard_stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_rows, options.k, options.l));
		shard_stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_rows, options.n, options.l));
		shard_stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_rows, options.n, options.l));

		log_init("before block_A memset");
		current_q_->memset(block_A, 0, static_cast<size_t>(options.m) * options.k * options.l * sizeof(ElementA)).wait();
		log_init("before block_B memset");
		current_q_->memset(block_B, 0, static_cast<size_t>(options.n) * options.k * options.l * sizeof(ElementB)).wait();
		log_init("before block_C memset");
		current_q_->memset(block_C, 0, static_cast<size_t>(options.m) * options.n * options.l * sizeof(ElementOutput)).wait();

		// Initialize GEMM operator once with template args
		if (!gemm_initialized_) {
			typename Gemm::GemmKernel::Arguments template_args{
				cutlass::gemm::GemmUniversalMode::kGemm,
				shard_problem_,
				{block_A, shard_stride_A, block_B, stride_B},
				{{options.alpha, options.beta}, static_cast<ElementC const*>(nullptr), shard_stride_C, block_C, shard_stride_D},
				hw_info};

			log_init("before can_implement");
			auto st = gemm_op_.can_implement(template_args);
			if (st != cutlass::Status::kSuccess) {
				throw std::runtime_error("GEMM cannot implement shard args.");
			}
			log_init("before gemm initialize");
			st = gemm_op_.initialize(template_args, nullptr, current_q_.get());
			if (st != cutlass::Status::kSuccess) {
				throw std::runtime_error("GEMM initialize failed.");
			}
			gemm_initialized_ = true;
			log_init("GEMM operator initialized");
		}
		log_init("initialize end");
	}

	cutlass::Status run_shard_gemm(
			sycl::queue& queue,
			ElementA* a_ptr,
			ElementB* b_ptr,
			ElementOutput* d_ptr,
			ElementCompute alpha,
			ElementCompute beta,
			cutlass::KernelHardwareInfo const& hw_info) {

		typename Gemm::GemmKernel::Arguments args{
			cutlass::gemm::GemmUniversalMode::kGemm,
			shard_problem_,
			{a_ptr, shard_stride_A, b_ptr, stride_B},
			{{alpha, beta}, static_cast<ElementC const*>(nullptr), shard_stride_C, d_ptr, shard_stride_D},
			hw_info};

		if (!gemm_initialized_) {
			return cutlass::Status::kErrorInternal;
		}

		auto st = gemm_op_.update(args, nullptr);
		if (st != cutlass::Status::kSuccess) return st;
		return gemm_op_.run(&queue);
	}

	struct reduce_scatter {
		ExampleRunner& runner;

		sycl::event operator()(
				sycl::queue& q,
				ElementA* block_A,
				ElementB* block_B,
				ElementOutput* block_C,
				SymmMemory& symm,
				Options const& options,
				cutlass::KernelHardwareInfo const& hw_info,
				int rank,
				int world_size) const {
			int local_rows = options.m / world_size;
			size_t shard_c_elems = static_cast<size_t>(local_rows) * options.n * options.l;
			size_t shard_a_elems = static_cast<size_t>(local_rows) * options.k * options.l;
			size_t local_off = static_cast<size_t>(rank) * shard_c_elems;
			size_t shard_bytes = shard_c_elems * sizeof(ElementOutput);

			ElementOutput* local_p2p_ = reinterpret_cast<ElementOutput*>(symm.local_data_ptr_);
			auto& remote_p2p_ptrs_ = symm.remote_data_ptrs_;

			if (local_p2p_ == nullptr || remote_p2p_ptrs_.empty()) {
				throw std::runtime_error("IPC pointers are null.");
			}

			auto& current_q = *runner.current_q_;
			auto& tmp_q = *runner.tmp_q_;

			symm.barrier(0, current_q);
			
			// Phase 1+2: run local GEMM for each shard and push to destination peer.
			for (int step = 1; step < world_size; ++step) {
				int dst_rank = (rank + step) % world_size;
				int channel = step % 2;
				auto& queue = (channel == 0) ? current_q : tmp_q;
				// printf("[rank %d] step=%d dst_rank=%d channel=%d queue=%p\n",
				// 	rank, step, dst_rank, channel, static_cast<void*>(&queue));

				ElementOutput* local_shard_out = local_p2p_ + dst_rank * shard_c_elems;
				auto st = runner.run_shard_gemm(queue,
					block_A + static_cast<size_t>(dst_rank) * shard_a_elems,
					block_B,
					local_shard_out,
					options.alpha, options.beta, hw_info);
				if (st != cutlass::Status::kSuccess)
					throw std::runtime_error("run_shard_gemm (remote shard) failed.");
				ElementOutput* remote_dst = reinterpret_cast<ElementOutput*>(remote_p2p_ptrs_[dst_rank]) + rank * shard_c_elems;
				queue.memcpy(remote_dst, local_shard_out, shard_bytes);
			}

			auto st_local = runner.run_shard_gemm(current_q,
				block_A + static_cast<size_t>(rank) * shard_a_elems,
				block_B,
				local_p2p_ + local_off,
				options.alpha, options.beta, hw_info);
			if (st_local != cutlass::Status::kSuccess)
				throw std::runtime_error("run_shard_gemm (local shard) failed.");

			current_q.ext_oneapi_submit_barrier({tmp_q.ext_oneapi_submit_barrier()});
			symm.barrier(0, current_q);

			// Phase 3: local reduction — sycl::vec 向量化版本
			// 参考 bitsandbytes xpu_kernels.cpp 的 reinterpret_cast<sycl::vec> 模式，
			// 每个 work-item 用 sycl::vec<uint16_t, NUM_PER_TH> 做向量化加载/存储，
			// 边界内走 vec 路径，边界外逐元素处理。
			constexpr int NUM_PER_TH = 8;   // bf16 * 8 = 16B（非常适合 Xe）
			const int WG_SIZE = static_cast<int>(
				current_q.get_device().get_info<sycl::info::device::max_work_group_size>()) / NUM_PER_TH;

			const int64_t n_elems = static_cast<int64_t>(shard_c_elems);
			const int64_t vec_elems = n_elems / NUM_PER_TH;

			const int64_t TILE_SIZE = WG_SIZE * NUM_PER_TH;
			const int64_t n_groups = (n_elems + TILE_SIZE - 1) / TILE_SIZE;
			
			using SyclBF16 = sycl::ext::oneapi::bfloat16;

			ElementOutput* local_buffer_0 = local_p2p_;
			ElementOutput* local_buffer_1 = local_p2p_ + shard_c_elems;
			ElementOutput* local_buffer_2 = local_p2p_ + 2 * shard_c_elems;
			ElementOutput* local_buffer_3 = local_p2p_ + 3 * shard_c_elems;

			// vector pointer（一次性转换，避免 kernel 内重复 cast）
			auto out_vec = reinterpret_cast<sycl::vec<SyclBF16, NUM_PER_TH>*>(block_C + local_off);

			auto buf0 = reinterpret_cast<const sycl::vec<SyclBF16, NUM_PER_TH>*>(local_buffer_0);
			auto buf1 = reinterpret_cast<const sycl::vec<SyclBF16, NUM_PER_TH>*>(local_buffer_1);
			auto buf2 = reinterpret_cast<const sycl::vec<SyclBF16, NUM_PER_TH>*>(local_buffer_2);
			auto buf3 = reinterpret_cast<const sycl::vec<SyclBF16, NUM_PER_TH>*>(local_buffer_3);

			return current_q.submit([&](sycl::handler& h) {
				h.parallel_for<class local_reduction_vec_kernel>(
					sycl::nd_range<1>(
						sycl::range<1>(n_groups * WG_SIZE),
						sycl::range<1>(WG_SIZE)),
					[=](sycl::nd_item<1> item)
					[[sycl::reqd_sub_group_size(16)]] {
						const int64_t base = item.get_group(0) * TILE_SIZE;
						const int64_t offset = item.get_local_id(0) * NUM_PER_TH;
						const int64_t global_idx = base + offset;

						// vector index
						const int64_t vec_idx = global_idx / NUM_PER_TH;

						// boundary check（vector 级别）
						if (vec_idx < vec_elems) {
							sycl::vec<SyclBF16, NUM_PER_TH> sum = buf0[vec_idx];
							sum += buf1[vec_idx];
							sum += buf2[vec_idx];
							sum += buf3[vec_idx];
							out_vec[vec_idx] = sum;
						}
					});
			});
		}
	};

	sycl::event run_iteration(
			sycl::queue& q,
			ElementA* block_A,
			ElementB* block_B,
			ElementOutput* block_C,
			SymmMemory& symm,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			int rank,
			int world_size) {

		reduce_scatter op{*this};
		return op(q, block_A, block_B, block_C, symm, options, hw_info, rank, world_size);
	}

	bool verify(
			ElementA* block_A,
			ElementB* block_B,
			ElementOutput* block_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			int rank,
			int world_size) {
		auto& q = *current_q_;
		int local_rows = options.m / world_size;
		size_t shard_c_elems = static_cast<size_t>(local_rows) * options.n * options.l;
		size_t shard_a_elems = static_cast<size_t>(local_rows) * options.k * options.l;
		size_t full_c_elems = static_cast<size_t>(options.m) * options.n * options.l;
		size_t a_elems = static_cast<size_t>(options.m) * options.k * options.l;
		size_t b_elems = static_cast<size_t>(options.n) * options.k * options.l;

		// Fill A and B with non-zero values for meaningful verification
		float a_val = 1.0f;
		float b_val = 1.0f;
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(a_elems), [=](sycl::id<1> i) {
				block_A[i] = static_cast<ElementA>(a_val);
			});
		});
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(b_elems), [=](sycl::id<1> i) {
				block_B[i] = static_cast<ElementB>(b_val);
			});
		});
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Step 1: Full GEMM — run world_size shard GEMMs to produce D_full[M x N]
		ElementOutput* d_full = sycl::malloc_device<ElementOutput>(full_c_elems, q);
		q.memset(d_full, 0, full_c_elems * sizeof(ElementOutput)).wait();

		for (int s = 0; s < world_size; ++s) {
			auto st = run_shard_gemm(q,
				block_A + static_cast<size_t>(s) * shard_a_elems,
				block_B,
				d_full + static_cast<size_t>(s) * shard_c_elems,
				options.alpha, options.beta, hw_info);
			if (st != cutlass::Status::kSuccess) {
				printf("[rank %d] verify: full GEMM shard %d failed\n", rank, s);
				sycl::free(d_full, q);
				return false;
			}
		}
		q.wait();

		// Step 2: Copy full GEMM result to host and convert to float for MPI
		std::vector<ElementOutput> host_full_raw(full_c_elems);
		q.memcpy(host_full_raw.data(), d_full, full_c_elems * sizeof(ElementOutput)).wait();
		sycl::free(d_full, q);

		std::vector<float> host_full_f32(full_c_elems);
		for (size_t i = 0; i < full_c_elems; ++i) {
			host_full_f32[i] = static_cast<float>(host_full_raw[i]);
		}

		// Step 3: MPI_Reduce_scatter to get reference for this rank's shard (in float)
		std::vector<float> host_ref(shard_c_elems);
		std::vector<int> recvcounts(world_size, static_cast<int>(shard_c_elems));
		MPI_Reduce_scatter(host_full_f32.data(), host_ref.data(), recvcounts.data(),
		                   MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		// Step 4: Run one reduce_scatter iteration on GPU
		q.memset(block_C, 0, full_c_elems * sizeof(ElementOutput)).wait();
		MPI_Barrier(MPI_COMM_WORLD);
		run_iteration(q, block_A, block_B, block_C, *symm_, options, hw_info, rank, world_size);
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Step 5: Copy GPU result (this rank's shard from block_C) to host
		std::vector<ElementOutput> host_result(shard_c_elems);
		size_t local_off = static_cast<size_t>(rank) * shard_c_elems;
		q.memcpy(host_result.data(), block_C + local_off, shard_c_elems * sizeof(ElementOutput)).wait();

		// Step 6: Compare
		double max_abs_diff = 0.0;
		double max_rel_diff = 0.0;
		size_t mismatch_count = 0;
		for (size_t i = 0; i < shard_c_elems; ++i) {
			double ref = static_cast<double>(host_ref[i]);
			double val = static_cast<double>(host_result[i]);
			double diff = std::abs(ref - val);
			double rel = (std::abs(ref) > 1e-6) ? diff / std::abs(ref) : diff;
			max_abs_diff = std::max(max_abs_diff, diff);
			max_rel_diff = std::max(max_rel_diff, rel);
			if (rel > 1e-2) mismatch_count++;
		}

		bool passed = (mismatch_count == 0);
		printf("[rank %d] Verification %s: max_abs=%.6e, max_rel=%.6e, mismatches=%zu/%zu (expected=%.3f)\n",
		       rank, passed ? "PASSED" : "FAILED",
		       max_abs_diff, max_rel_diff, mismatch_count, shard_c_elems,
		       static_cast<double>(host_ref[0]));
		return passed;
	}

	cutlass::Status run(
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::device const& device,
			int rank,
			int world_size) {

		if (options.m % world_size != 0) {
			throw std::runtime_error("GEMM+reduce-scatter requires M divisible by world_size.");
		}

		size_t a_elems = static_cast<size_t>(options.m) * options.k * options.l;
		size_t b_elems = static_cast<size_t>(options.n) * options.k * options.l;
		size_t c_elems = static_cast<size_t>(options.m) * options.n * options.l;
		if (a_elems == 0 || b_elems == 0 || c_elems == 0) {
			throw std::runtime_error("Invalid zero-sized allocation request. Check m/n/k/world_size values.");
		}
		if (!device.get_info<sycl::info::device::usm_device_allocations>()) {
			throw std::runtime_error("Selected SYCL device does not support USM device allocations.");
		}

		if (!current_q_) {
			auto queue_context = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					queue_context,
					device,
					sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
		}

		ElementA* block_A = sycl::malloc_device<ElementA>(a_elems, *current_q_);
		ElementB* block_B = sycl::malloc_device<ElementB>(b_elems, *current_q_);
		ElementOutput* block_C = sycl::malloc_device<ElementOutput>(c_elems, *current_q_);
		auto mb = [](size_t bytes) {
			return static_cast<double>(bytes) / (1024.0 * 1024.0);
		};
		printf("[rank %d] Allocated: A=%.2f MiB, B=%.2f MiB, C=%.2f MiB\n",
				rank,
				mb(a_elems * sizeof(ElementA)),
				mb(b_elems * sizeof(ElementB)),
				mb(c_elems * sizeof(ElementOutput)));
		if (block_A == nullptr || block_B == nullptr || block_C == nullptr) {
			throw std::runtime_error(
				"Device allocation failed: A=" + std::to_string(mb(a_elems * sizeof(ElementA))) +
				" MiB, B=" + std::to_string(mb(b_elems * sizeof(ElementB))) +
				" MiB, C=" + std::to_string(mb(c_elems * sizeof(ElementOutput))) + " MiB.");
		}

		auto cleanup = [&]() {
			if (block_A) sycl::free(block_A, *current_q_);
			if (block_B) sycl::free(block_B, *current_q_);
			if (block_C) sycl::free(block_C, *current_q_);
		};

		initialize(block_A, block_B, block_C, options, hw_info, device, rank, world_size);
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "[rank " << rank << "] initialization complete" << std::endl;

		// verification
		if (options.verify != 0 && !verify(block_A, block_B, block_C, options, hw_info, rank, world_size)) {
			std::cerr << "[rank " << rank << "] verification failed!" << std::endl;
			cleanup();
			return cutlass::Status::kErrorInternal;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// warmup
		constexpr int kWarmupIters = 10;
		std::cout << "[rank " << rank << "] warmup start (" << kWarmupIters << " iters)" << std::endl;
		for (int iter = 0; iter < kWarmupIters; ++iter) {
			run_iteration(*current_q_, block_A, block_B, block_C, *symm_, options, hw_info, rank, world_size);
		}

		current_q_->wait();
		MPI_Barrier(MPI_COMM_WORLD); // ensure all ranks have finished warmup before starting benchmark iterations
		std::cout << "[rank " << rank << "] warmup done" << std::endl;

		// benchmark
		sycl::event ev_before;
		auto benchmark_start = std::chrono::high_resolution_clock::now();
		for (int iter = 0; iter < options.iterations; ++iter) {
			if (iter == 9) {
				ev_before = run_iteration(*current_q_, block_A, block_B, block_C, *symm_, options, hw_info, rank, world_size);
			} else {
				run_iteration(*current_q_, block_A, block_B, block_C, *symm_, options, hw_info, rank, world_size);
			}
		}
		auto benchmark_stop = std::chrono::high_resolution_clock::now();
		auto ev_after = current_q_->ext_oneapi_submit_barrier();
		current_q_->wait();

		MPI_Barrier(MPI_COMM_WORLD);

		double total_ms = std::chrono::duration<double, std::milli>(benchmark_stop - benchmark_start).count();
		auto dev_start_ns = ev_before.get_profiling_info<sycl::info::event_profiling::command_end>();
		auto dev_end_ns   = ev_after.get_profiling_info<sycl::info::event_profiling::command_start>();
		double total_device_ms = static_cast<double>(dev_end_ns - dev_start_ns) / 1e6;

		if (true) {
			double avg_ms = total_ms / options.iterations;
			double avg_device_ms = total_device_ms / (options.iterations - 10);
			int local_rows = options.m / world_size;
			double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
			std::cout << "[" << rank << "] Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
			          << 'x' << options.l << ", TP=" << world_size << std::endl;
			printf("[%d] GEMM shard per call: M=%d N=%d K=%d\n", rank, local_rows, options.n, options.k);
			printf("[%d] Pipelined GEMM + Reduce-Scatter (host):   [%4.3f]TFlop/s  (%6.4f)ms\n", rank, tflops / (avg_ms / 1000.0), avg_ms);
			printf("[%d] Pipelined GEMM + Reduce-Scatter (device): [%4.3f]TFlop/s  (%6.4f)ms\n", rank, tflops / (avg_device_ms / 1000.0), avg_device_ms);
		}

		cleanup();

		return cutlass::Status::kSuccess;
	}
};

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int world_size = 1;
	int rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	Options options;
	options.parse(argc, argv);

	if (options.help) {
		options.print_usage(std::cout) << std::endl;
		MPI_Finalize();
		return 0;
	}
	if (options.error) {
		MPI_Finalize();
		return -1;
	}

	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if (devices.empty()) {
		if (rank == 0) {
			std::cerr << "No GPU devices found" << std::endl;
		}
		MPI_Finalize();
		return 1;
	}
	if (static_cast<size_t>(rank) >= devices.size()) {
		std::cerr << "Rank " << rank << " requires GPU device[" << rank << "], but only "
						<< devices.size() << " devices are available" << std::endl;
		MPI_Finalize();
		return 1;
	}

	auto device = devices[rank];

	cutlass::KernelHardwareInfo hw_info;
	hw_info.device_id = rank;
	hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

	using ElementAccumulator = float;
	using ElementComputeEpilogue = float;
	using ElementInputA = bfloat16_t;
	using ElementInputB = bfloat16_t;
	using ElementOutput = bfloat16_t;

	using LayoutA = cutlass::layout::RowMajor;
	using LayoutB = cutlass::layout::RowMajor;
	using LayoutC = cutlass::layout::RowMajor;
	using LayoutD = cutlass::layout::RowMajor;

	using GmemTiledCopyA = void;
	using GmemTiledCopyB = void;
	using TileShape = Shape<_256, _256, _32>;
	using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<TileShape>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
	constexpr int PipelineStages = 2;
	using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
	using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
	using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
			ElementOutput,
			ElementComputeEpilogue,
			ElementAccumulator,
			ElementAccumulator,
			cutlass::FloatRoundStyle::round_to_nearest>;
	using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
			EpilogueDispatchPolicy,
			EpilogueOp,
			TileShape,
			decltype(tile_shape(TiledMma()))>;
	using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
			EpilogueDispatchPolicy,
			TileShape,
			void,
			ElementAccumulator,
			cutlass::gemm::TagToStrideC_t<LayoutC>,
			ElementOutput,
			cutlass::gemm::TagToStrideC_t<LayoutD>,
			FusionCallbacks,
			void,
			void>;
	using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
			GEMMDispatchPolicy,
			TileShape,
			ElementInputA,
			cutlass::gemm::TagToStrideA_t<LayoutA>,
			ElementInputB,
			cutlass::gemm::TagToStrideB_t<LayoutB>,
			TiledMma,
			GmemTiledCopyA,
			void,
			void,
			cute::identity,
			GmemTiledCopyB,
			void,
			void,
			cute::identity>;
	using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
			Shape<int, int, int, int>,
			CollectiveMainloop,
			CollectiveEpilogue>;
	using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

	ExampleRunner<Gemm> runner;
	try {
		CUTLASS_CHECK(runner.run(options, hw_info, device, rank, world_size));
	} catch (std::exception const& e) {
		std::cerr << "[rank " << rank << "] " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Finalize();
	return 0;
}
