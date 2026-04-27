
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

#include "cutlass/util/command_line.h"
#include "cutlass/util/packed_stride.hpp"
#include "helper.h"
#include "symm.hpp"
#include "sycl_common.hpp"

using namespace cute;

struct Options {
	bool help = false;
	bool error = false;
	int m = 8192;
	int n = 1536;
	int k = 4096;
	int l = 1;
	int iterations = 20;
	int debug_log = 1;
	int gemm_only = 0;
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
		cmd.get_cmd_line_argument("n", n, 1536);
		cmd.get_cmd_line_argument("k", k, 4096);
		cmd.get_cmd_line_argument("l", l, 1);
		cmd.get_cmd_line_argument("alpha", alpha, 1.0f);
		cmd.get_cmd_line_argument("beta", beta, 0.0f);
		cmd.get_cmd_line_argument("iterations", iterations, 20);
		cmd.get_cmd_line_argument("debug_log", debug_log, 1);
		cmd.get_cmd_line_argument("gemm_only", gemm_only, 0);
		cmd.get_cmd_line_argument("verify", verify, 0);
	}

	std::ostream& print_usage(std::ostream& out) const {
		out << "BMG allgather + GEMM Example\n\n"
				<< "  --m=<int>        global M, must be divisible by TP\n"
				<< "  --n=<int>        N\n"
				<< "  --k=<int>        K\n"
				<< "  --l=<int>        batch count\n"
				<< "  --iterations=<int>\n"
				<< "  --debug_log=<int> 0/1 progress logs (default 1)\n"
				<< "  --gemm_only=<int> 0/1 run gemm_only path (default 0)\n"
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
	std::unique_ptr<sycl::queue> current_q_;
	std::unique_ptr<sycl::queue> tmp_q_;
	std::unique_ptr<SymmMemory> symm_;
	Gemm gemm_op_;
	bool gemm_initialized_ = false;
	uint64_t seed = 0;

	void initialize(ElementA* local_A,
					ElementB* B,
					ElementOutput* final_C,
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
		int local_m = options.m / world_size;
		int n = options.n;
		int k = options.k;

		if (!current_q_) {
			log_init("creating current_q");
			auto ctx = sycl::context(device);
			current_q_ = std::make_unique<sycl::queue>(
					ctx,
					device,
					sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
			log_init("current_q created");
		}

		if (!symm_) {
			log_init("creating SymmMemory");
			size_t allgather_data_elems = static_cast<size_t>(local_m) * k * world_size;
			symm_ = std::make_unique<SymmMemory>(local_m, n, k, rank, world_size, *current_q_, 8,
					allgather_data_elems);
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] SymmMemory initialized" << std::endl;
			}
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

		stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(local_m, k, 1));
		stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
		stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(local_m, n, 1));
		stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(local_m, n, 1));

		// gemm_only uses full M; allgather uses local_m per shard
		int gemm_m = (options.gemm_only != 0) ? options.m : local_m;
		shard_problem_ = ProblemShapeType{gemm_m, n, k, 1};
		shard_stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(gemm_m, k, 1));
		shard_stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(gemm_m, n, 1));
		shard_stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(gemm_m, n, 1));

		if (options.debug_log) {
			printf("[rank %d] GEMM problem per call: M=%d N=%d K=%d (gemm_only=%d)\n",
					rank, gemm_m, n, k, options.gemm_only);
		}

		log_init("Memory set A B C");
		current_q_->memset(local_A, 0, static_cast<size_t>(local_m) * k * sizeof(ElementA)).wait();
		current_q_->memset(B, 0, static_cast<size_t>(n) * k * sizeof(ElementB)).wait();
		current_q_->memset(final_C, 0, static_cast<size_t>(options.m) * n * sizeof(ElementOutput)).wait();
		log_init("Memory set A B C finished");

		ElementA* gathered_A = reinterpret_cast<ElementA*>(symm_->local_data_ptr_);
		if (gathered_A == nullptr) {
			throw std::runtime_error("symm local_data_ptr is null.");
		}

        // todo: gemm template stable ptr?
		typename Gemm::GemmKernel::Arguments template_args{
				cutlass::gemm::GemmUniversalMode::kGemm,
				shard_problem_,
				{gathered_A, shard_stride_A, B, stride_B},
				{{options.alpha, options.beta}, static_cast<ElementC const*>(nullptr), shard_stride_C, final_C, shard_stride_D},
				hw_info};

		if (!gemm_initialized_) {
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
			log_init("after gemm initialize");
			if (options.debug_log) {
				std::cout << "[rank " << rank << "] GEMM operator initialized" << std::endl;
			}
		}
		log_init("All initialize end");
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

		if (Gemm::get_workspace_size(args) != 0) {
			return cutlass::Status::kErrorInternal;
		}

		if (!gemm_initialized_) {
			return cutlass::Status::kErrorInternal;
		}

		auto st = gemm_op_.update(args, nullptr);
		if (st != cutlass::Status::kSuccess) {
			return st;
		}
		return gemm_op_.run(&queue);
	}
	struct allgather_gemm {
		ExampleRunner& runner;

		sycl::event operator()(
				sycl::queue& q,
				ElementA* local_A,
				ElementB* B,
				ElementOutput* final_C,
				SymmMemory& symm,
				Options const& options,
				cutlass::KernelHardwareInfo const& hw_info,
				sycl::context const&,
				sycl::device const&,
				int rank,
				int world_size) const {
			int local_m = options.m / world_size;
			size_t shard_a_elems = static_cast<size_t>(local_m) * options.k;
			size_t shard_c_elems = static_cast<size_t>(local_m) * options.n;
			size_t shard_a_bytes = shard_a_elems * sizeof(ElementA);
			ElementA* gathered_A = reinterpret_cast<ElementA*>(symm.local_data_ptr_);
			if (gathered_A == nullptr) {
				throw std::runtime_error("SymmMemory IPC pointers are null.");
			}

			auto& current_q = *runner.current_q_;
			auto& tmp_q = *runner.tmp_q_;

			// copy local to symm memory for allgather
			current_q.memcpy(gathered_A + static_cast<size_t>(rank) * shard_a_elems, local_A, shard_a_bytes);
			symm.barrier(0, current_q);

			// do local GEMM first without waiting for allgather to complete, to achieve better overlap between communication and computation
			auto local_st = runner.run_shard_gemm(
					current_q,
					gathered_A + static_cast<size_t>(rank) * shard_a_elems,
					B,
					final_C + static_cast<size_t>(rank) * shard_c_elems,
					options.alpha,
					options.beta,
					hw_info);
			if (local_st != cutlass::Status::kSuccess) {
				throw std::runtime_error("allgather_gemm local shard GEMM submission failed.");
			}

			for (int step = 1; step < world_size; ++step) {
				int remote_rank = (rank + step) % world_size;
				int channel = step % 2;
				auto& queue = (channel == 0) ? current_q : tmp_q;

				ElementA* remote_buf = reinterpret_cast<ElementA*>(symm.get_data_buffer(remote_rank));
				ElementA* remote_src = remote_buf + static_cast<size_t>(remote_rank) * shard_a_elems;
				ElementA* local_dst = gathered_A + static_cast<size_t>(remote_rank) * shard_a_elems;

				if (remote_src == nullptr || local_dst == nullptr) {
				    throw std::runtime_error("SymmMemory remote pointers are null.");
			    }

				queue.memcpy(local_dst, remote_src, shard_a_bytes); // copy from remote to local peer buffer

				auto st = runner.run_shard_gemm(
						queue,
						local_dst,
						B,
						final_C + static_cast<size_t>(remote_rank) * shard_c_elems,
						options.alpha,
						options.beta,
						hw_info);
				if (st != cutlass::Status::kSuccess) {
					throw std::runtime_error("allgather_gemm shard GEMM submission failed.");
				}
			}
			current_q.ext_oneapi_submit_barrier({tmp_q.ext_oneapi_submit_barrier()});
			auto event = symm.barrier(0, current_q);
			return event;
		}
	};

	struct gemm_only {
		ExampleRunner& runner;

		sycl::event operator()(
				sycl::queue& q,
				ElementA* full_A,
				ElementB* B,
				ElementOutput* final_C,
				Options const& options,
				cutlass::KernelHardwareInfo const& hw_info,
				sycl::context const&,
				sycl::device const&,
				int rank,
				int world_size) const {
			int local_m = options.m / world_size;
			size_t shard_a_elems = static_cast<size_t>(local_m) * options.k;
			size_t shard_c_elems = static_cast<size_t>(local_m) * options.n;
			if (full_A == nullptr || B == nullptr || final_C == nullptr) {
				throw std::runtime_error("gemm_only input pointer is null.");
			}

			auto& current_q = *runner.current_q_;
			auto st = runner.run_shard_gemm(
						current_q,
						full_A,
						B,
						final_C,
						options.alpha,
						options.beta,
						hw_info);
						
			if (st != cutlass::Status::kSuccess) {
				throw std::runtime_error("gemm_only shard GEMM submission failed.");
			}
			return current_q.ext_oneapi_submit_barrier();
		}
	};

	sycl::event run_iteration(
			sycl::queue& q,
			ElementA* local_A,
			ElementB* B,
			ElementOutput* final_C,
			SymmMemory& symm,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::context const& ctx,
			sycl::device const& dev,
			int rank,
			int world_size) {

		allgather_gemm op{*this};
		return op(q, local_A, B, final_C, symm, options, hw_info, ctx, dev, rank, world_size);
	}

	sycl::event run_iteration_gemm_only(
			sycl::queue& q,
			ElementA* full_A,
			ElementB* B,
			ElementOutput* final_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			sycl::context const& ctx,
			sycl::device const& dev,
			int rank,
			int world_size) {

		gemm_only op{*this};
		return op(q, full_A, B, final_C, options, hw_info, ctx, dev, rank, world_size);
	}

	bool verify(
			ElementA* local_A,
			ElementB* B,
			ElementOutput* final_C,
			Options const& options,
			cutlass::KernelHardwareInfo const& hw_info,
			int rank,
			int world_size) {
		auto& q = *current_q_;
		int local_m = options.m / world_size;
		size_t local_a_elems = static_cast<size_t>(local_m) * options.k;
		size_t full_a_elems = static_cast<size_t>(options.m) * options.k;
		size_t b_elems = static_cast<size_t>(options.n) * options.k;
		size_t full_c_elems = static_cast<size_t>(options.m) * options.n;
		size_t shard_c_elems = static_cast<size_t>(local_m) * options.n;

		// Fill local_A and B with non-zero values
		float a_val = 1.0f;
		float b_val = 1.0f;
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(local_a_elems), [=](sycl::id<1> i) {
				local_A[i] = static_cast<ElementA>(a_val);
			});
		});
		q.submit([&](sycl::handler& h) {
			h.parallel_for(sycl::range<1>(b_elems), [=](sycl::id<1> i) {
				B[i] = static_cast<ElementB>(b_val);
			});
		});
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Step 1: MPI_Allgather local_A → host_full_A, upload to GPU
		std::vector<ElementA> host_local_a(local_a_elems);
		q.memcpy(host_local_a.data(), local_A, local_a_elems * sizeof(ElementA)).wait();

		std::vector<ElementA> host_full_a(full_a_elems);
		MPI_Allgather(host_local_a.data(), static_cast<int>(local_a_elems * sizeof(ElementA)), MPI_BYTE,
		              host_full_a.data(), static_cast<int>(local_a_elems * sizeof(ElementA)), MPI_BYTE,
		              MPI_COMM_WORLD);

		ElementA* gpu_full_a = sycl::malloc_device<ElementA>(full_a_elems, q);
		q.memcpy(gpu_full_a, host_full_a.data(), full_a_elems * sizeof(ElementA)).wait();

		// Step 2: Reference full GEMM (shard by shard) → d_ref[M×N]
		ElementOutput* d_ref = sycl::malloc_device<ElementOutput>(full_c_elems, q);
		q.memset(d_ref, 0, full_c_elems * sizeof(ElementOutput)).wait();

		for (int s = 0; s < world_size; ++s) {
			auto st = run_shard_gemm(q,
				gpu_full_a + static_cast<size_t>(s) * local_a_elems,
				B,
				d_ref + static_cast<size_t>(s) * shard_c_elems,
				options.alpha, options.beta, hw_info);
			if (st != cutlass::Status::kSuccess) {
				printf("[rank %d] verify: ref GEMM shard %d failed\n", rank, s);
				sycl::free(gpu_full_a, q);
				sycl::free(d_ref, q);
				return false;
			}
		}
		q.wait();

		std::vector<ElementOutput> host_ref(full_c_elems);
		q.memcpy(host_ref.data(), d_ref, full_c_elems * sizeof(ElementOutput)).wait();
		sycl::free(gpu_full_a, q);
		sycl::free(d_ref, q);

		// Step 3: Run one allgather_gemm iteration
		q.memset(final_C, 0, full_c_elems * sizeof(ElementOutput)).wait();
		MPI_Barrier(MPI_COMM_WORLD);
		auto ctx = q.get_context();
		auto dev = q.get_device();
		run_iteration(q, local_A, B, final_C, *symm_, options, hw_info, ctx, dev, rank, world_size);
		q.wait();
		MPI_Barrier(MPI_COMM_WORLD);

		// Step 4: Copy GPU result to host
		std::vector<ElementOutput> host_result(full_c_elems);
		q.memcpy(host_result.data(), final_C, full_c_elems * sizeof(ElementOutput)).wait();

		// Step 5: Compare
		double max_abs_diff = 0.0;
		double max_rel_diff = 0.0;
		size_t mismatch_count = 0;
		for (size_t i = 0; i < full_c_elems; ++i) {
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
		       max_abs_diff, max_rel_diff, mismatch_count, full_c_elems,
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
			throw std::runtime_error("allgather+gemm requires M divisible by world_size.");
		}

		int local_m = options.m / world_size;
		size_t local_a_elems = static_cast<size_t>(local_m) * options.k;
		size_t full_a_elems = static_cast<size_t>(options.m) * options.k;
		size_t b_elems = static_cast<size_t>(options.n) * options.k;
		size_t full_c_elems = static_cast<size_t>(options.m) * options.n;
		if (local_a_elems == 0 || full_a_elems == 0 || b_elems == 0 || full_c_elems == 0) {
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

		sycl::context ctx = current_q_->get_context();
		ElementA* local_A = sycl::malloc_device<ElementA>(local_a_elems, *current_q_);
		ElementA* full_A = nullptr;
		if (options.gemm_only != 0) {
			full_A = sycl::malloc_device<ElementA>(full_a_elems, *current_q_);
		}
		ElementB* B = sycl::malloc_device<ElementB>(b_elems, *current_q_);
		ElementOutput* final_C = sycl::malloc_device<ElementOutput>(full_c_elems, *current_q_);
		auto mb = [](size_t bytes) {
			return static_cast<double>(bytes) / (1024.0 * 1024.0);
		};
		printf("[rank %d] Allocated: local_A=%.2f MiB, full_A=%.2f MiB, B=%.2f MiB, final_C=%.2f MiB\n",
				rank,
				mb(local_a_elems * sizeof(ElementA)),
				mb(options.gemm_only != 0 ? full_a_elems * sizeof(ElementA) : 0),
				mb(b_elems * sizeof(ElementB)),
				mb(full_c_elems * sizeof(ElementOutput)));
		if (local_A == nullptr || B == nullptr || final_C == nullptr || (options.gemm_only != 0 && full_A == nullptr)) {
			throw std::runtime_error(
				"Device allocation failed: local_A=" + std::to_string(mb(local_a_elems * sizeof(ElementA))) +
				" MiB, full_A=" + std::to_string(mb(full_a_elems * sizeof(ElementA))) +
				" MiB, B=" + std::to_string(mb(b_elems * sizeof(ElementB))) +
				" MiB, final_C=" + std::to_string(mb(full_c_elems * sizeof(ElementOutput))) + " MiB.");
		}

		auto cleanup = [&]() {
			if (local_A) sycl::free(local_A, *current_q_);
			if (full_A) sycl::free(full_A, *current_q_);
			if (B) sycl::free(B, *current_q_);
			if (final_C) sycl::free(final_C, *current_q_);
		};

		initialize(local_A, B, final_C,
				options, hw_info, device, rank, world_size);
		if (options.gemm_only != 0) {
			current_q_->memset(full_A, 0, full_a_elems * sizeof(ElementA)).wait();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout << "[rank " << rank << "] initialization complete" << std::endl;

		// verification (allgather path only)
		if (options.verify != 0 && options.gemm_only == 0) {
			if (!verify(local_A, B, final_C, options, hw_info, rank, world_size)) {
				std::cerr << "[rank " << rank << "] verification failed!" << std::endl;
				cleanup();
				return cutlass::Status::kErrorInternal;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// warmup
		constexpr int kWarmupIters = 10;
		
		std::cout << "[rank " << rank << "] warmup start (" << kWarmupIters << " iters)" << std::endl;
		for (int iter = 0; iter < kWarmupIters; ++iter) {
			if (options.gemm_only != 0) {
				run_iteration_gemm_only(*current_q_, full_A, B, final_C, options, hw_info, ctx, device, rank, world_size);
			} else {
				run_iteration(*current_q_, local_A, B, final_C, *symm_, options, hw_info, ctx, device, rank, world_size);
			}
		}

		current_q_->wait();
		MPI_Barrier(MPI_COMM_WORLD); // ensure all ranks have finished warmup before starting benchmark iterations
		std::cout << "[rank " << rank << "] warmup done" << std::endl;

		// benchmark
		sycl::event ev_before;
		auto benchmark_start = std::chrono::high_resolution_clock::now();
		for (int iter = 0; iter < options.iterations; ++iter) {
			if (iter == 9) {
				if (options.gemm_only != 0) {
					ev_before = run_iteration_gemm_only(*current_q_, full_A, B, final_C, options, hw_info, ctx, device, rank, world_size);
				} else {
					ev_before = run_iteration(*current_q_, local_A, B, final_C, *symm_, options, hw_info, ctx, device, rank, world_size);
				}
			} else {
				if (options.gemm_only != 0) {
					run_iteration_gemm_only(*current_q_, full_A, B, final_C, options, hw_info, ctx, device, rank, world_size);
				} else {
					run_iteration(*current_q_, local_A, B, final_C, *symm_, options, hw_info, ctx, device, rank, world_size);
				}
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
			double tflops = (2.0 * options.m * options.n * options.k) * 1e-12;
			const char* label = (options.gemm_only != 0) ? "GEMM only" : "Pipelined allgather+GEMM";
			std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
			          << ", TP=" << world_size << std::endl;
			printf("%s (host):   [%4.3f]TFlop/s  (%6.4f)ms\n", label, tflops / (avg_ms / 1000.0), avg_ms);
			printf("%s (device): [%4.3f]TFlop/s  (%6.4f)ms\n", label, tflops / (avg_device_ms / 1000.0), avg_device_ms);
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
