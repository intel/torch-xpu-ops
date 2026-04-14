/*
 * Two-device (rank0 <-> rank1) ping-pong latency benchmark.
 *
 * This version uses SymmMemory IPC mapping so each rank can access the peer
 * GPU allocation directly from device code. The signaling path keeps the same
 * low-level pattern as pingpong_latency.cpp: lsc_load.ugm.uc.uc polling,
 * lsc_store.ugm.uc.uc writes, and lsc_fence ordering.
 */

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <mpi.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "symm.hpp"

static constexpr uint32_t NUM_ROUNDS = 100000u;
static constexpr uint32_t NUM_WARMUP = 2000u;

struct alignas(64) PingPongPad {
	uint32_t ctr;
	uint32_t ready;
	uint32_t start;
	uint32_t done;
	uint8_t pad[48];
};

static_assert(sizeof(PingPongPad) == 64, "PingPongPad must be 64 bytes");

static double get_gpu_tick_ns(ze_device_handle_t ze_dev) {
	ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
	ze_result_t rc = zeDeviceGetProperties(ze_dev, &props);
	if (rc != ZE_RESULT_SUCCESS) {
		throw std::runtime_error("zeDeviceGetProperties failed");
	}
	return static_cast<double>(props.timerResolution);
}

/*
 * D2D ping-pong protocol record (inside submit_pingpong_kernel)
 *
 * Roles:
 *   - initiator (rank0): sends request first, then waits for reply.
 *   - responder (rank1): waits for request first, then sends reply.
 *
 * Signaling pads:
 *   local->ready / peer->ready : both kernels publish readiness.
 *   local->start / peer->start : initiator starts responder after both ready.
 *   local->ctr   / peer->ctr   : request/reply counter transfer.
 *   local->done                : session completion marker per rank.
 *
 * Stages:
 *   1) Ready stage:
 *      each rank writes local->ready = 1.
 *   2) Start stage:
 *      initiator waits peer->ready == 1, then writes peer->start = 1.
 *      responder spins on local->start until == 1.
 *   3) Round stage (r = 0 .. num_rounds-1):
 *      req = 2*r+1, rep = req+1.
 *      initiator writes peer->ctr=req, then waits local->ctr==rep.
 *      responder waits local->ctr==req, then writes peer->ctr=rep.
 *   4) Done stage:
 *      initiator writes local->done = num_rounds.
 *      responder writes local->done = 1.
 *
 * Memory ordering path:
 *   - Poll loads:  lsc_load.ugm.uc.uc
 *   - Writes:      lsc_store.ugm.uc.uc
 *   - Fences:      lsc_fence.ugm.invalidate / lsc_fence.ugm.evict
 */

static sycl::event submit_pingpong_kernel(sycl::queue &kq,
									  PingPongPad *local,
									  PingPongPad *peer,
									  uint32_t num_rounds,
									  bool is_initiator) {
	return kq.submit([&](sycl::handler &h) {
		h.parallel_for(
			sycl::nd_range<1>(sycl::range<1>(16), sycl::range<1>(16)),
			[=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
				if (item.get_local_id(0) != 0) return;

				uint32_t *local_ctr = &local->ctr;
				uint32_t *peer_ctr = &peer->ctr;
				uint32_t *local_ready = &local->ready;
				uint32_t *peer_ready = &peer->ready;
				uint32_t *local_start = &local->start;
				uint32_t *peer_start = &peer->start;
				uint32_t *local_done = &local->done;

				// Stage 1: publish local readiness.
				{
					uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile (
						"lsc_store.ugm.uc.uc (M1, 16)"
						"  flat[%0]:a64 %1:d32\n"
						: : "rw"(local_ready), "rw"(one)
					);
					__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
					*local_ready = one;
#endif
				}

				// Stage 2: coordinated start.
				if (is_initiator) {
					uint32_t ready = 0u;
					do {
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
							: "=rw"(ready) : "rw"(peer_ready)
						);
#else
						ready = *peer_ready;
#endif
					} while (ready != 1u);

					{
						uint32_t one = 1u;
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_store.ugm.uc.uc (M1, 16)"
							"  flat[%0]:a64 %1:d32\n"
							: : "rw"(peer_start), "rw"(one)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*peer_start = one;
#endif
					}
				} else {
					uint32_t started = 0u;
					do {
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
							: "=rw"(started) : "rw"(local_start)
						);
#else
						started = *local_start;
#endif
					} while (started != 1u);
				}

				// Stage 3: round-trip exchange loop.
				for (uint32_t r = 0; r < num_rounds; ++r) {
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile ("lsc_fence.ugm.invalidate.tile\n" : : :);
#endif

					const uint32_t req = 2u * r + 1u; // request value (odd)
					const uint32_t rep = req + 1u; // reply value (even)
					uint32_t val = 0u;

					if (is_initiator) {
#ifdef __SYCL_DEVICE_ONLY__
						// initiator: send req to peer, then wait local rep.
#endif
#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile (
							"lsc_store.ugm.uc.uc (M1, 16)"
							"  flat[%0]:a64 %1:d32\n"
							: : "rw"(peer_ctr), "rw"(req)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*peer_ctr = req;
#endif

						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(val) : "rw"(local_ctr)
							);
#else
							val = *local_ctr;
#endif
						} while (val != rep);
					} else {
						// responder: wait local req, then send rep to peer.
						do {
#ifdef __SYCL_DEVICE_ONLY__
							__asm__ volatile (
								"lsc_load.ugm.uc.uc (M1, 16) %0:d32 flat[%1]:a64\n"
								: "=rw"(val) : "rw"(local_ctr)
							);
#else
							val = *local_ctr;
#endif
						} while (val != req);

#ifdef __SYCL_DEVICE_ONLY__
						__asm__ volatile ("lsc_fence.ugm.invalidate.tile\n" : : :);
						__asm__ volatile (
							"lsc_store.ugm.uc.uc (M1, 16)"
							"  flat[%0]:a64 %1:d32\n"
							: : "rw"(peer_ctr), "rw"(rep)
						);
						__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
						*peer_ctr = rep;
#endif
					}
				}

				// Stage 4: publish rank-local completion.
				{
					uint32_t done = is_initiator ? num_rounds : 1u;
#ifdef __SYCL_DEVICE_ONLY__
					__asm__ volatile (
						"lsc_store.ugm.uc.uc (M1, 16)"
						"  flat[%0]:a64 %1:d32\n"
						: : "rw"(local_done), "rw"(done)
					);
					__asm__ volatile ("lsc_fence.ugm.evict.tile\n" : : :);
#else
					*local_done = done;
#endif
				}
			}
		);
	});
}

static double run_session(sycl::queue &q,
						  PingPongPad *local,
						  PingPongPad *peer,
						  uint32_t rounds,
						  bool is_initiator,
						  double gpu_tick_ns,
						  ze_device_handle_t ze_dev,
						  int rank) {
	q.memset(local, 0, sizeof(PingPongPad)).wait();

	uint64_t host_ts_begin = 0;
	uint64_t gpu_ts_begin = 0;
	uint64_t host_ts_end = 0;
	uint64_t gpu_ts_end = 0;

	MPI_Barrier(MPI_COMM_WORLD);
	auto evt = submit_pingpong_kernel(q, local, peer, rounds, is_initiator);

	if (is_initiator) {
		zeDeviceGetGlobalTimestamps(ze_dev, &host_ts_begin, &gpu_ts_begin);
	}
	evt.wait_and_throw();
	if (is_initiator) {
		zeDeviceGetGlobalTimestamps(ze_dev, &host_ts_end, &gpu_ts_end);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	uint32_t final_local_ctr = 0;
	uint32_t final_local_done = 0;
	q.memcpy(&final_local_ctr, &local->ctr, sizeof(uint32_t)).wait();
	q.memcpy(&final_local_done, &local->done, sizeof(uint32_t)).wait();

	const uint32_t expected_ctr = is_initiator ? (2u * rounds) : (2u * rounds - 1u);
	const uint32_t expected_done = is_initiator ? rounds : 1u;
	if (final_local_ctr != expected_ctr || final_local_done != expected_done) {
		throw std::runtime_error(
			"rank " + std::to_string(rank) +
			" final check failed: ctr=" + std::to_string(final_local_ctr) +
			" done=" + std::to_string(final_local_done));
	}

	if (is_initiator) {
		return double(gpu_ts_end - gpu_ts_begin) * gpu_tick_ns / rounds * 1e-3;
	}
	return 0.0;
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int rank = 0;
	int world_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size != 2) {
		if (rank == 0) {
			std::cerr << "This benchmark requires exactly 2 MPI ranks.\n";
		}
		MPI_Finalize();
		return 1;
	}

	auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if (devices.size() < 2) {
		if (rank == 0) {
			std::cerr << "Need at least 2 GPU devices, found " << devices.size() << "\n";
		}
		MPI_Finalize();
		return 1;
	}

	sycl::device dev = devices[static_cast<size_t>(rank) % devices.size()];
	sycl::context ctx(dev);
	sycl::queue q(ctx, dev, sycl::property::queue::in_order{});

	auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
	const double gpu_tick_ns = get_gpu_tick_ns(ze_dev);

	if (rank == 0) {
		std::cout << "Using 2-rank D2D ping-pong via SymmMemory IPC\n";
	}
	std::cout << "[rank " << rank << "] Device: "
			  << dev.get_info<sycl::info::device::name>() << "\n";

	SymmMemory symm(
		1, 1, 1,
		rank, world_size,
		q,
		1,
		sizeof(PingPongPad) / sizeof(uint16_t),
		static_cast<size_t>(world_size)
	);

	auto *local = reinterpret_cast<PingPongPad *>(symm.get_data_buffer(rank));
	auto *peer = reinterpret_cast<PingPongPad *>(symm.get_data_buffer(1 - rank));
	const bool is_initiator = (rank == 0);

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		std::cout << std::fixed << std::setprecision(1)
				  << "GPU timer (rank0): " << gpu_tick_ns << " ns/tick\n"
				  << "Warming up (" << NUM_WARMUP << " rounds)...\n";
	}
	(void)run_session(q, local, peer, NUM_WARMUP, is_initiator, gpu_tick_ns, ze_dev, rank);

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		std::cout << "Measuring (" << NUM_ROUNDS << " rounds)...\n";
	}
	const double mean_roundtrip_us =
		run_session(q, local, peer, NUM_ROUNDS, is_initiator, gpu_tick_ns, ze_dev, rank);

	if (rank == 0) {
		std::cout << std::fixed << std::setprecision(3)
				  << "D2D round-trip mean latency (" << NUM_ROUNDS << " rounds):\n"
				  << "  GPU-timer mean = " << mean_roundtrip_us << " us\n"
				  << "  estimated one-way = " << (mean_roundtrip_us * 0.5) << " us\n";
	}

	MPI_Finalize();
	return 0;
}
