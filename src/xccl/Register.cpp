/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

namespace c10d {
namespace ops {
namespace {
c10::intrusive_ptr<Work> send_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::SEND, /*async_op=*/true, /*root=*/dstRank, tensor_vec);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
  process_group->firePostHook(
      HookOpName::SEND, /*async_op=*/true, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<Work> recv_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::RECV,
      /*async_op=*/true,
      /*root=*/srcRank,
      std::vector<at::Tensor>(),
      tensor_vec);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
  process_group->firePostHook(
      HookOpName::RECV, /*async_op=*/true, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<Work> recv_any_source_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::RECV,
      /*async_op=*/true,
      /*root=*/-1,
      std::vector<at::Tensor>(),
      tensor_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->recvAnysource(tensor_vec, static_cast<int>(tag));
  process_group->firePostHook(
      HookOpName::RECV, /*async_op=*/true, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<Work> reduce_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::REDUCE, asyncOp, root_rank, tensor_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->reduce(
                      tensor_vec,
                      ReduceOptions{
                          *reduce_op.get(),
                          root_rank,
                          root_tensor,
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(HookOpName::REDUCE, asyncOp, hook_op_id, work);
  return work;
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::BROADCAST, asyncOp, root_rank, tensor_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->broadcast(
                      tensor_vec,
                      BroadcastOptions{
                          root_rank,
                          root_tensor,
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(HookOpName::BROADCAST, asyncOp, hook_op_id, work);
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    const std::optional<at::Tensor>& sparse_indices,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto tensor_vec = tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLREDUCE, asyncOp, -1, tensor_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->allreduce(
                      tensor_vec,
                      AllreduceOptions{
                          .reduceOp = *reduce_op.get(),
                          .timeout = std::chrono::milliseconds(timeout),
                          .asyncOp = asyncOp,
                          .config = std::move(config)});
  process_group->firePostHook(HookOpName::ALLREDUCE, asyncOp, hook_op_id, work);
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

c10::intrusive_ptr<Work> allreduce_coalesced_XPU(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = asyncOp;
  opts.config = std::move(config);
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLREDUCE_COALESCED, asyncOp, -1, tensor_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->allreduce_coalesced(tensor_vec, opts);
  process_group->firePostHook(
      HookOpName::ALLREDUCE_COALESCED, asyncOp, hook_op_id, work);
  return work;
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_XPU(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto input_tensors_vec = input_tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLGATHER, asyncOp, -1, input_tensors_vec, output_tensors);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{
                  std::chrono::milliseconds(timeout),
                  asyncOp,
                  std::move(config)});
  process_group->firePostHook(HookOpName::ALLGATHER, asyncOp, hook_op_id, work);
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_XPU(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLGATHER_BASE, asyncOp, -1, input_tensor, output_tensor);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->all_gather_single(
                      output_tensor,
                      input_tensor,
                      AllgatherOptions{
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(
      HookOpName::ALLGATHER_BASE, asyncOp, hook_op_id, work);
  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

c10::intrusive_ptr<Work> allgather_coalesced_XPU(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    OptionalCollectiveConfig config) {
  auto input_list_vec = input_list.vec();
  auto opts = AllgatherOptions{};
  opts.asyncOp = asyncOp;
  opts.config = std::move(config);
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLGATHER_COALESCED,
      asyncOp,
      -1,
      input_list_vec,
      output_lists);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->allgather_coalesced(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
              input_list_vec,
              opts);
  process_group->firePostHook(
      HookOpName::ALLGATHER_COALESCED, asyncOp, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_XPU(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    OptionalCollectiveConfig config) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  auto opts = AllgatherOptions{};
  opts.asyncOp = asyncOp;
  opts.config = std::move(config);
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLGATHER_INTO_TENSOR_COALESCED,
      asyncOp,
      -1,
      input_vec,
      output_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->all_gather_single_coalesced(output_vec, input_vec, opts);
  process_group->firePostHook(
      HookOpName::ALLGATHER_INTO_TENSOR_COALESCED, asyncOp, hook_op_id, work);
  return work;
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> reduce_scatter_XPU(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto output_tensors_vec = output_tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::REDUCE_SCATTER,
      asyncOp,
      -1,
      input_tensors,
      output_tensors_vec);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(),
                  std::chrono::milliseconds(timeout),
                  asyncOp,
                  std::move(config)});
  process_group->firePostHook(
      HookOpName::REDUCE_SCATTER, asyncOp, hook_op_id, work);
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_XPU(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto hook_op_id = process_group->firePreHook(
      HookOpName::REDUCE_SCATTER_BASE,
      asyncOp,
      -1,
      input_tensor,
      output_tensor);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->reduce_scatter_single(
                      output_tensor,
                      input_tensor,
                      ReduceScatterOptions{
                          *reduce_op.get(),
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(
      HookOpName::REDUCE_SCATTER_BASE, asyncOp, hook_op_id, work);
  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced_XPU(
    at::TensorList outputs,
    at::TensorList inputs,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::REDUCE_SCATTER_TENSOR_COALESCED,
      asyncOp,
      -1,
      input_vec,
      output_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->reduce_scatter_single_coalesced(
                      output_vec,
                      input_vec,
                      ReduceScatterOptions{
                          *reduce_op.get(),
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(
      HookOpName::REDUCE_SCATTER_TENSOR_COALESCED, asyncOp, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<Work> gather_XPU(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto input_tensors_vec = input_tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::GATHER,
      asyncOp,
      root_rank,
      input_tensors_vec,
      output_tensors);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->gather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              GatherOptions{
                  root_rank,
                  std::chrono::milliseconds(timeout),
                  asyncOp,
                  std::move(config)});
  process_group->firePostHook(HookOpName::GATHER, asyncOp, hook_op_id, work);
  return work;
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_XPU(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    bool asyncOp,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::SCATTER,
      asyncOp,
      root_rank,
      input_tensors,
      output_tensors_vec);
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{
                  root_rank, std::chrono::milliseconds(timeout), asyncOp});
  process_group->firePostHook(HookOpName::SCATTER, asyncOp, hook_op_id, work);
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_XPU(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLTOALL, asyncOp, -1, input_tensors_vec, output_tensors_vec);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(HookOpName::ALLTOALL, asyncOp, hook_op_id, work);
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

c10::intrusive_ptr<Work> alltoall_base_XPU(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool asyncOp,
    int64_t timeout,
    OptionalCollectiveConfig config) {
  auto hook_op_id = process_group->firePreHook(
      HookOpName::ALLTOALL_BASE, asyncOp, -1, input, output);
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->all_to_all_single(
                      output,
                      input,
                      output_split_sizes,
                      input_split_sizes,
                      AllToAllOptions{
                          std::chrono::milliseconds(timeout),
                          asyncOp,
                          std::move(config)});
  process_group->firePostHook(
      HookOpName::ALLTOALL_BASE, asyncOp, hook_op_id, work);
  return work;
}

c10::intrusive_ptr<Work> barrier_XPU(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    bool asyncOp,
    int64_t timeout) {
  auto opts = BarrierOptions{};
  opts.device_ids = device_ids;
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = asyncOp;
  auto hook_op_id =
      process_group->firePreHook(HookOpName::BARRIER, asyncOp, -1);
  auto work = process_group->getBackend(c10::DeviceType::XPU)->barrier(opts);
  process_group->firePostHook(HookOpName::BARRIER, asyncOp, hook_op_id, work);
  return work;
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("send", send_XPU);
  m.impl("recv_", recv_XPU);
  m.impl("recv_any_source_", recv_any_source_XPU);
  m.impl("reduce_", reduce_XPU);
  m.impl("broadcast_", broadcast_XPU);
  m.impl("allreduce_", allreduce_XPU);
  m.impl("allreduce_coalesced_", allreduce_coalesced_XPU);
  m.impl("allgather_", allgather_XPU);
  m.impl("_allgather_base_", _allgather_base_XPU);
  m.impl("allgather_coalesced_", allgather_coalesced_XPU);
  m.impl(
      "allgather_into_tensor_coalesced_", allgather_into_tensor_coalesced_XPU);
  m.impl("reduce_scatter_", reduce_scatter_XPU);
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_XPU);
  m.impl(
      "reduce_scatter_tensor_coalesced_", reduce_scatter_tensor_coalesced_XPU);
  m.impl("gather_", gather_XPU);
  m.impl("scatter_", scatter_XPU);
  m.impl("alltoall_", alltoall_XPU);
  m.impl("alltoall_base_", alltoall_base_XPU);
  m.impl("barrier", barrier_XPU);
}
} // namespace

} // namespace ops
} // namespace c10d
