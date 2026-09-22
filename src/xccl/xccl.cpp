/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <xccl/xccl.h>

namespace c10d {
namespace xccl {

void oneccl_group_start() {
  C10D_XCCL_CHECK(onecclGroupStart(), nullptr);
}

void oneccl_group_end() {
  C10D_XCCL_CHECK(onecclGroupEnd(), nullptr);
}

void onecclAllReduce(
    at::Tensor& input,
    at::Tensor& output,
    onecclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input, xcclDataType, comm);
  C10D_XCCL_CHECK(
      onecclAllReduce(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          xcclDataType,
          xcclReduceOp,
          comm,
          &stream.queue()),
      comm);
}

void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    onecclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    const int root,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input, xcclDataType, comm);
  C10D_XCCL_CHECK(
      onecclReduce(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          xcclDataType,
          xcclReduceOp,
          root,
          comm,
          &stream.queue()),
      comm);
}

void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    onecclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), false);
  C10D_XCCL_CHECK(
      onecclBroadcast(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          xcclDataType,
          root,
          comm,
          &stream.queue()),
      comm);
}

void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    onecclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input, xcclDataType, comm);
  C10D_XCCL_CHECK(
      onecclReduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)output.numel(),
          xcclDataType,
          xcclReduceOp,
          comm,
          &stream.queue()),
      comm);
}

void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    onecclComm_t& comm,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), false);
  C10D_XCCL_CHECK(
      onecclAllGather(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          xcclDataType,
          comm,
          &stream.queue()),
      comm);
}

void onecclSend(
    at::Tensor& input,
    onecclComm_t& comm,
    const int dstRank,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), false);
  C10D_XCCL_CHECK(
      onecclSend(
          input.data_ptr(),
          (size_t)input.numel(),
          xcclDataType,
          dstRank,
          comm,
          &stream.queue()),
      comm);
}

void onecclRecv(
    at::Tensor& output,
    onecclComm_t& comm,
    const int srcRank,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(output.scalar_type(), false);
  C10D_XCCL_CHECK(
      onecclRecv(
          output.data_ptr(),
          (size_t)output.numel(),
          xcclDataType,
          srcRank,
          comm,
          &stream.queue()),
      comm);
}

void onecclGather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    onecclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream) {
  size_t count = inputs.numel();
  auto xcclDataType = getXcclDataType(inputs.scalar_type(), false);
  int numranks = 0, cur_rank = 0;
  C10D_XCCL_CHECK(onecclCommCount(comm, &numranks), comm);
  C10D_XCCL_CHECK(onecclCommUserRank(comm, &cur_rank), comm);
  OnecclGroupGuard group_guard;
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());
        C10D_XCCL_CHECK(
            onecclRecv(recvbuff, count, xcclDataType, r, comm, &stream.queue()),
            comm);
      } else {
        // on its own rank, simply copy from the input
        outputs[r].copy_(inputs);
      }
    }
  } else {
    C10D_XCCL_CHECK(
        onecclSend(
            inputs.data_ptr(),
            count,
            xcclDataType,
            root,
            comm,
            &stream.queue()),
        comm);
  }
}

void onecclScatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    onecclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(outputs.scalar_type(), false);
  int numranks = 0, cur_rank = 0;
  C10D_XCCL_CHECK(onecclCommCount(comm, &numranks), comm);
  C10D_XCCL_CHECK(onecclCommUserRank(comm, &cur_rank), comm);
  OnecclGroupGuard group_guard;
  if (cur_rank == root) {
    for (const auto r : c10::irange(numranks)) {
      if (r != root) {
        size_t send_count = inputs[r].numel();
        C10D_XCCL_CHECK(
            onecclSend(
                inputs[r].data_ptr(),
                send_count,
                xcclDataType,
                r,
                comm,
                &stream.queue()),
            comm);
      } else {
        // on its own rank, simply copy from the input
        outputs.copy_(inputs[r]);
      }
    }
  } else {
    size_t recv_count = outputs.numel();
    C10D_XCCL_CHECK(
        onecclRecv(
            outputs.data_ptr(),
            recv_count,
            xcclDataType,
            root,
            comm,
            &stream.queue()),
        comm);
  }
}

static std::pair<bool, size_t> checkUniformAllToAll(
    const size_t* sendcounts,
    const size_t* senddispls,
    const size_t* recvcounts,
    const size_t* recvdispls,
    int numranks) {
  if (numranks <= 0) {
    return {false, 0};
  }
  size_t uniformCount = sendcounts[0];
  for (int r = 0; r < numranks; ++r) {
    if (sendcounts[r] != uniformCount || recvcounts[r] != uniformCount) {
      return {false, 0};
    }
    // Check for contiguous displacements
    if (senddispls[r] != static_cast<size_t>(r) * uniformCount ||
        recvdispls[r] != static_cast<size_t>(r) * uniformCount) {
      return {false, 0};
    }
  }
  return {uniformCount > 0, uniformCount};
}

void onecclAllToAll(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    at::ScalarType dataType,
    onecclComm_t& comm,
    at::xpu::XPUStream& stream) {
  auto xcclDataType = getXcclDataType(dataType, false);
  int numranks = 0;
  C10D_XCCL_CHECK(onecclCommCount(comm, &numranks), comm);

  auto [isUniform, uniformCount] = checkUniformAllToAll(
      sendcounts, senddispls, recvcounts, recvdispls, numranks);

  if (isUniform) {
    // Use native onecclAllToAll for uniform case
    C10D_XCCL_CHECK(
        onecclAllToAll(
            sendbuff,
            recvbuff,
            uniformCount,
            xcclDataType,
            comm,
            &stream.queue()),
        comm);
    return;
  }

  // Fallback to send/recv based implementation for non-uniform case
  OnecclGroupGuard group_guard;
  for (const auto r : c10::irange(numranks)) {
    if (sendcounts[r] != 0) {
      C10D_XCCL_CHECK(
          onecclSend(
              ((char*)sendbuff) + senddispls[r] * size,
              sendcounts[r],
              xcclDataType,
              r,
              comm,
              &stream.queue()),
          comm);
    }
    if (recvcounts[r] != 0) {
      C10D_XCCL_CHECK(
          onecclRecv(
              ((char*)recvbuff) + recvdispls[r] * size,
              recvcounts[r],
              xcclDataType,
              r,
              comm,
              &stream.queue()),
          comm);
    }
  }
}

} // namespace xccl
} // namespace c10d
