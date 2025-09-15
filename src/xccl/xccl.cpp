#include "xccl/xccl.h"

namespace c10d {
namespace xccl {

void oneccl_v2_group_start() {
  if (isCCLV2EnabledCached()) {
    onecclGroupStart();
  } else {
    ccl::group_start();
  }
}

void oneccl_v2_group_end() {
  if (isCCLV2EnabledCached()) {
    onecclGroupEnd();
  } else {
    ccl::group_end();
  }
}

void onecclAllReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV2(reduceOp, input);
    onecclAllReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV1(reduceOp, input);
    ccl::allreduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    const int root,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV2(reduceOp, input);
    onecclReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        root,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV1(reduceOp, input);
    ccl::reduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        root,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), false);
    onecclBroadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        root,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), false);
    ccl::broadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        root,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV2(reduceOp, input);
    onecclReduceScatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        xcclReduceOp,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), true);
    auto xcclReduceOp = getXcclReduceOpV1(reduceOp, input);
    ccl::reduce_scatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        xcclReduceOp,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), false);
    onecclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), false);
    ccl::allgather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(input.scalar_type(), false);
    onecclSend(
        input.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        dstRank,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(input.scalar_type(), false);
    ccl::send(
        input.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        dstRank,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclRecv(
    at::Tensor& output,
    xcclComm_t& comm,
    const int srcRank,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(output.scalar_type(), false);
    onecclRecv(
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        srcRank,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    auto xcclDataType = getXcclDataTypeV1(output.scalar_type(), false);
    ccl::recv(
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        srcRank,
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclGather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  size_t count = inputs.numel();

  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(inputs.scalar_type(), false);
    int numranks = 0, cur_rank = 0;
    onecclCommCount(std::get<onecclComm_t>(comm), &numranks);
    onecclCommUserRank(std::get<onecclComm_t>(comm), &cur_rank);
    onecclGroupStart();
    if (cur_rank == root) {
      for (const auto r : c10::irange(numranks)) {
        if (r != root) {
          auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());
          onecclRecv(
              recvbuff,
              count,
              xcclDataType,
              r,
              std::get<onecclComm_t>(comm),
              &SyclQueue);
        } else {
          // on its own rank, simply copy from the input
          outputs[r].copy_(inputs);
        }
      }
    } else {
      onecclSend(
          inputs.data_ptr(),
          count,
          xcclDataType,
          root,
          std::get<onecclComm_t>(comm),
          &SyclQueue);
    }
    onecclGroupEnd();
  } else {
    auto xcclDataType = getXcclDataTypeV1(inputs.scalar_type(), false);
    int numranks = std::get<ccl::communicator>(comm).size();
    int cur_rank = std::get<ccl::communicator>(comm).rank();
    ccl::group_start();
    if (cur_rank == root) {
      for (const auto r : c10::irange(numranks)) {
        if (r != root) {
          auto* recvbuff = reinterpret_cast<char*>(outputs[r].data_ptr());
          ccl::recv(
              recvbuff,
              count,
              xcclDataType,
              r,
              std::get<ccl::communicator>(comm),
              xcclStream);
        } else {
          // on its own rank, simply copy from the input
          outputs[r].copy_(inputs);
        }
      }
    } else {
      ccl::send(
          inputs.data_ptr(),
          count,
          xcclDataType,
          root,
          std::get<ccl::communicator>(comm),
          xcclStream);
    }
    ccl::group_end();
  }
  return;
}

void onecclScatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(outputs.scalar_type(), false);
    int numranks = 0, cur_rank = 0;
    onecclCommCount(std::get<onecclComm_t>(comm), &numranks);
    onecclCommUserRank(std::get<onecclComm_t>(comm), &cur_rank);
    onecclGroupStart();
    if (cur_rank == root) {
      for (const auto r : c10::irange(numranks)) {
        if (r != root) {
          size_t send_count = inputs[r].numel();
          onecclSend(
              inputs[r].data_ptr(),
              send_count,
              xcclDataType,
              r,
              std::get<onecclComm_t>(comm),
              &SyclQueue);
        } else {
          // on its own rank, simply copy from the input
          outputs.copy_(inputs[r]);
        }
      }
    } else {
      size_t recv_count = outputs.numel();
      onecclRecv(
          outputs.data_ptr(),
          recv_count,
          xcclDataType,
          root,
          std::get<onecclComm_t>(comm),
          &SyclQueue);
    }
    onecclGroupEnd();
  } else {
    auto xcclDataType = getXcclDataTypeV1(outputs.scalar_type(), false);
    int numranks = std::get<ccl::communicator>(comm).size();
    int cur_rank = std::get<ccl::communicator>(comm).rank();
    ccl::group_start();
    if (cur_rank == root) {
      for (const auto r : c10::irange(numranks)) {
        if (r != root) {
          size_t send_count = inputs[r].numel();
          ccl::send(
              inputs[r].data_ptr(),
              send_count,
              xcclDataType,
              r,
              std::get<ccl::communicator>(comm),
              xcclStream);
        } else {
          // on its own rank, simply copy from the input
          outputs.copy_(inputs[r]);
        }
      }
    } else {
      size_t recv_count = outputs.numel();
      ccl::recv(
          outputs.data_ptr(),
          recv_count,
          xcclDataType,
          root,
          std::get<ccl::communicator>(comm),
          xcclStream);
    }
    ccl::group_end();
  }
  return;
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
    xcclComm_t& comm,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  xccl::oneccl_v2_group_start();
  if (isCCLV2EnabledCached()) {
    auto xcclDataType = getXcclDataTypeV2(dataType, false);
    int numranks = 0;
    onecclCommCount(std::get<onecclComm_t>(comm), &numranks);
    for (const auto r : c10::irange(numranks)) {
      if (sendcounts[r] != 0) {
        onecclSend(
            ((char*)sendbuff) + senddispls[r] * size,
            sendcounts[r],
            xcclDataType,
            r,
            std::get<onecclComm_t>(comm),
            &SyclQueue);
      }
      if (recvcounts[r] != 0) {
        onecclRecv(
            ((char*)recvbuff) + recvdispls[r] * size,
            recvcounts[r],
            xcclDataType,
            r,
            std::get<onecclComm_t>(comm),
            &SyclQueue);
      }
    }
  } else {
    auto xcclDataType = getXcclDataTypeV1(dataType, false);
    int numranks = std::get<ccl::communicator>(comm).size();
    for (const auto r : c10::irange(numranks)) {
      if (sendcounts[r] != 0) {
        ccl::send(
            ((char*)sendbuff) + senddispls[r] * size,
            sendcounts[r],
            xcclDataType,
            r,
            std::get<ccl::communicator>(comm),
            xcclStream);
      }
      if (recvcounts[r] != 0) {
        ccl::recv(
            ((char*)recvbuff) + recvdispls[r] * size,
            recvcounts[r],
            xcclDataType,
            r,
            std::get<ccl::communicator>(comm),
            xcclStream);
      }
    }
  }
  xccl::oneccl_v2_group_end();
  return;
}

} // namespace xccl
} // namespace c10d
