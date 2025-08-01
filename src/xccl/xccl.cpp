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
    XcclDataType& xcclDataType,
    XcclRedOp& xcclReduceOp,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclAllReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        std::get<onecclRedOp_t>(xcclReduceOp),
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::allreduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<ccl::datatype>(xcclDataType),
        std::get<ccl::reduction>(xcclReduceOp),
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    XcclDataType& xcclDataType,
    XcclRedOp& xcclReduceOp,
    const int root,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        std::get<onecclRedOp_t>(xcclReduceOp),
        root,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::reduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<ccl::datatype>(xcclDataType),
        std::get<ccl::reduction>(xcclReduceOp),
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
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclBroadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        root,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::broadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<ccl::datatype>(xcclDataType),
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
    XcclDataType& xcclDataType,
    XcclRedOp& xcclReduceOp,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclReduceScatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        std::get<onecclRedOp_t>(xcclReduceOp),
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::reduce_scatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        std::get<ccl::datatype>(xcclDataType),
        std::get<ccl::reduction>(xcclReduceOp),
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::allgather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        std::get<ccl::datatype>(xcclDataType),
        std::get<ccl::communicator>(comm),
        xcclStream);
  }
  return;
}

void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclSend(
        input.data_ptr(),
        (size_t)input.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        dstRank,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::send(
        input.data_ptr(),
        (size_t)input.numel(),
        std::get<ccl::datatype>(xcclDataType),
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
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
    onecclRecv(
        output.data_ptr(),
        (size_t)output.numel(),
        std::get<onecclDataType_t>(xcclDataType),
        srcRank,
        std::get<onecclComm_t>(comm),
        &SyclQueue);
  } else {
    ccl::recv(
        output.data_ptr(),
        (size_t)output.numel(),
        std::get<ccl::datatype>(xcclDataType),
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
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  size_t count = inputs.numel();

  if (isCCLV2EnabledCached()) {
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
              std::get<onecclDataType_t>(xcclDataType),
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
          std::get<onecclDataType_t>(xcclDataType),
          root,
          std::get<onecclComm_t>(comm),
          &SyclQueue);
    }
    onecclGroupEnd();
  } else {
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
              std::get<ccl::datatype>(xcclDataType),
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
          std::get<ccl::datatype>(xcclDataType),
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
    XcclDataType& xcclDataType,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  if (isCCLV2EnabledCached()) {
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
              std::get<onecclDataType_t>(xcclDataType),
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
          std::get<onecclDataType_t>(xcclDataType),
          root,
          std::get<onecclComm_t>(comm),
          &SyclQueue);
    }
    onecclGroupEnd();
  } else {
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
              std::get<ccl::datatype>(xcclDataType),
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
          std::get<ccl::datatype>(xcclDataType),
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
    XcclDataType& xcclDataType,
    xcclComm_t& comm,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  xccl::oneccl_v2_group_start();
  if (isCCLV2EnabledCached()) {
    int numranks = 0;
    onecclCommCount(std::get<onecclComm_t>(comm), &numranks);
    for (const auto r : c10::irange(numranks)) {
      if (sendcounts[r] != 0) {
        onecclSend(
            ((char*)sendbuff) + senddispls[r] * size,
            sendcounts[r],
            std::get<onecclDataType_t>(xcclDataType),
            r,
            std::get<onecclComm_t>(comm),
            &SyclQueue);
      }
      if (recvcounts[r] != 0) {
        onecclRecv(
            ((char*)recvbuff) + recvdispls[r] * size,
            recvcounts[r],
            std::get<onecclDataType_t>(xcclDataType),
            r,
            std::get<onecclComm_t>(comm),
            &SyclQueue);
      }
    }
  } else {
    int numranks = std::get<ccl::communicator>(comm).size();
    for (const auto r : c10::irange(numranks)) {
      if (sendcounts[r] != 0) {
        ccl::send(
            ((char*)sendbuff) + senddispls[r] * size,
            sendcounts[r],
            std::get<ccl::datatype>(xcclDataType),
            r,
            std::get<ccl::communicator>(comm),
            xcclStream);
      }
      if (recvcounts[r] != 0) {
        ccl::recv(
            ((char*)recvbuff) + recvdispls[r] * size,
            recvcounts[r],
            std::get<ccl::datatype>(xcclDataType),
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
