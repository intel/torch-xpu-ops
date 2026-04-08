#include <comm/SYCLContext.h>
#include <ATen/native/xpu/sycl/JaggedIndexSelect2D.h>

namespace at::native::xpu {

// Helper for binary search (upper_bound)
// Finds the first index i such that data[i] > target
// range: [0, n)
template <typename T>
inline int binary_search_upper_bound(const T* data, int n, T target) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (data[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

template <typename scalar_t, typename index_t, typename offset_t>
class JaggedIndexSelect2dKernel {
public:
    JaggedIndexSelect2dKernel(
        scalar_t* output,
        const scalar_t* values,
        const index_t* indices,
        const offset_t* input_offsets,
        const offset_t* output_offsets,
        int64_t num_dense_output_rows,
        int64_t num_output_rows, // Size of indices/output_offsets
        int64_t num_cols
    ) : output_(output), values_(values), indices_(indices), 
        input_offsets_(input_offsets), output_offsets_(output_offsets),
        num_dense_output_rows_(num_dense_output_rows),
        num_output_rows_(num_output_rows), num_cols_(num_cols) {}

    void operator()(sycl::nd_item<1> item) const {
        int group_id = item.get_group(0);
        int group_range = item.get_group_range(0);
        int local_id = item.get_local_id(0);
        int local_range = item.get_local_range(0);

        // Map groups to output rows
        for (offset_t dense_output_offset = group_id; 
             dense_output_offset < num_dense_output_rows_; 
             dense_output_offset += group_range) {
            
            // Binary search to find which row this offset belongs to
            // We do it in the first thread of the group and broadcast
            int index_pos;
            if (local_id == 0) {
                index_pos = binary_search_upper_bound(output_offsets_, (int)num_output_rows_, dense_output_offset);
            }
            index_pos = sycl::group_broadcast(item.get_group(), index_pos, 0);
            
            // Calculate offsets
            offset_t prev_offset = (index_pos == 0) ? 0 : output_offsets_[index_pos - 1];
            offset_t rel_index = dense_output_offset - prev_offset;
            
            index_t index = indices_[index_pos];
            offset_t input_start_offset = (index == 0) ? 0 : input_offsets_[index - 1];
            offset_t input_offset = input_start_offset + rel_index;
            
            // Copy columns
            for (int i = local_id; i < num_cols_; i += local_range) {
                output_[dense_output_offset * num_cols_ + i] = values_[input_offset * num_cols_ + i];
            }
        }
    }

private:
    scalar_t* output_;
    const scalar_t* values_;
    const index_t* indices_;
    const offset_t* input_offsets_;
    const offset_t* output_offsets_;
    int64_t num_dense_output_rows_;
    int64_t num_output_rows_;
    int64_t num_cols_;
};

at::Tensor jagged_index_select_2d_forward_xpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t num_dense_output_rows) {
    
    // Checks
    TORCH_CHECK(values.is_xpu(), "values must be XPU tensor");
    TORCH_CHECK(indices.is_xpu(), "indices must be XPU tensor");
    TORCH_CHECK(input_offsets.is_xpu(), "input_offsets must be XPU tensor");
    TORCH_CHECK(output_offsets.is_xpu(), "output_offsets must be XPU tensor");

    TORCH_CHECK(values.dim() == 2, "values must be 2D tensor");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D tensor");
    TORCH_CHECK(input_offsets.dim() == 1, "input_offsets must be 1D tensor");
    TORCH_CHECK(output_offsets.dim() == 1, "output_offsets must be 1D tensor");
    
    auto num_cols = values.size(1);
    auto num_output_rows = indices.size(0);
    
    auto output = at::empty({num_dense_output_rows, num_cols}, values.options());
    
    if (num_dense_output_rows == 0) return output;

    // Launch configuration
    const int64_t max_work_group_size = 256;
    // Use a fixed work group size of 256 for simplicity and good occupancy
    int64_t work_group_size = 256; 
    
    // Number of groups: one per output row, capped at some reasonable number
    // If we have many rows, we loop in the kernel.
    // If we have few rows, we use few groups.
    int64_t num_groups = std::min((int64_t)65535, num_dense_output_rows);
    if (num_groups == 0) num_groups = 1;
    
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    
    // Dispatch macros for types
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "jagged_index_select_2d_forward_xpu", [&] {
        using scalar_t = scalar_t;
        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "jagged_index_select_2d_forward_xpu_indices", [&] {
            using index_t = index_t;
            // Assume offsets are int64_t (long) as is common for offsets
            using offset_t = int64_t; 
            sycl_kernel_submit<JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>>(
                sycl::range<1>(num_groups * work_group_size),
                sycl::range<1>(work_group_size),
                getCurrentSYCLQueue(),
                JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>(
                    output.data_ptr<scalar_t>(),
                    values.data_ptr<scalar_t>(),
                    indices.data_ptr<index_t>(),
                    input_offsets.data_ptr<offset_t>(),
                    output_offsets.data_ptr<offset_t>(),
                    num_dense_output_rows,
                    num_output_rows,
                    num_cols));
        });
    });
    
    return output;
}

} // namespace at::native::xpu
