#pragma once

#include <c10/util/Exception.h>
#include <limits>

#define XPU_KERNEL_LOOP_TYPE(item, i, n, index_type)                      \
  int64_t _i_n_d_e_x =                                                    \
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                       \
       _i_n_d_e_x += item.get_local_range(0) * item.get_group_range(0),   \
                  i = _i_n_d_e_x)

#define XPU_KERNEL_LOOP(item, i, n) XPU_KERNEL_LOOP_TYPE(item, i, n, int)
