/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <iostream>
#include "simple_kernel.hpp"

void test_simple_kernel() {
  int numel = 1024;
  float a[1024];

  // a simple sycl kernel
  itoa(a, numel);

  bool success = true;
  for (int i = 0; i < numel; i++) {
    if (a[i] != i) {
      success = false;
      break;
    }
  }

  if (success) {
    std::cout << "Pass" << std::endl;
  } else {
    std::cout << "Fail" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  test_simple_kernel();
  return 0;
}
