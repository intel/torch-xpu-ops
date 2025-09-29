@echo off
set "PYTORCH_EXTRA_INSTALL_REQUIREMENTS=intel-cmplr-lib-rt==2025.2.1 ^^^
^^^| intel-cmplr-lib-ur==2025.2.1 ^^^
^^^| intel-cmplr-lic-rt==2025.2.1 ^^^
^^^| intel-sycl-rt==2025.2.1 ^^^
^^^| oneccl-devel==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' ^^^
^^^| oneccl==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' ^^^
^^^| impi-rt==2021.16.1; platform_system == 'Linux' and platform_machine == 'x86_64' ^^^
^^^| onemkl-sycl-blas==2025.2.0 ^^^
^^^| onemkl-sycl-dft==2025.2.0 ^^^
^^^| onemkl-sycl-lapack==2025.2.0 ^^^
^^^| onemkl-sycl-rng==2025.2.0 ^^^
^^^| onemkl-sycl-sparse==2025.2.0 ^^^
^^^| dpcpp-cpp-rt==2025.2.1 ^^^
^^^| intel-opencl-rt==2025.2.1 ^^^
^^^| mkl==2025.2.0 ^^^
^^^| intel-openmp==2025.2.1 ^^^
^^^| tbb==2022.2.0 ^^^
^^^| tcmlib==1.4.0 ^^^
^^^| umf==0.11.0 ^^^
^^^| intel-pti==0.13.1"

echo %PYTORCH_EXTRA_INSTALL_REQUIREMENTS%