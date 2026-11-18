npu_op_kernel_options(ascendc_kernels ALL OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels SRC_BASE ${IRS_SOURCE_DIR}/op_kernel TILING_LIBRARY cust_optiling)
npu_op_package_add(${IRS_VENDOR} LIBRARY ascendc_kernels)
