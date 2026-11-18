cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_indexer_native LANGUAGES CXX)

find_package(ASC REQUIRED)
get_filename_component(IRS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(IRS_VENDOR vllm_ascend_indexer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnnop
)
add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/cann_compat.h)
npu_op_package(${IRS_VENDOR} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}/)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
