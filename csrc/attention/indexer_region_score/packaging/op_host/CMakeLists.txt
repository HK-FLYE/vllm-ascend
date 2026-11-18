file(GLOB IRS_HOST_SOURCES "${IRS_SOURCE_DIR}/op_host/*.cpp")
npu_op_code_gen(
    SRC ${IRS_SOURCE_DIR}/op_host/indexer_region_score_def.cpp
    PACKAGE ${IRS_VENDOR}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    JOIN_OP_DEF True
    COMPILE_OPTIONS -std=c++17 -I${ASCEND_CANN_PACKAGE_PATH}/include/op_common
                    -include${CMAKE_CURRENT_SOURCE_DIR}/../cann_compat.h
)
npu_op_library(cust_opapi ACLNN ${IRS_SOURCE_DIR}/op_host/op_api/aclnn_indexer_region_score.cpp)
# The native API calls l0op::Contiguous/ViewCopy from the CANN operator API.
# Keep this dependency on the native library rather than the PyTorch adapter.
find_library(IRS_CANN_OPAPI_LIBRARY NAMES opapi
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH)
if(NOT IRS_CANN_OPAPI_LIBRARY)
    message(FATAL_ERROR "The CANN SDK must provide lib64/libopapi.so")
endif()
target_link_libraries(cust_opapi PRIVATE ${IRS_CANN_OPAPI_LIBRARY} graph c_sec)
target_include_directories(cust_opapi PRIVATE ${IRS_SOURCE_DIR}/op_host/op_api)
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)
file(GLOB IRS_PROTO_SOURCES ${ASCEND_AUTOGEN_PATH}/op_proto.cc ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
set_source_files_properties(${IRS_PROTO_SOURCES} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH ${IRS_HOST_SOURCES} ${IRS_PROTO_SOURCES})
file(GLOB IRS_FALLBACK_SOURCES ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${IRS_FALLBACK_SOURCES} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING ${IRS_HOST_SOURCES} ${IRS_FALLBACK_SOURCES})
npu_op_package_add(${IRS_VENDOR} LIBRARY cust_optiling cust_opapi cust_op_proto)
# ASC creates the final shared library from the ACLNN object target above.
set(IRS_OPAPI_SHARED_TARGET ${IRS_VENDOR}_ascendc_cust_opapi)
target_link_options(${IRS_OPAPI_SHARED_TARGET} PRIVATE -Wl,--no-undefined)
set_target_properties(${IRS_OPAPI_SHARED_TARGET} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
