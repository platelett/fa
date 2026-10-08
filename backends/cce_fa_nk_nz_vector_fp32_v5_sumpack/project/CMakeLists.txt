cmake_minimum_required(VERSION 3.16)
project(fa_fa_nk_nz_vector_fp32_v5_sumpack_project)
find_package(ASC REQUIRED)
set(ASCEND_COMPUTE_UNIT ascend910_93)
set(package_name fa_fa_nk_nz_vector_fp32_v5_sumpack)
npu_op_package(${package_name} TYPE SHARED CONFIG
    INSTALL_PATH ${CMAKE_BINARY_DIR}
    ENABLE_BINARY_PACKAGE TRUE)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
set_target_properties(${package_name} PROPERTIES OUTPUT_NAME fa_fa_nk_nz_vector_fp32_v5_sumpack)
