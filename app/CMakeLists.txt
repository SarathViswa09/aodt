# ***************************************************************************
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from NVIDIA CORPORATION is strictly prohibited.
# **************************************************************************

add_executable(aodt_sim sim_main.cpp sim_entrypoint.cpp ConfigLoader.cpp stage_utils.cpp)

# To avoid refactoring code and docs that rely on the location of the main binary
# Change it to be in the same location as the original one, which is the root of the project.
# Either ${CMAKE_BINARY_DIR} or ${aerial_ov_optix_BINARY_DIR} which are identical in this case.
set_target_properties(aodt_sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(aodt_sim PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(aodt_sim PRIVATE aerial_omniclient sim_controller ${Boost_LIBRARIES} Backward::Interface)

target_compile_definitions(
  aodt_sim PRIVATE # avoid compiler warnings in some boost includes in usd
                   -DBOOST_ALLOW_DEPRECATED_HEADERS)

target_compile_options(
  aodt_sim
  PRIVATE -Wno-deprecated $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
