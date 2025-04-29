# ***************************************************************************
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# **************************************************************************

cmake_minimum_required(VERSION 3.21)
project(chapi LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
include(FetchContent)

if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "Project is top level.  Fetching external dependencies pybind11 and clickhouse-cpp.")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1)
    FetchContent_MakeAvailable(pybind11)

    FetchContent_Declare(
        clickhouse-cpp-lib
        GIT_REPOSITORY https://github.com/ClickHouse/clickhouse-cpp
        GIT_TAG        v2.5.1)

    FetchContent_MakeAvailable(clickhouse-cpp-lib)
    set_property(TARGET clickhouse-cpp-lib PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
    FetchContent_Declare(pybind11 SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/pybind11")
    FetchContent_MakeAvailable(pybind11)
endif()

set(SRC chapi.cpp)
set(LIB chapi_bindings)
pybind11_add_module(${LIB} ${SRC})
set_property(TARGET ${LIB} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${LIB}
    PRIVATE
    clickhouse
    aerial_mm
    aerial_emsolver
    shared_headers
    logger
    CUDA::cudart
)

target_compile_options(chapi_bindings
  PRIVATE
  -Wno-deprecated
)

# OPTIONAL install since if this does not exist, we dont fail.
# Install of aodt_sim will fail if building only aodt_sim w/o building
# chapi_bindings and we end up with cmake install phase error.
# Making it OPTIONAL, will not fail if this is missing, as
# the existence of it is anyway optional.
install(TARGETS chapi_bindings clickhouse-cpp-lib
        DESTINATION ${CMAKE_BINARY_DIR}/examples/install/
        OPTIONAL
)
