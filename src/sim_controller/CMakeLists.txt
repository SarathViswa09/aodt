add_library(sim_controller
  lib/aerial_live.cpp
  lib/aerial_stage.cpp
  lib/simulation_state.cpp
  lib/asim_loop.cpp
  lib/load_from_stage.cpp
  lib/store_to_stage.cpp
  lib/cnt_utils.cpp
  lib/asim_cuda_checks.cpp
  lib/real_cuda_compile_and_runtime_env.cpp
  lib/time_sample.cpp
)

set_property(TARGET sim_controller PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(sim_controller
  aerial_emsolver
  aerial_mm
  aerial_omniclient
  #ar js trace work pxOsd hdx
  usd_arch usd_gf usd_kind usd_pcp usd_plug usd_sdf usd_tf usd_usd usd_usdGeom usd_usdSkel usd_usdShade usd_vt usd_hd usd_usdImaging usd_usdLux usd_usdUtils
  boost_python310
  dl pthread
  omniclient
  omniverse_connection
  CUDA::cudart
  CUDA::nvtx3
  omni_usd_resolver
  omni_utils
  clickhouse
  shared_headers
  be_ctrl
  channel_predictor
  logger
  pycuphycpp
  gsl-lite
)

target_include_directories(sim_controller
  PUBLIC
  api
  ${PACKMAN_BUILD_DIR}/target-deps/omni_client_library/include
  ${PACKMAN_BUILD_DIR}/target-deps/nlohmann-json/include
  ${msgpack_SOURCE_DIR}/include
  ${PACKMAN_BUILD_DIR}/target-deps/usd/${USD_BUILD_TYPE_DIR}/include
  ${PACKMAN_BUILD_DIR}/target-deps/omni_client_library/${USD_BUILD_TYPE_DIR}/include
  ${PACKMAN_BUILD_DIR}/target-deps/omni_usd_resolver/include
  ${Python_INCLUDE_DIRS}
  ${AERIAL_EMSOLVER_INC_DIR}
  ${PROJECT_SOURCE_DIR}/src_be/bindings
  ${PROJECT_SOURCE_DIR}/src_be/controller/include
  ${PACKMAN_BUILD_DIR}/target-deps/tinytoml/include
  ${PROJECT_SOURCE_DIR}/external/cuBB/pyaerial/pybind11)

target_link_directories(sim_controller
  PUBLIC
  ${PACKMAN_BUILD_DIR}/linux-x86_64/${USD_BUILD_TYPE_DIR})

target_compile_definitions(sim_controller
  PRIVATE
  # avoid compiler warnings in some boost includes in usd
  -DBOOST_ALLOW_DEPRECATED_HEADERS)

target_compile_options(sim_controller PRIVATE
  -Wno-deprecated
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_compile_definitions(sim_controller PRIVATE
  "AODT_SIM_CUDA_ARCHITECTURE=\"${AODT_SIM_CUDA_ARCHITECTURE}\"")
