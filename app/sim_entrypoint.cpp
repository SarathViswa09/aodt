#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include "stage_utils.hpp"
#include "nlohmann/json.hpp"

#include "OmniUsdResolver.h"
#include "LiveSessionInfo.h"
#include "LiveSessionConfigFile.h"
#include "shared_types.hpp"
#include "omni_messenger.hpp"
#include "connector_state.hpp"
#include "logger.hpp"
#include "clickhouse.hpp"
#include "aerial_live.hpp"
#include "aerial_emsolver_api.h"
#include "load_from_stage.hpp"
#include "simulation_state.hpp"
#include "store_to_stage.hpp"
#include "asim_loop.hpp"
#ifdef AERIALSIM_NVTX_ON
#include <nvtx3/nvToolsExt.h>
#include "cuda_profiler_api.h"
#endif

aerial::sim::db_settings dbs{};
std::string du_asset_path{};
std::string ru_asset_path{};
std::string ue_asset_path{};
std::string spawn_zone_asset_path{};
std::string panel_asset_path{};
std::vector<std::string> opt_in_tables{};
bool opened_at_least_one_scene{};
bool started_python_interpreter{};

// Defined in sim_main.cpp
void send_progress_func(float progress);
OmniClientRequestId get_channel_id();
aerial::sim::simulation_state *get_main_state();
void set_main_state(aerial::sim::simulation_state *state);
void delete_main_state();

constexpr auto limited_to_one_freq_for_now = 1;

[[nodiscard]] bool
is_main_state_valid(aerial::sim::simulation_state *main_state) {
  if (main_state == nullptr) {
    LOG(WARNING)
        << "Connector main state was null.  User needs to load a scene first.";
  }

  return main_state != nullptr;
}

namespace aerial::sim::mm {

void finalize_python_interpreter() {
  if (started_python_interpreter) {
    LOG(DEBUG) << "Thread-ID:" << std::this_thread::get_id()
               << ", Shutting down Python interpreter..." << std::endl;
    pybind11::finalize_interpreter();
    started_python_interpreter = false;
  }
}

auto handle_attach_worker_request(const asim::attach_worker_request &wrq,
                                  void *user_data) {
  LOG(DEBUG) << "got worker request from";
  return asim::attach_worker_reply{.versions_compatible = true};
}

auto handle_attach_worker_decision(const asim::attach_worker_decision &wd,
                                   void *user_data) {
  LOG(DEBUG) << "got worker decision";
  return asim::ready_for_sim_config{.ready = true};
}

auto handle_detach_worker_request(const asim::detach_worker_request &dwq,
                                  void *user_data) {
  LOG(DEBUG) << "got detach worker request";
  return asim::detach_worker_reply{.detach_success = true};
}

auto handle_write_db_and_assets_request(
    const asim::write_db_and_assets_request &wdar, void *user_data) {
  LOG(DEBUG) << "got write db and assets request";
  dbs.name = wdar.db_name;
  dbs.port = wdar.db_port;
  dbs.address = wdar.db_host;
  du_asset_path = wdar.du_asset_path;
  ru_asset_path = wdar.ru_asset_path;
  ue_asset_path = wdar.ue_asset_path;
  spawn_zone_asset_path = wdar.spawn_zone_asset_path;
  panel_asset_path = wdar.panel_asset_path;

  const bool invalid_db_fields =
      dbs.name.empty() || dbs.port == 0 || dbs.address.empty();
  const bool invalid_assets =
      wdar.ru_asset_path.empty() || wdar.du_asset_path.empty() ||
      wdar.ue_asset_path.empty() || wdar.spawn_zone_asset_path.empty() ||
      wdar.panel_asset_path.empty();

  if (invalid_db_fields) {
    LOG(WARNING) << "Some database settings were empty (db_host: "
                 << dbs.address << ", db_port:" << dbs.port
                 << ", db_name: " << dbs.name << ")";
  }

  if (invalid_assets) {
    LOG(WARNING) << "Some asset settings were empty "
                 << "\n\tru_asset_path: " << wdar.ru_asset_path
                 << "\n\tdu_asset_path: " << wdar.du_asset_path
                 << "\n\tue_asset_path:" << wdar.ue_asset_path
                 << "\n\tspawn_zone_asset_path: " << wdar.spawn_zone_asset_path
                 << "\n\tpanel_asset_path: " << wdar.panel_asset_path;
  }

  if (invalid_assets || invalid_db_fields) {
    return asim::write_db_and_assets_reply{.update_db_success = false,
                                           .update_assets_success = false};
  }

  LOG(INFO) << "Connecting to clickhouse @ " << dbs.address.c_str();
  try {
    if (createDatabaseAndTables(dbs) != 0) {
      LOG(ERROR) << "Error creating clickhouse database and tables.";
      return asim::write_db_and_assets_reply{.update_db_success = false,
                                             .update_assets_success = false};
    }
  } catch (std::system_error &e) {
    LOG(ERROR)
        << "Error with clickhouse database: " << e.what() << ".\n"
        << "Please check that clickhouse is running with 'clickhouse status'.\n"
        << "If no server is running, you can start one with 'clickhouse "
           "start'.";
    return asim::write_db_and_assets_reply{.update_db_success = false,
                                           .update_assets_success = false};
  }

  std::vector<db_info> dbv{
      db_info{.scene_url = std::string(),
              .scene_timestamp = std::string(),
              .db_author = wdar.db_author,
              .db_notes = wdar.db_notes,
              .db_timestamp = wdar.db_timestamp,
              .db_schemas_version = "1.2.0",
              .db_content = {},
              .du_asset_path = du_asset_path,
              .ru_asset_path = ru_asset_path,
              .ue_asset_path = ue_asset_path,
              .spawn_zone_asset_path = spawn_zone_asset_path,
              .panel_asset_path = panel_asset_path,
              .opt_in_tables = opt_in_tables}};
  int32_t ret = 0;
  if ((ret = clearTable(dbs, "db_info")) != 0) {
    return asim::write_db_and_assets_reply{.update_db_success = false,
                                           .update_assets_success = false};
  }
  if ((ret = insert_db_info(dbs, dbv)) != 0) {
    return asim::write_db_and_assets_reply{.update_db_success = false,
                                           .update_assets_success = false};
  }

  return asim::write_db_and_assets_reply{.update_db_success = true,
                                         .update_assets_success = true};
}

auto handle_open_scene_request(const asim::open_scene_request &osr,
                               void *user_data) {
  LOG(DEBUG) << "got open scene request";
  if (dbs.address.empty() || dbs.name.empty()) {
    LOG(WARNING) << "did not get db host and db name";
    return asim::open_scene_reply{.open_success = false};
  }

  // Before opening a new stage, the main state must first be deleted,
  // thereby zeroing the reference count of any previous stages
  // (UsdStageRefPtr). Otherwise if the user deletes the live session in between
  // opening stages, the worker and UI live sessions will be out of sync.
  if (get_main_state()) {
    delete_main_state();
  }

  auto stage = pxr::UsdStage::Open(osr.scene_url);
  if (!stage) {
    LOG(ERROR) << "Failure to open stage: " << osr.scene_url << ".\n"
               << "Please check your connection to the nucleus server and that "
               << "you've properly set the OMNI_USER and OMNI_PASS environment "
                  "variables.";
    return asim::open_scene_reply{.open_success = false};
  } else {
    LOG(INFO) << "Omniverse USD Stage Traversal: " << osr.scene_url
              << std::endl;
    addSimpleDUAndRU(stage);
    stage->GetRootLayer()->Save();
  }

  LiveSessionInfo live_session_info_root(osr.scene_url.c_str());
  int32_t ret = 0;
  if ((ret = findOrCreateSession(stage, live_session_info_root,
                                 "_" + osr.live_session_name, "root")) != 0) {
    return asim::open_scene_reply{.open_success = false};
  }

  omniClientLiveWaitForPendingUpdates();

  // We allow the interpreter to be initialized outside the main state setup.
  // We don't want the lifetime of the interpreter tied to the lifetime of the
  // main state as it is not guaranteed that we can safely restart the
  // interpreter after destroying it. Note we need to start the interpreter in
  // the same thread in which it is used, hence we cannot simply start it at the
  // beginning of main.
  if (!started_python_interpreter) {
    constexpr auto init_signal_handlers = false;
    LOG(DEBUG) << "Thread-ID:" << std::this_thread::get_id()
               << ", Initialize Python interpreter." << std::endl;
    pybind11::initialize_interpreter(init_signal_handlers);
    started_python_interpreter = true;
  }
  auto result = aerial::sim::cnt::setupMainState(stage);
  if (result.first == nullptr || result.second != 0) {
    return asim::open_scene_reply{.open_success = false};
  }

  opened_at_least_one_scene = true;
  set_main_state(result.first);
  get_main_state()->requestId = get_channel_id();
  get_main_state()->ue_asset = ue_asset_path;
  db_settings settings{
      .address = dbs.address, .name = dbs.name, .port = dbs.port};
  get_main_state()->db().set_settings(settings);

  return asim::open_scene_reply{.open_success = stage != nullptr};
}

auto handle_db_update_request(const asim::db_update_request &dur,
                              void *user_data) {
  LOG(DEBUG) << "got db update request";

  dbs.name = dur.db_name;
  dbs.port = dur.db_port;
  dbs.address = dur.db_host;
  opt_in_tables = dur.opt_in_tables;

  const bool invalid_db_fields =
      dbs.name.empty() || dbs.port == 0 || dbs.address.empty();
  if (invalid_db_fields) {
    LOG(WARNING) << "Some database settings were empty (db_host: "
                 << dbs.address << ", db_port:" << dbs.port
                 << ", db_name: " << dbs.name << ")";
    return asim::db_update_reply{.update_success = false};
  }

  if (opt_in_tables.empty()) {
    opt_in_tables = {"cfrs", "cirs", "raypaths", "telemetry",
                     "training_result"};
    LOG(WARNING) << "No tables specified in `opt_in_tables`. Defaulting to "
                    "include all tables.";
  }

  LOG(INFO) << "Connecting to clickhouse @ " << dbs.address.c_str();
  try {
    if (createDatabaseAndTables(dbs) != 0) {
      LOG(ERROR) << "Error creating clickhouse database and tables.";
      return asim::db_update_reply{.update_success = false};
    }
  } catch (std::system_error &e) {
    LOG(ERROR)
        << "Error with clickhouse database: " << e.what() << ".\n"
        << "Please check that clickhouse is running with 'clickhouse status'.\n"
        << "If no server is running, you can start one with 'clickhouse "
           "start'.";
    return asim::db_update_reply{.update_success = false};
  }

  if (opened_at_least_one_scene && is_main_state_valid(get_main_state())) {
    LOG(DEBUG) << "Setting DBName to " << dbs.name;
    db_settings settings{
        .address = dbs.address, .name = dbs.name, .port = dbs.port};
    get_main_state()->db().set_settings(settings);
    get_main_state()->db().set_opt_in_tables(opt_in_tables);
  }
  std::vector<db_info> dbv{db_info{
      .scene_url = dur.scene_url,
      .scene_timestamp = dur.scene_timestamp,
      .db_author = dur.db_author,
      .db_notes = dur.db_notes,
      .db_timestamp = dur.db_timestamp,
      .db_schemas_version = "1.2.0",
      .db_content = {},
      .du_asset_path = du_asset_path,
      .ru_asset_path = ru_asset_path,
      .ue_asset_path = ue_asset_path,
      .spawn_zone_asset_path = spawn_zone_asset_path,
      .panel_asset_path = panel_asset_path,
      .opt_in_tables = opt_in_tables,
  }};
  int32_t ret = 0;
  if ((ret = clearTable(dbs, "db_info")) != 0) {
    return asim::db_update_reply{.update_success = false};
  }
  if ((ret = insert_db_info(dbs, dbv)) != 0) {
    return asim::db_update_reply{.update_success = false};
  }

  return asim::db_update_reply{.update_success = true};
}

auto handle_mobility_request(const asim::mobility_request &mrq,
                             void *user_data) {
  LOG(DEBUG) << "got mobility request";
  const auto valid = is_main_state_valid(get_main_state());
  if (valid) {
    int32_t ret = 0;
    if ((ret = aerial::sim::cnt::handler_mobility(get_main_state())) != 0) {
      return asim::mobility_reply{.mobility_success = false};
    }
  }
  return asim::mobility_reply{.mobility_success = valid};
}

auto handle_start_sim_request(const asim::start_sim_request &ssr,
                              void *user_data) {
  LOG(DEBUG) << "got start sim request";
  bool success = is_main_state_valid(get_main_state());
  if (success) {
#ifdef AERIALSIM_NVTX_ON
    cudaProfilerStart();
    auto simulation_range = nvtxRangeStartA("aodt_profile");
#endif
    if (get_main_state()->get_state() == aerial::sim::simulation_state::init) {
      int32_t ret = 0;
      if ((ret = aerial::sim::cnt::handler_mobility(get_main_state())) != 0) {
        return asim::start_sim_reply{.start_success = false};
      }
    }

    aerial::sim::cnt::getRANSimAttribute(get_main_state());
    if (get_main_state()->simulate_ran) {
      success = aerial::sim::cnt::handler_play_full_sim(
                    get_main_state(), send_progress_func) == 0;
    } else {
      success = aerial::sim::cnt::handler_play(get_main_state(),
                                               send_progress_func) == 0;
    }
#ifdef AERIALSIM_NVTX_ON
    nvtxRangeEnd(simulation_range);
    cudaProfilerStop();
#endif
  }
  return asim::start_sim_reply{.start_success = success};
}

auto handle_pause_sim_request(const asim::pause_sim_request &psr,
                              void *user_data) {
  LOG(DEBUG) << "got pause sim request";
  const auto valid = is_main_state_valid(get_main_state());
  if (valid) {
    if (aerial::sim::cnt::handler_pause(get_main_state()) != 0) {
      return asim::pause_sim_reply{.pause_success = false};
    }
  }
  return asim::pause_sim_reply{.pause_success = valid};
}

auto handle_stop_sim_request(const asim::stop_sim_request &ssr,
                             void *user_data) {
  LOG(DEBUG) << "got stop sim request";
  const auto valid = is_main_state_valid(get_main_state());
  if (valid) {
    if (aerial::sim::cnt::handler_stop(get_main_state()) != 0) {
      return asim::stop_sim_reply{.stop_success = false};
    }
  }
  return asim::stop_sim_reply{.stop_success = valid};
}

void handle_sync_cleanup() {
  LOG(DEBUG)
      << "Handling Finalize Python Interpreter as part of sync-thread cleanup";
  aerial::sim::mm::finalize_python_interpreter();
}

[[nodiscard]] emsolver::AntennaPattern
to_emsolver_pattern(const asim::antenna_pattern &pattern) {
  emsolver::AntennaPattern emsolverPattern;
  emsolverPattern.pattern_type = pattern.pattern_type;

  if (!pattern.ampls_theta_complex.empty()) {
    emsolverPattern.ampls_theta.resize(limited_to_one_freq_for_now);
    auto ampl_thetas = pattern.ampls_theta_complex.at(0);
    const int num_samples = ampl_thetas.size();
    emsolverPattern.ampls_theta.at(0).resize(num_samples);

    for (int i = 0; i < num_samples; i++) {
      emsolverPattern.ampls_theta.at(0).at(i) = emsolver::d_complex(
          std::get<0>(ampl_thetas.at(i)), std::get<1>(ampl_thetas.at(i)));
    }
  }

  if (!pattern.ampls_phi_complex.empty()) {
    emsolverPattern.ampls_phi.resize(limited_to_one_freq_for_now);
    auto ampl_phis = pattern.ampls_phi_complex.at(0);
    const int num_samples = ampl_phis.size();
    emsolverPattern.ampls_phi.at(0).resize(num_samples);

    for (int i = 0; i < num_samples; i++) {
      emsolverPattern.ampls_phi.at(0).at(i) = emsolver::d_complex(
          std::get<0>(ampl_phis.at(i)), std::get<1>(ampl_phis.at(i)));
    }
  }
  return emsolverPattern;
}

[[nodiscard]] float round_to_decimals(float value, int decimals = 3) {
  float multiplier = std::pow(10.0f, decimals);
  return std::round(value * multiplier) / multiplier;
}

[[nodiscard]] asim::antenna_pattern
to_asim_pattern(const emsolver::AntennaPattern &pattern) {
  asim::antenna_pattern asim_pattern;
  asim_pattern.pattern_type = pattern.pattern_type;

  if (!pattern.ampls_theta.empty()) {
    asim_pattern.ampls_theta_complex.resize(limited_to_one_freq_for_now);
    auto ampl_thetas = pattern.ampls_theta.at(0);
    const int num_samples = ampl_thetas.size();
    asim_pattern.ampls_theta_complex.at(0).resize(num_samples);

    for (int i = 0; i < num_samples; i++) {
      asim_pattern.ampls_theta_complex.at(0).at(i) =
          std::make_tuple(round_to_decimals(ampl_thetas.at(i).real()),
                          round_to_decimals(ampl_thetas.at(i).imag()));
    }
  }

  if (!pattern.ampls_phi.empty()) {
    asim_pattern.ampls_phi_complex.resize(limited_to_one_freq_for_now);
    auto ampl_phis = pattern.ampls_phi.at(0);
    const int num_samples = ampl_phis.size();
    asim_pattern.ampls_phi_complex.at(0).resize(num_samples);

    for (int i = 0; i < num_samples; i++) {
      asim_pattern.ampls_phi_complex.at(0).at(i) =
          std::make_tuple(round_to_decimals(ampl_phis.at(i).real()),
                          round_to_decimals(ampl_phis.at(i).imag()));
    }
  }

  return asim_pattern;
}

[[nodiscard]] emsolver::AntennaPanel
to_emsolver_panel(const asim::antenna_panel &panel) {
  emsolver::AntennaPanel emsolverPanel;
  emsolverPanel.panel_name = panel.panel_name;
  emsolverPanel.antenna_names = panel.antenna_names;
  emsolverPanel.antenna_pattern_indices = panel.antenna_pattern_indices;
  if (!panel.frequencies.empty()) {
    emsolverPanel.frequencies.resize(limited_to_one_freq_for_now);
    emsolverPanel.frequencies.at(0) =
        panel.frequencies.at(0) * 1.0e6; // convert MHz to Hz
  }
  emsolverPanel.thetas = panel.thetas; // radians
  emsolverPanel.phis = panel.phis;     // radians
  emsolverPanel.reference_freq =
      panel.reference_freq_mhz * 1.0e6; // convert MHz to Hz
  emsolverPanel.dual_polarized = panel.dual_polarized;
  emsolverPanel.num_loc_antenna_horz = panel.num_loc_antenna_horz;
  emsolverPanel.num_loc_antenna_vert = panel.num_loc_antenna_vert;
  emsolverPanel.antenna_spacing_horz =
      panel.antenna_spacing_horz_mm / 10.0; // convert mm to cm
  emsolverPanel.antenna_spacing_vert =
      panel.antenna_spacing_vert_mm / 10.0; // convert mm to cm
  emsolverPanel.antenna_roll_angle_first_polz =
      panel.antenna_roll_angle_first_polz_rad;
  emsolverPanel.antenna_roll_angle_second_polz =
      panel.antenna_roll_angle_second_polz_rad;
  return emsolverPanel;
}

auto handle_antenna_panels_request(const asim::antenna_panels_request &apr,
                                   void *user_data) {
  LOG(DEBUG) << "got antenna panels request";
  const auto valid = is_main_state_valid(get_main_state());
  auto &antenna_info = get_main_state()->get_aerial_stage().antenna_info;
  antenna_info.patterns.clear(); // clear patterns from previous runs
  for (const auto &pattern : apr.patterns) {
    antenna_info.patterns.emplace_back(to_emsolver_pattern(pattern));
  }
  antenna_info.panels.clear(); // clear panels from previous runs
  for (const auto &panel : apr.panels) {
    antenna_info.panels.emplace_back(to_emsolver_panel(panel));
  }
  return asim::antenna_panels_reply{.valid_antenna_data = valid};
}

auto handle_calculate_active_element_patterns_request(
    const asim::calculate_active_element_patterns_request &req,
    void *user_data) {
  LOG(DEBUG) << "got calculate mutual coupling request";
  const auto panel = to_emsolver_panel(req.panel);
  const auto panel_name = panel.panel_name;

  auto em_patterns = emsolver::compute_mutual_coupling_patterns(panel);

  std::vector<asim::antenna_pattern> asim_patterns(em_patterns.size());
  std::transform(em_patterns.begin(), em_patterns.end(), asim_patterns.begin(),
                 to_asim_pattern);
  return asim::calculate_active_element_patterns_reply{
      .panel_name = panel_name, .patterns = asim_patterns};
}

} // namespace aerial::sim::mm
