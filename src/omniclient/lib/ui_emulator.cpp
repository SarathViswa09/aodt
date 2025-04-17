/***************************************************************************
 # Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto. Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
**************************************************************************/

#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include "connector_state.hpp"
#include "omni_messenger.hpp"
#include "logger.hpp"

namespace {
struct ui_config final {
  // nucleus
  std::string username = "sarath2";
  std::string nucleus_server = "omniverse://omniverse-server";
  std::string broadcast_channel_url = nucleus_server + "/broadcast";
  std::string private_channel_url = nucleus_server + "/channel";

  // database
  std::string db_author = username;
  std::string db_host = "clickhouse";
  std::string db_name = "sarath_final_apr_10";
  std::string db_notes = "simulation ran from ui emulator";
  uint32_t db_port = 9000;
  const std::vector<std::string> opt_in_tables = {
      "cfrs", "cirs", "raypaths", "telemetry", "training_result"};

  // assets
  std::string du_asset_path = "omniverse://omniverse-server/Users/aerial/assets/du.usda";
  std::string panel_asset_path = "omniverse://omniverse-server/Users/aerial/assets/panel_properties.usda";
  std::string ru_asset_path = "omniverse://omniverse-server/Users/aerial/assets/gnb.usda";
  std::string spawn_zone_asset_path = "omniverse://omniverse-server/Users/aerial/assets/spawn_zone.usda";
  std::string ue_asset_path = "omniverse://omniverse-server/Users/aerial/assets/ue.usda";

  // scene
  std::string scene_url = "omniverse://omniverse-server/Users/aerial/plateau/kyoto.usd";
  std::string live_session_name = "asim-session_sarath2_live_session";
};

struct ui_lifecycle final {
  bool worker_attached{};
  bool ready_for_sim_config{};
  bool wrote_db_and_assets{};
  bool opened_scene{};
  bool detached_worker{};
  bool updated_mobility{};
  bool updated_db{};
  bool started_sim{};
  bool paused_sim{};
  bool stopped_sim{};
  bool sim_finished{};
  bool received_antenna_reply{};
  bool calculated_mutual_coupling{};
  bool antenna_panel_reply{};
};

bool fatal_error{};
ui_lifecycle lc{};
ui_config cfg{};
std::unique_ptr<asim::asim_state_machine> sm;

[[nodiscard]] auto omni_message_timestamp() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S");
  return ss.str();
}

asim::attach_worker_decision
handle_attach_worker_reply(const asim::attach_worker_reply &wry,
                           void *user_data) {
  LOG(DEBUG) << "got worker reply (versions_compatible="
             << wry.versions_compatible << ')';
  if (!wry.versions_compatible) {
    LOG(ERROR) << "attach worker versions incompatible";
    fatal_error = true;
  }

  auto ret = sm->join_private_channel(cfg.private_channel_url);
  if (ret < 0) {
    LOG(ERROR) << "join private channel failed with return code " << ret;
    fatal_error = true;
  }

  lc.worker_attached = true;
  return asim::attach_worker_decision{.worker_accepted = true,
                                      .versions_compatible = true,
                                      .omni_channel_url =
                                          cfg.private_channel_url};
}

void handle_ready_for_sim_config(const asim::ready_for_sim_config &rfc,
                                 void *user_data) {
  LOG(DEBUG) << "got ready for config (ready=" << rfc.ready << ')';
  if (!rfc.ready) {
    LOG(ERROR) << "worker is not ready for sim config";
    fatal_error = true;
  }

  lc.ready_for_sim_config = rfc.ready;
}

void handle_write_db_and_assets_reply(
    const asim::write_db_and_assets_reply &wdar, void *user_data) {
  lc.wrote_db_and_assets = wdar.update_assets_success && wdar.update_db_success;
  LOG(DEBUG) << "got write db and assets reply (success="
             << lc.wrote_db_and_assets << ')';
  if (!lc.wrote_db_and_assets) {
    LOG(ERROR) << "worker failed to write db and assets";
    fatal_error = true;
  }
}

void handle_open_scene_reply(const asim::open_scene_reply &osr,
                             void *user_data) {
  LOG(DEBUG) << "got open scene reply (success=" << osr.open_success << ')';
  if (!osr.open_success) {
    LOG(ERROR) << "worker failed to open scene";
    fatal_error = true;
  }
  lc.opened_scene = osr.open_success;
}

void handle_detach_worker_reply(const asim::detach_worker_reply &dwy,
                                void *user_data) {
  LOG(DEBUG) << "got detach worker reply (success=" << dwy.detach_success
             << ')';
  if (!dwy.detach_success) {
    LOG(ERROR) << "worker failed to detach";
    fatal_error = true;
  }
  lc.detached_worker = dwy.detach_success;
}

void handle_mobility_reply(const asim::mobility_reply &mry, void *user_data) {
  LOG(DEBUG) << "got mobility reply (success=" << mry.mobility_success << ')';
  if (!mry.mobility_success) {
    LOG(ERROR) << "worker failed to update mobility";
    fatal_error = true;
  }
  lc.updated_mobility = mry.mobility_success;
}

void handle_db_update_reply(const asim::db_update_reply &dur, void *user_data) {
  LOG(DEBUG) << "got db update reply (success=" << dur.update_success << ')';
  if (!dur.update_success) {
    LOG(ERROR) << "worker failed to update db";
    fatal_error = true;
  }
  lc.updated_db = dur.update_success;
}

void handle_sim_progress_update(const asim::sim_progress_update &spu,
                                void *user_data) {
  LOG(DEBUG) << "got sim progress update (progress=" << spu.progress << ')';
  lc.sim_finished = spu.progress == -1;
}

void handle_start_sim_reply(const asim::start_sim_reply &ssr, void *user_data) {
  LOG(DEBUG) << "got start sim reply (successs=" << ssr.start_success << ')';
  if (!ssr.start_success) {
    LOG(ERROR) << "worker failed to start sim";
    fatal_error = true;
  }
  lc.started_sim = ssr.start_success;
}

void handle_pause_sim_reply(const asim::pause_sim_reply &psr, void *user_data) {
  LOG(DEBUG) << "got pause sim reply (success=" << psr.pause_success << ')';
  if (!psr.pause_success) {
    LOG(ERROR) << "worker failed to pause sim";
    fatal_error = true;
  }
  lc.paused_sim = psr.pause_success;
}

void handle_stop_sim_reply(const asim::stop_sim_reply &ssr, void *user_data) {
  LOG(DEBUG) << "got stop sim reply (success=" << ssr.stop_success << ')';
  if (!ssr.stop_success) {
    LOG(ERROR) << "worker failed to stop sim";
    fatal_error = true;
  }
  lc.stopped_sim = ssr.stop_success;
}

auto handle_heartbeat(const asim::heartbeat &hb, void *user_data) {
  LOG(DEBUG) << "got heartbeat (count=" << hb.count
             << ", periodicity_in_seconds=" << hb.periodicity_in_seconds << ')';
  return asim::heartbeat_reply{hb.count};
}

void handle_connector_log(const asim::connector_log &cl, void *user_data) {
  LOG(DEBUG) << "got connector log: " << cl.level << " " << cl.body;
}


void handle_calculate_active_element_patterns_reply(
    const asim::calculate_active_element_patterns_reply &rep, void *user_data) {
  LOG(DEBUG) << "got calculate mutual coupling reply (" << rep.patterns.size()
             << " patterns)";
  lc.calculated_mutual_coupling = true;
}

void handle_antenna_panels_reply(const asim::antenna_panels_reply &apr,
                                 void *user_data) {
  LOG(DEBUG) << "got antenna panels reply (valid_antenna_data="
             << apr.valid_antenna_data << ')';
  lc.received_antenna_reply = true; // antenna data is an optional parameter
}

int cleanup_app(const int32_t omni_handle) {
  asim::deregister_log_settings();
  asim::omni_messenger_destroy(omni_handle);
  sm->stop();
  return fatal_error ? EXIT_FAILURE : EXIT_SUCCESS;
}
} // namespace

int main() {
  const asim::state_handlers sh{
      .on_attach_worker_reply = handle_attach_worker_reply,
      .on_ready_for_sim_config = handle_ready_for_sim_config,
      .on_detach_worker_reply = handle_detach_worker_reply,
      .on_write_db_and_assets_reply = handle_write_db_and_assets_reply,
      .on_open_scene_reply = handle_open_scene_reply,
      .on_antenna_panels_reply = handle_antenna_panels_reply,
      .on_calculate_active_element_patterns_reply =
          handle_calculate_active_element_patterns_reply,
      .on_mobility_reply = handle_mobility_reply,
      .on_sim_progress_update = handle_sim_progress_update,
      .on_db_update_reply = handle_db_update_reply,
      .on_start_sim_reply = handle_start_sim_reply,
      .on_pause_sim_reply = handle_pause_sim_reply,
      .on_stop_sim_reply = handle_stop_sim_reply,
      .on_heartbeat = handle_heartbeat,
      .on_connector_log = handle_connector_log,
  };

  constexpr auto show_func_line = false;
  constexpr auto min_callback_level = WARNING;
  constexpr auto min_log_level = DEBUG;
  asim::register_log_settings(show_func_line, min_log_level, min_callback_level,
                              {});

  std::atomic_bool error_flag{};
  const auto handle = asim::omni_messenger_initialize(&error_flag);
  if (handle < 0) {
    LOG(ERROR) << "Failed to initialize omni messenger (ret=" << handle << ')';
    std::exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Attempting to connect to nucleus url: "
            << cfg.broadcast_channel_url;
  sm = std::make_unique<asim::asim_state_machine>(asim::app_type::ui, sh,
                                                  cfg.broadcast_channel_url);
  auto bid = sm->broadcast_channel_id();
  LOG(DEBUG) << "join channel returned " << bid;

  using namespace std::chrono_literals;
  constexpr auto wait = 1s;
  constexpr auto poll = 100ms;
  std::this_thread::sleep_for(wait);

  asim::attach_worker_request wr{};
  auto rid = sm->send_message(
      bid, asim::connector_message::attach_worker_request, nlohmann::json(wr));
  LOG(DEBUG) << "send attach_worker_request returned " << rid;
  while (!sm->private_channel_id().has_value()) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }

  auto ts = omni_message_timestamp();
  asim::write_db_and_assets_request wdar{
      .db_host = cfg.db_host,
      .db_port = cfg.db_port,
      .db_name = cfg.db_name,
      .db_author = cfg.db_author,
      .db_notes = cfg.db_notes,
      .db_timestamp = ts,
      .du_asset_path = cfg.du_asset_path,
      .ru_asset_path = cfg.ru_asset_path,
      .ue_asset_path = cfg.ue_asset_path,
      .spawn_zone_asset_path = cfg.spawn_zone_asset_path,
      .panel_asset_path = cfg.panel_asset_path};
  auto pid = sm->private_channel_id().value();
  rid = sm->send_message(pid,
                         asim::connector_message::write_db_and_assets_request,
                         nlohmann::json(wdar));
  LOG(DEBUG) << "send write_db_and_assets_request returned " << rid;
  while (!lc.ready_for_sim_config) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }

  asim::db_update_request dur{
      .scene_url = cfg.scene_url,
      .scene_timestamp = ts,
      .db_host = cfg.db_host,
      .db_port = cfg.db_port,
      .db_name = cfg.db_name,
      .db_author = cfg.db_author,
      .db_notes = cfg.db_notes,
      .db_timestamp = wdar.db_timestamp,
      .opt_in_tables = cfg.opt_in_tables,
  };
  rid = sm->send_message(pid, asim::connector_message::db_update_request,
                         nlohmann::json(dur));
  LOG(DEBUG) << "send db_update_request returned " << rid;
  while (!lc.updated_db) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }

  asim::open_scene_request osr{.scene_url = cfg.scene_url,
                               .live_session_name = cfg.live_session_name};
  rid = sm->send_message(pid, asim::connector_message::open_scene_request,
                         nlohmann::json(osr));
  LOG(DEBUG) << "send open_scene_request returned " << rid;
  while (!lc.opened_scene) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }


  asim::antenna_panels_request apr{
    .panels = {
        {
            .panel_name = "panel_01",
            .antenna_names = {"halfwave_dipole","halfwave_dipole","halfwave_dipole",  "halfwave_dipole"},
            .antenna_pattern_indices = {0, 0, 0, 0},
            .frequencies = {3600.0f},
            .thetas = {0.0f},
            .phis = {0.0f},
            .reference_freq_mhz = 3600.0,
            .dual_polarized = true,
            .num_loc_antenna_horz = 2,
            .num_loc_antenna_vert = 1,
            .antenna_spacing_horz_mm = 45.8,
            .antenna_spacing_vert_mm = 45.8,
            .antenna_roll_angle_first_polz_rad = 0.0,
            .antenna_roll_angle_second_polz_rad = 1.570796
        }
    },
    .patterns = {
        {
            .pattern_type = 2,
        }
    }
  };
  rid = sm->send_message(pid, asim::connector_message::antenna_panels_request,
                         nlohmann::json(apr));
    LOG(DEBUG) << "!!!!send antenna panel req!!!!" << rid;
    while (!lc.received_antenna_reply) {
        std::this_thread::sleep_for(poll);
        if (fatal_error) {
          return cleanup_app(handle);
        }
    }

  asim::start_sim_request tsr{};
  rid = sm->send_message(pid, asim::connector_message::start_sim_request,
                         nlohmann::json(tsr));
  LOG(DEBUG) << "send start_sim_request returned " << rid;
  while (!lc.started_sim && !lc.sim_finished) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }


  asim::detach_worker_request dwr{};
  rid = sm->send_message(pid, asim::connector_message::detach_worker_request,
                         nlohmann::json(dwr));
  LOG(DEBUG) << "send detach_worker_request returned " << rid;
  while (!lc.detached_worker) {
    std::this_thread::sleep_for(poll);
    if (fatal_error) {
      return cleanup_app(handle);
    }
  }

  return cleanup_app(handle);
}
