/***************************************************************************
 # Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto. Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#include "mobility_api.hpp"
#include "load_from_stage.hpp"
#include "store_to_stage.hpp"
#include "cnt_utils.hpp"
#include "time_sample.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "pxr/usd/usdGeom/xformCache.h"
#include "pxr/usd/usdGeom/xformOp.h"
#include "aerial_stage.hpp"
#include "aerial_emsolver_api.h"
#include <OmniClient.h>
#include "aerial_live.hpp"
#include "cnt_utils.hpp"
#include <mutex>
#include <unordered_map>
#include <chrono>
#include "clickhouse.hpp"
#include "database.hpp"
#include "omni_messenger.hpp"
#include "trainer.hpp"
#include "be_ctrl.hpp"
using namespace aerialsim;

namespace {

// Returns bounding box information, including vertices and bounding box min/max
// in local coordinates. We transform to local coordinates, instead of world
// coordinates, so determining if vertices lie within the bounding box becomes a
// trivial comparison to the bounding box extents. Note the user may have
// deleted the spawn zone prim, in which case this function returns a
// std::nullopt and automatic generation of UEs does not obey a spawn zone.
std::optional<aerial::sim::mm::bounding_box_info>
get_spawn_zone_bounding_box(pxr::UsdStageRefPtr stage,
                            const std::vector<float3> &world_vertices) {
  auto sz = stage->GetPrimAtPath(pxr::SdfPath("/World/spawn_zone"));
  if (!sz) {
    LOG(DEBUG) << "Could not get node /World/spawn_zone.";
    return std::nullopt;
  }

  pxr::VtVec3fArray extent{};
  auto success =
      sz.GetAttribute(pxr::TfToken("aerial:spawn_zone:szWorldExtent"))
          .Get(&extent);
  if (!success) {
    LOG(DEBUG) << "Could not get attribute aerial:spawn_zone:szWorldExtent.";
    return std::nullopt;
  }
  aerial::sim::mm::bounding_box_info bbi{};
  bbi.bb_min = make_float3(std::min(extent[0][0], extent[1][0]),
                           std::min(extent[0][1], extent[1][1]),
                           std::min(extent[0][2], extent[1][2]));
  bbi.bb_max = make_float3(std::max(extent[0][0], extent[1][0]),
                           std::max(extent[0][1], extent[1][1]),
                           std::max(extent[0][2], extent[1][2]));
  return bbi;
}

const auto ant_size_based_on_antenna_usage(
    const Config &be_config, const bool use_only_first_antenna_pair,
    const bool dual_polarized_rx, const bool dual_polarized_tx) {
  if (use_only_first_antenna_pair == false) {
    int ue = be_config.ue_panel_el();
    int ru = be_config.gnb_panel_el();
    return std::make_pair(ue, ru);
  } else {
    int ue = dual_polarized_rx ? 2 : 1;
    int ru = dual_polarized_tx ? 2 : 1;
    return std::make_pair(ue, ru);
  }
};

auto umap_to_pair_vector(const std::unordered_map<uint32_t, float> &umap) {
  std::vector<std::pair<uint32_t, float>> vec;
  vec.reserve(umap.size());
  for (const auto [k, v] : umap) {
    vec.emplace_back(k, v);
  }
  return vec;
}

[[nodiscard]] int32_t
add_training_result_to_db(const aerial::sim::db_settings &dbs,
                          const aerial::sim::training_info &ti) {

  if (ti.num_iterations == 0) {
    return 0;
  }

  training_result tr{.time_id = ti.time_id,
                     .batch_id = ti.batch_id,
                     .slot_id = ti.slot_id,
                     .symbol_id = ti.symbol_id,
                     .name = ti.name,
                     .title = ti.title,
                     .y_label = ti.y_label,
                     .x_label = ti.x_label};

  tr.training_losses = umap_to_pair_vector(ti.training_losses);
  tr.validation_losses = umap_to_pair_vector(ti.validation_losses);
  tr.test_losses = umap_to_pair_vector(ti.test_losses);
  tr.baseline_losses = umap_to_pair_vector(ti.baseline_losses);

  std::vector<training_result> trs;
  trs.push_back(tr);
  return insert_training_result(dbs, trs);
}

auto config_to_scenario_info(aerial::sim::mm::config &cfg) {
  aerial::sim::scenario_info si{
      .slot_symbol_mode = cfg.slot_symbol_mode,
      .batches = static_cast<uint32_t>(cfg.batches),
      .slots_per_batch = static_cast<uint32_t>(cfg.slots_per_batch),
      .symbols_per_slot = static_cast<uint32_t>(cfg.samples_per_slot),
      .duration = cfg.duration,
      .interval = cfg.interval,
      .ue_min_speed_mps = cfg.speed.x,
      .ue_max_speed_mps = cfg.speed.y,
      .seeded_mobility = cfg.is_seeded,
      .seed = cfg.seed,
      .scale = cfg.scale,
      .ue_height_m = cfg.height_ue};

  return si;
}

bool consistent_tx_rx_info(const std::vector<emsolver::TXInfo> &tx_info,
                           const std::vector<emsolver::RXInfo> &rx_info) {
  if (tx_info.empty() || rx_info.empty()) {
    return false;
  }

  const auto first_tx_id = tx_info.at(0).tx_ID;
  const auto tx_pol = tx_info.at(0).dual_polarized_antenna ? 2 : 1;
  const auto ants_per_ru = tx_info.at(0).loc_antenna.size() * tx_pol;
  const auto scs = tx_info.at(0).subcarrier_spacing;
  const auto fft_size = tx_info.at(0).fft_size;

  for (const auto &txi : tx_info) {
    const auto this_tx_pol = txi.dual_polarized_antenna ? 2 : 1;
    const auto this_ants_per_ru = txi.loc_antenna.size() * this_tx_pol;
    const auto this_scs = txi.subcarrier_spacing;
    const auto this_fft_size = txi.fft_size;

    if (this_tx_pol != tx_pol) {
      LOG(ERROR) << "Inconsistent RU settings.  "
                 << "ru_id=" << txi.tx_ID << " polarization (" << this_tx_pol
                 << ") != "
                 << "ru_id=" << first_tx_id << " polarization (" << tx_pol
                 << ")";
      return false;
    }

    if (this_ants_per_ru != ants_per_ru) {
      LOG(ERROR) << "Inconsistent RU settings.  "
                 << "ru_id=" << txi.tx_ID << " ants_per_ru ("
                 << this_ants_per_ru << ") != "
                 << "ru_id=" << first_tx_id << " ants_per_ru (" << ants_per_ru
                 << ")";
      return false;
    }

    if (this_scs != scs) {
      LOG(ERROR) << "Inconsistent RU settings.  "
                 << "ru_id=" << txi.tx_ID << " subcarrier_spacing (" << this_scs
                 << ") != "
                 << "ru_id=" << first_tx_id << " subcarrier_spacing (" << scs
                 << ")";
      return false;
    }

    if (this_fft_size != fft_size) {
      LOG(ERROR) << "Inconsistent RU settings.  "
                 << "ru_id=" << txi.tx_ID << " fft_size (" << this_fft_size
                 << ") != "
                 << "ru_id=" << first_tx_id << " fft_size (" << fft_size << ")";
      return false;
    }
  }

  const auto first_rx_id = rx_info.at(0).rx_ID;
  const auto rx_pol = rx_info.at(0).dual_polarized_antenna ? 2 : 1;
  const auto ants_per_ue = rx_info.at(0).loc_antenna.size() * rx_pol;

  for (const auto &rxi : rx_info) {
    const auto this_rx_pol = rxi.dual_polarized_antenna ? 2 : 1;
    const auto this_ants_per_ue = rxi.loc_antenna.size() * this_rx_pol;

    if (this_rx_pol != rx_pol) {
      LOG(ERROR) << "Inconsistent UE settings.  "
                 << "ue_id=" << rxi.rx_ID << " polarization (" << this_rx_pol
                 << ") != "
                 << "ue_id=" << first_rx_id << " polarization (" << rx_pol
                 << ")";
      return false;
    }

    if (this_ants_per_ue != ants_per_ue) {
      LOG(ERROR) << "Inconsistent UE settings.  "
                 << "ue_id=" << rxi.rx_ID << " ants_per_ue ("
                 << this_ants_per_ue << ") != "
                 << "ue_id=" << first_rx_id << " ants_per_ue (" << ants_per_ue
                 << ")";
      return false;
    }
  }

  return true;
}

auto tx_rx_to_ru_ue_info(const std::vector<emsolver::TXInfo> &tx_info,
                         const std::vector<emsolver::RXInfo> &rx_info) {
  const auto rx_pol = rx_info.at(0).dual_polarized_antenna ? 2 : 1;
  const auto tx_pol = tx_info.at(0).dual_polarized_antenna ? 2 : 1;
  const auto ants_per_ue = rx_info.at(0).loc_antenna.size() * rx_pol;
  const auto ants_per_ru = tx_info.at(0).loc_antenna.size() * tx_pol;
  const auto mu = tx_info.at(0).subcarrier_spacing / 15e3;

  aerial::sim::ru_ue_info info{
      .num_ues = static_cast<uint32_t>(rx_info.size()),
      .num_rus = static_cast<uint32_t>(tx_info.size()),
      .ue_pol = static_cast<uint32_t>(rx_pol),
      .ru_pol = static_cast<uint32_t>(tx_pol),
      .ants_per_ue = static_cast<uint32_t>(ants_per_ue),
      .ants_per_ru = static_cast<uint32_t>(ants_per_ru),
      .fft_size = static_cast<uint32_t>(tx_info.at(0).fft_size),
      .numerology = static_cast<uint32_t>(mu)};

  return info;
}

auto tx_rx_users_to_ru_association_info(
    const std::vector<aerial::sim::mm::user> &users,
    const std::vector<std::vector<uint32_t>> &selected_rx_indices,
    const std::vector<emsolver::TXInfo> &tx_info,
    const std::vector<emsolver::RXInfo> &rx_info, const int time_id,
    const double scale) {

  std::vector<aerial::sim::ru_association_info> info;
  info.reserve(selected_rx_indices.size());

  uint32_t ru_idx = 0;
  for (const auto &ue_ids : selected_rx_indices) {
    const auto rid = tx_info.at(ru_idx).tx_ID;
    aerial::sim::ru_association_info rai{.ru_index = ru_idx,
                                         .ru_id = static_cast<uint32_t>(rid)};

    rai.associated_ues.reserve(ue_ids.size());
    for (const auto ue_id : ue_ids) {
      const auto &samples = users.at(ue_id).flattened_samples;
      const auto &sample = samples.at(time_id);

      const auto uid = users.at(ue_id).uID;
      aerial::sim::ue_info ui{.ue_index = ue_id,
                              .ue_id = uid,
                              .position_x = sample.point.x,
                              .position_y = sample.point.y,
                              .position_z = sample.point.z,
                              .speed_mps =
                                  static_cast<float>(sample.speed_mps / scale)};
      rai.associated_ues.emplace_back(ui);
    }

    info.emplace_back(rai);
    ++ru_idx;
  }

  return info;
}

void transition_to_stop_state_if_sim_completed_or_error(
    aerial::sim::simulation_state &state, const int error_code) {
  const auto &users = state.get_mobility_users();
  const auto num_time_steps =
      !users.empty() ? users.at(0).flattened_samples.size() : 0;
  using ss = aerial::sim::simulation_state;
  const bool sim_completed = (state.get_state() == ss::ongoing_sim) &&
                             (state.get_time_idx() == num_time_steps);
  if (sim_completed) {
    std::ignore = state.consume(ss::sim_stop);
  }
  if (error_code != 0) {
    std::ignore = state.consume(ss::sim_stop);
  }
}

[[nodiscard]] auto read_rt_config_from_stage(pxr::UsdStageRefPtr stage,
                                             const bool simulate_ran) {
  constexpr auto default_rays = 500;
  constexpr auto default_interactions = 3;
  constexpr auto default_sphere_radius_meter = 2.0;
  constexpr auto default_diffuse_type = 0; // 0 is Lambertian, 1 is directional
  int num_rays_in_thousands = default_rays;
  int max_interactions = default_interactions;
  float rx_sphere_radius_meter = default_sphere_radius_meter;
  int diffuse_type = default_diffuse_type;

  omniClientLiveWaitForPendingUpdates();

  bool rays_valid = false;
  bool interactions_valid = false;
  bool sphere_radius_valid = false;
  bool diffuse_type_valid = false;
  auto node = stage->GetPrimAtPath(pxr::SdfPath("/Scenario"));
  auto rays_attr = node.GetAttribute(pxr::TfToken("sim:em:rays"));
  if (rays_attr) {
    rays_valid = rays_attr.Get(&num_rays_in_thousands);
  }
  auto interactions_attr =
      node.GetAttribute(pxr::TfToken("sim:em:interactions"));
  if (interactions_attr) {
    interactions_valid = interactions_attr.Get(&max_interactions);
  }
  auto sphere_radius_attr =
      node.GetAttribute(pxr::TfToken("sim:em:sphere_radius"));
  if (sphere_radius_attr) {
    sphere_radius_valid = sphere_radius_attr.Get(&rx_sphere_radius_meter);
  }
  auto diffuse_type_attr =
      node.GetAttribute(pxr::TfToken("sim:em:diffuse_type"));
  if (diffuse_type_attr) {
    diffuse_type_valid = diffuse_type_attr.Get(&diffuse_type);
  }

  if (rays_valid) {
    LOG(DEBUG) << "Found attribute /Scenario.sim:em:rays (value="
               << num_rays_in_thousands << " in thousands)";
  } else {
    num_rays_in_thousands = default_rays;
    LOG(WARNING)
        << "Did not find attribute /Scenario.sim:em:rays so setting to default "
        << num_rays_in_thousands << " (in thousands)";
  }

  if (interactions_valid) {
    LOG(DEBUG) << "Found attribute /Scenario.sim:em:interactions (value="
               << max_interactions << ")";
  } else {
    max_interactions = default_interactions;
    LOG(WARNING) << "Did not find attribute /Scenario.sim:em:interactions so "
                    "setting to default "
                 << max_interactions;
  }

  if (sphere_radius_valid) {
    LOG(DEBUG) << "Found attribute /Scenario.sim:em:sphere_radius (value="
               << rx_sphere_radius_meter * 100 << " centimeters)";
  } else {
    rx_sphere_radius_meter = default_sphere_radius_meter;
    LOG(WARNING) << "Did not find attribute /Scenario.sim:em:sphere_radius so "
                    "setting to default "
                 << rx_sphere_radius_meter * 100 << " (in centimeters)";
  }

  if (diffuse_type_valid) {
    LOG(DEBUG) << "Found attribute /Scenario.sim:em:diffuse_type (value="
               << diffuse_type << ")";
  } else {
    diffuse_type = default_diffuse_type;
    LOG(WARNING) << "Did not find attribute /Scenario.sim:em:diffuse_type so "
                    "setting to default "
                 << diffuse_type;
  }

  emsolver::RTConfig rt_cfg{};
  rt_cfg.max_num_bounces = max_interactions;
  rt_cfg.em_diffuse_type = static_cast<emsolver::EM_DIFFUSE_TYPE>(diffuse_type);
  rt_cfg.use_only_first_antenna_pair = false;
  rt_cfg.calc_tau_mins = false;
  rt_cfg.num_rays_in_thousands = num_rays_in_thousands;
  rt_cfg.rx_sphere_radius_cm =
      rx_sphere_radius_meter * 100; // convert meters to centimeters
  rt_cfg.simulate_ran = simulate_ran;

  return rt_cfg;
}

void emsolver_log_callback(emsolver::EMLogLevel level,
                           const std::string &message) {
  switch (level) {
  case emsolver::EMLogLevel::ERROR:
    LOG(ERROR) << message;
    break;
  case emsolver::EMLogLevel::WARNING:
    LOG(WARNING) << message;
    break;
  case emsolver::EMLogLevel::INFO:
    LOG(INFO) << message;
    break;
  case emsolver::EMLogLevel::DEBUG:
    LOG(DEBUG) << message;
    break;
  case emsolver::EMLogLevel::VERBOSE:
    LOG(VERBOSE) << message;
    break;
  default:
    throw std::invalid_argument("unknown em log level"); // should never happen
  }
}

[[nodiscard]] bool read_enable_training_from_stage(pxr::UsdStageRefPtr stage) {
  auto node = stage->GetPrimAtPath(pxr::SdfPath("/Scenario"));
  auto attr = node.GetAttribute(pxr::TfToken("sim:enable_training"));
  bool enable_training = false;
  bool valid = false;
  if (attr) {
    valid = attr.Get(&enable_training);
  }
  if (valid && attr) {
    LOG(DEBUG) << "Found attribute /Scenario.sim:enable_training (value="
               << enable_training << ")";
  } else {
    LOG(WARNING) << "Did not find attribute /Scenario.sim:enable_training so "
                    "setting to default value of false.";
  }

  return enable_training;
}

template <typename T>
[[nodiscard]] auto read_prim_with_default(const pxr::UsdPrim &prim,
                                          const std::string &attribute_name) {
  auto attr = prim.GetAttribute(pxr::TfToken(attribute_name));

  T val{};
  if constexpr (std::is_same_v<T, std::string>) {
    pxr::TfToken token{};
    if (!attr) {
      LOG(WARNING) << "Could not read attribute " << attribute_name
                   << " from prim " << prim.GetName().GetString()
                   << " so setting to value " << val;
    } else {
      attr.Get(&token);
      val = token.GetString();
    }
  } else {
    if (!attr) {
      LOG(WARNING) << "Could not read attribute " << attribute_name
                   << " from prim " << prim.GetName().GetString()
                   << " so setting to value " << val;
    } else {
      attr.Get(&val);
    }
  }

  return val;
}

[[nodiscard]] int32_t add_scenario_to_db(aerial::sim::simulation_state &state) {

  pxr::UsdStageRefPtr stage = state.get_aerial_stage().int_stage;
  auto node = stage->GetPrimAtPath(pxr::SdfPath("/Scenario"));

  std::vector<scenario> scenarios;
  scenarios.emplace_back(scenario{
      .default_ue_panel =
          read_prim_with_default<std::string>(node, "sim:ue:panel_type"),
      .default_ru_panel =
          read_prim_with_default<std::string>(node, "sim:gnb:panel_type"),
      .num_emitted_rays_in_thousands =
          read_prim_with_default<int32_t>(node, "sim:em:rays"),
      .num_scene_interactions_per_ray =
          read_prim_with_default<int32_t>(node, "sim:em:interactions"),
      .rx_sphere_radius_m =
          read_prim_with_default<float>(node, "sim:em:sphere_radius"),
      .diffuse_type =
          read_prim_with_default<int32_t>(node, "sim:em:diffuse_type"),
      .max_paths_per_ru_ue_pair =
          read_prim_with_default<uint32_t>(node, "pathViz:maxNumPaths"),
      .ray_sparsity =
          read_prim_with_default<int32_t>(node, "pathViz:raysSparsity"),
      .num_batches = read_prim_with_default<int32_t>(node, "sim:batches"),
      .slots_per_batch =
          read_prim_with_default<int32_t>(node, "sim:slots_per_batch"),
      .symbols_per_slot =
          read_prim_with_default<int32_t>(node, "sim:samples_per_slot"),
      .duration = read_prim_with_default<float>(node, "sim:duration"),
      .interval = read_prim_with_default<float>(node, "sim:interval"),
      .enable_wideband_cfrs =
          read_prim_with_default<bool>(node, "sim:enable_wideband"),
      .num_ues = read_prim_with_default<uint32_t>(node, "sim:num_users"),
      .percentage_indoor_ues =
          read_prim_with_default<float>(node, "sim:perc_indoor_procedural_ues"),
      .ue_height = read_prim_with_default<float>(node, "sim:ue:height"),
      .ue_min_speed = read_prim_with_default<float>(node, "sim:ueMinSpeed"),
      .ue_max_speed = read_prim_with_default<float>(node, "sim:ueMaxSpeed"),
      .is_seeded = read_prim_with_default<bool>(node, "sim:is_seeded"),
      .seed = read_prim_with_default<uint32_t>(node, "sim:seed"),
      .simulate_ran = read_prim_with_default<bool>(node, "sim:is_full"),
      .enable_training =
          read_prim_with_default<bool>(node, "sim:enable_training"),
  });

  return insert_scenario(state.db().settings(), scenarios);
}

[[nodiscard]] int32_t add_world_to_db(aerial::sim::simulation_state &state) {
  pxr::UsdStageRefPtr stage = state.get_aerial_stage().int_stage;
  auto buildings_prim = stage->GetPrimAtPath(pxr::SdfPath("/World/buildings"));

  std::vector<world> worlds;
  for (const auto &w : buildings_prim.GetAllChildren()) {
    auto attr = w.GetAttribute(pxr::TfToken("ObjectType"));
    std::string obj_type;
    attr.Get(&obj_type);
    if (obj_type == "building") { // filter out object type edge_cylinder
      worlds.emplace_back(world{
          .prim_path = w.GetName().GetString(),
          .material = read_prim_with_default<std::string>(w, "asim:material"),
          .is_rf_active = read_prim_with_default<bool>(w, "AerialRFMesh"),
          .is_rf_diffuse = read_prim_with_default<bool>(w, "AerialRFDiffuse"),
          .is_rf_diffraction =
              read_prim_with_default<bool>(w, "AerialRFDiffraction"),
          .is_rf_transmission =
              read_prim_with_default<bool>(w, "AerialRFTransmission")});
    }
  }

  auto ground_prim = stage->GetPrimAtPath(pxr::SdfPath("/World/ground_plane"));
  if (!ground_prim) {
    LOG(WARNING) << "Could not read prim /World/ground_plane.";
    return -1;
  }
  worlds.emplace_back(world{
      .prim_path = ground_prim.GetName().GetString(),
      .material =
          read_prim_with_default<std::string>(ground_prim, "asim:material"),
      .is_rf_active = read_prim_with_default<bool>(ground_prim, "AerialRFMesh"),
      .is_rf_diffuse =
          read_prim_with_default<bool>(ground_prim, "AerialRFDiffuse"),
      .is_rf_diffraction =
          read_prim_with_default<bool>(ground_prim, "AerialRFDiffraction"),
      .is_rf_transmission =
          read_prim_with_default<bool>(ground_prim, "AerialRFTransmission")});

  return insert_world(state.db().settings(), worlds);
}

[[nodiscard]] int32_t
add_materials_to_db(aerial::sim::simulation_state &state) {
  pxr::UsdStageRefPtr stage = state.get_aerial_stage().int_stage;
  auto materials_prim =
      stage->GetPrimAtPath(pxr::SdfPath("/Materials/standard"));

  std::vector<material> materials;
  for (const auto &m : materials_prim.GetAllChildren()) {
    materials.emplace_back(material{
        .label = m.GetName().GetString(),
        .itu_r_p2040_a = read_prim_with_default<double>(m, "a"),
        .itu_r_p2040_b = read_prim_with_default<double>(m, "b"),
        .itu_r_p2040_c = read_prim_with_default<double>(m, "c"),
        .itu_r_p2040_d = read_prim_with_default<double>(m, "d"),
        .scattering_xpd = read_prim_with_default<double>(m, "k_xpol"),
        .rms_roughness = read_prim_with_default<double>(m, "roughness_rms"),
        .scattering_coeff =
            read_prim_with_default<double>(m, "scattering_coeff"),
        .exponent_alpha_r =
            read_prim_with_default<int32_t>(m, "exponent_alpha_r"),
        .exponent_alpha_i =
            read_prim_with_default<int32_t>(m, "exponent_alpha_i"),
        .lambda_r = read_prim_with_default<double>(m, "lambda_r"),
        .thickness_m = read_prim_with_default<double>(m, "thickness"),
    });
  }

  return insert_materials(state.db().settings(), materials);
}

[[nodiscard]] int32_t
add_time_info_to_db(aerial::sim::simulation_state &state) {

  const std::vector<aerial::sim::mm::user> &users = state.get_mobility_users();
  if (users.empty()) {
    return 0;
  }

  const auto &time_steps = users.at(0).flattened_samples;
  if (time_steps.empty()) {
    return 0;
  }

  const auto num_time_steps = time_steps.size();
  if (num_time_steps == 0) {
    return 0;
  }

  const auto slots_per_batch = state.get_mm_config().slots_per_batch;
  const auto samples_per_slot = state.get_mm_config().samples_per_slot;
  const auto batches = state.get_mm_config().batches;
  const auto slot_symbol_mode = state.get_mm_config().slot_symbol_mode;

  std::vector<time_info> time_infos;
  time_infos.reserve(num_time_steps);

  for (uint32_t t = 0; t < num_time_steps; ++t) {
    const auto batch =
        asim::time_index_to_batch(t, slot_symbol_mode, slots_per_batch,
                                  samples_per_slot, num_time_steps, batches);
    const auto slot = asim::time_index_to_slot_within_batch(
        t, slot_symbol_mode, slots_per_batch, samples_per_slot, num_time_steps,
        batches);
    const auto symbol = asim::time_index_to_sample_within_slot(
        t, slot_symbol_mode, slots_per_batch, samples_per_slot, num_time_steps,
        batches);

    time_infos.emplace_back(
        time_info{.time = t, .batch = batch, .slot = slot, .symbol = symbol});
  }

  return insert_time_info(state.db().settings(), time_infos);
}
} // namespace

namespace aerial {
namespace sim {
namespace cnt {
void send_progress_task(std::atomic<float> &progress,
                        std::function<void(float)> send_progress_func) {
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto curr_progress = progress.load();
    if (send_progress_func) {
      send_progress_func(curr_progress);
    }
    if (curr_progress < 0.0f) {
      break;
    }
  }
}
//==================================================================================================================================================
static bool prepareFeConfig(aerial::sim::simulation_state *state,
                            const bool enable_training, FeConfig &feConfig) {

  MATX_NVTX_START("prepareFeConfig", matx::MATX_NVTX_LOG_USER);
  const auto &mm_cfg = state->get_mm_config();
  const int org_samples_per_slot = mm_cfg.samples_per_slot;

  const std::vector<aerial::sim::mm::user> &users = state->get_mobility_users();

  if (!(state->get_mm_config().samples_per_slot == 1 ||
        state->get_mm_config().samples_per_slot == 14)) {
    LOG(ERROR) << "Samples per slot value to be either 1 or 14";
    return false;
  }

  auto &aerial_stage = state->get_aerial_stage();
  auto &tx_info = aerial_stage.tx_info;
  auto &rx_info = aerial_stage.rx_info;
  auto &ru_du_info_map = aerial_stage.ru_du_info_map;

  feConfig.numTx = tx_info.size();
  feConfig.numRx = rx_info.size();
  float pwr = tx_info[0].radiated_power;
  for (auto &i : tx_info) {
    if (i.radiated_power != pwr) {
      LOG(ERROR) << "RAN simulation does not support heterogeneous RU power "
                    "levels. All RUs should have same radiated power";
      return false;
    }
    feConfig.cids_be_ui_mapping.emplace_back(i.tx_ID);
  }

  pwr = rx_info[0].radiated_power;
  for (auto &i : rx_info) {
    if (i.radiated_power != pwr) {
      LOG(ERROR) << "RAN simulation does not support heterogeneous UE power "
                    "levels. All UEs should have same radiated power";
      return false;
    }
    feConfig.uids_be_ui_mapping.emplace_back(i.rx_ID);
  }

  for (auto &user : users) {
    feConfig.bler_target.emplace_back(user.params.bler_target);
  }

  feConfig.duRuAssoc.clear();
  for (auto iter = aerial_stage.du_info_map.begin();
       iter != aerial_stage.du_info_map.end(); iter++) {
    auto temp = iter->second;
    if (!temp.tx_info_idx.empty()) {
      feConfig.duRuAssoc.emplace_back(temp.tx_info_idx);
      feConfig.duids_be_ui_mapping.emplace_back(temp.id);
    }
  }

  feConfig.numSlots = mm_cfg.slots_per_batch;
  feConfig.subcarrierSpacing = tx_info[0].subcarrier_spacing;
  feConfig.frequencyCenter = tx_info[0].carrier_freq;
  feConfig.gnbPower = tx_info[0].radiated_power;
  feConfig.uePower = rx_info[0].radiated_power;
  feConfig.ue_panel_nh = rx_info[0].loc_antenna.size();
  feConfig.ue_panel_nv = rx_info[0].dual_polarized_antenna ? 2 : 1;
  feConfig.gnb_panel_nh = tx_info[0].loc_antenna.size();
  feConfig.gnb_panel_nv = tx_info[0].dual_polarized_antenna ? 2 : 1;
  feConfig.fftSize = tx_info[0].fft_size;

  if (feConfig.ue_panel_nh * feConfig.ue_panel_nv != 4 ||
      (feConfig.gnb_panel_nh * feConfig.gnb_panel_nv != 4 &&
       feConfig.gnb_panel_nh * feConfig.gnb_panel_nv != 64)) {
    LOG(ERROR) << "Check antenna configuration for RAN Simulation. Only 4T4R "
                  "supported for UE, only 4T4R/64T64R supported for RU";
    return false;
  }

  if ((feConfig.fftSize != MAX_FFT_SIZE_CONST) ||
      (feConfig.subcarrierSpacing != 30000)) {
    LOG(ERROR) << "RAN Simulation supports only for 100 MHz, FFT Size of 4096 "
                  "and 30 kHz subcarrier spacing, but received FFT Size "
               << feConfig.fftSize << " and SCS "
               << feConfig.subcarrierSpacing / 1000.0 << " kHz.";
    return false;
  }
  feConfig.gnbBw = 1e8;

  feConfig.enable_training = enable_training;
  feConfig.samples_per_slot = org_samples_per_slot;

  return true;
}

ASIM_EXPORT int32_t
handler_play_full_sim(aerial::sim::simulation_state *state,
                      std::function<void(float)> send_progress_func) {
  MATX_NVTX_START("handler_play_full_sim", matx::MATX_NVTX_LOG_USER);
  int32_t ret = 0;
  int insert_cirs_ret = 0;
  int insert_raypaths_ret = 0;
  int insert_cfrs_ret = 0;
  auto init_state = state->get_state();
  std::vector<std::shared_ptr<bool>> thread_running;
  std::vector<std::thread> ray_write_threads;
  constexpr int MAX_OUTSTANDING_DB_THREADS = 10;
  auto sim_start = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  // handle sim progress
  std::atomic<float> progressValue(0.0f);
  std::thread progressThread;

  if (init_state == aerial::sim::simulation_state::mobility_available ||
      init_state == aerial::sim::simulation_state::paused_sim) {
    if (init_state == aerial::sim::simulation_state::mobility_available) {
      if (state->update_stage_sim_params() != 0) {
        ret = -1;
        goto cleanup;
      }
      if (state->load_deployed_ues() != 0) {
        ret = -1;
        goto cleanup;
      }
      if (state->load_deployed_rus() != 0) {
        ret = -1;
        goto cleanup;
      }

      // clear all tables except db_info
      if (clear_sim_results_from_db(state->db().settings()) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (insertUEs(state->db().settings(), state->get_mobility_users(),
                    state->get_mm_config().scale) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_time_info_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_scenario_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_materials_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      // if (add_world_to_db(*state) != 0) {
      //   ret = -1;
      //   goto cleanup;
      // }
    }

    emsolver::RTConfig rt_cfg = read_rt_config_from_stage(
        state->get_aerial_stage().int_stage, state->simulate_ran);

    AerialStage &aerial_stage = state->get_aerial_stage();

    if (aerial_stage.tx_info.empty()) {
      LOG(ERROR) << "No RUs found.  Please deploy at least one RU and re-run "
                    "the simulation.";
      ret = -1;
      goto cleanup;
    }
    if (aerial_stage.rx_info.empty()) {
      LOG(ERROR) << "No UEs found.  Please deploy at least one UE and re-run "
                    "the simulation.";
      ret = -1;
      goto cleanup;
    }

    const bool enable_training =
        read_enable_training_from_stage(aerial_stage.int_stage);

    if (enable_training &&
        !consistent_tx_rx_info(aerial_stage.tx_info, aerial_stage.rx_info)) {
      ret = -1;
      goto cleanup;
    }

    LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
        logGPUMemUsage("[Before EM solver setup]"));
    std::unique_ptr<emsolver::AerialEMSolver> aerial_emsolver;
    try {
      aerial_emsolver = std::make_unique<emsolver::AerialEMSolver>(
          aerial_stage.tx_info, aerial_stage.rx_info, aerial_stage.antenna_info,
          aerial_stage.geometry_info, rt_cfg, state->getStream(0));
    } catch (const std::runtime_error &e) {
      LOG(ERROR) << "Problem initializing EM solver: " << e.what() << ".\n"
                 << "Recommend checking your scenario settings and GPU memory "
                    "usage on compute node.";
      ret = -1;
      goto cleanup;
    }
    LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
        aerial_emsolver->registerLogCallback(emsolver_log_callback));
    LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
        logGPUMemUsage("[After EM solver setup]"));

    const std::vector<aerial::sim::mm::user> &users =
        state->get_mobility_users();

    getMobilityParams(state);
    auto mm_cfg = state->get_mm_config();
    const int org_samples_per_slot = mm_cfg.samples_per_slot;

    BECtrl beCtrl;
    FeConfig feConfig;
    if (prepareFeConfig(state, enable_training, feConfig)) {
      LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.setup(feConfig));
      auto status =
          insertRanConfig(state->db().settings(), beCtrl.getRanConfigDb());
      if (status != 0) {
        LOG(ERROR) << "Error returned by insertRanConfig: " << status;
        ret = -1;
        goto cleanup;
      }
    } else {
      ret = -1;
      goto cleanup;
    }
    LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
        logGPUMemUsage("[After RAN backend setup]"));

    LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
        state->consume(aerial::sim::simulation_state::sim_play));

    const bool write_cfrs =
        state->db().is_opt_in_table(aerial::sim::db_data::table_name::CFRS);
    const bool write_cirs =
        state->db().is_opt_in_table(aerial::sim::db_data::table_name::CIRS);
    const bool write_raypaths =
        state->db().is_opt_in_table(aerial::sim::db_data::table_name::RAYPATHS);
    const bool write_telemetry = state->db().is_opt_in_table(
        aerial::sim::db_data::table_name::TELEMETRY);
    const bool write_training_result = state->db().is_opt_in_table(
        aerial::sim::db_data::table_name::TRAINING_RESULT);

    LOG(INFO) << "Opt-in DB tables: write_cfrs=" << write_cfrs
              << ", write_cirs=" << write_cirs
              << ", write_raypaths=" << write_raypaths
              << ", write_telemetry=" << write_telemetry
              << ", write_training_result=" << write_training_result;

    if (!users.empty()) {
      const std::vector<aerial::sim::mm::sample> &time_steps =
          users.at(0).flattened_samples;

      if (!time_steps.empty()) {
        if (aerial_stage.tx_info.empty()) {
          LOG(ERROR) << "No RUs found.  Please deploy at least one RU and "
                        "re-run the simulation.";
          ret = -1;
          goto cleanup;
        }
        if (aerial_stage.rx_info.empty()) {
          LOG(ERROR) << "No UEs found.  Please deploy at least one UE and "
                        "re-run the simulation.";
          ret = -1;
          goto cleanup;
        }
        bool rt_cell_association_ctrl =
            true; // SET TO TRUE to enable cell association step (tracing for
                  // all links)
        const bool calc_tau_mins_ctrl =
            true; // SET TO TRUE to enable calculating min propagation delays
                  // (tau_mins)

        progressThread = std::thread([&progressValue, &send_progress_func]() {
          send_progress_task(progressValue, send_progress_func);
        });

        // initial rt_cell_association_state
        bool rt_cell_association_state =
            rt_cell_association_ctrl &&
            mm_cfg.slot_symbol_mode; // only do cell-association step in
                                     // slot/symbol mode

        // set of selected txs and associated rxs for EMSolver call for each
        // time sample
        auto &selected_tx_indices = state->selected_tx_indices;
        auto &selected_rx_indices = state->selected_rx_indices;
        selected_tx_indices.resize(aerial_stage.tx_info.size());
        std::iota(selected_tx_indices.begin(), selected_tx_indices.end(), 0);

        std::vector<emsolver::d_complex *> d_all_CFR_results(
            selected_tx_indices.size(), nullptr);

        std::vector<float *> d_all_tau_mins(selected_tx_indices.size(),
                                            nullptr);

        while ((state->get_state() ==
                aerial::sim::simulation_state::ongoing_sim) &&
               (state->get_time_idx() < time_steps.size())) {
          auto slot_start =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch());

          const auto sample = state->get_time_idx();
          const auto num_time_steps = time_steps.size();
          auto batch = asim::time_index_to_batch(
              sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
              mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          const auto slot = asim::time_index_to_slot_within_batch(
              sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
              mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          const auto sample_within_slot =
              asim::time_index_to_sample_within_slot(
                  sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
                  mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
              logGPUMemUsage("[slot " + std::to_string(slot) + " start]"));

          const int num_samples_per_batch =
              mm_cfg.slot_symbol_mode
                  ? mm_cfg.slots_per_batch * mm_cfg.samples_per_slot
                  : num_time_steps / mm_cfg.batches;

          rt_cfg.calc_tau_mins = calc_tau_mins_ctrl;

          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.setBatch(batch));
          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.setSlot(slot));

          // SELECT RUs AND UEs LISTS
          /* Updation of selected_rx_indices -
              For init cell association - all rx/ues should be considered
              For sample which is also slot boundary - Ues which need periodic
             CFR updates should be considered For other samples within slot -
             selected_rx_indices should remain unchanged Allocate memory for
             CFRs once selected_rx_indices is updated */
          if (state->get_time_idx() % num_samples_per_batch == 0 &&
              rt_cell_association_state) {
            MATX_NVTX_START("asim_loop:first_sample_first_slot",
                            matx::MATX_NVTX_LOG_USER);
            // first sample of the first slot
            // reset to all RUs and UEs for cell association
            selected_rx_indices.clear();
            selected_rx_indices.resize(selected_tx_indices.size());
            for (int selected_tx_idx = 0;
                 selected_tx_idx < selected_tx_indices.size();
                 selected_tx_idx++) {
              selected_rx_indices[selected_tx_idx].resize(
                  aerial_stage.rx_info.size());
              std::iota(selected_rx_indices[selected_tx_idx].begin(),
                        selected_rx_indices[selected_tx_idx].end(), 0);
            }

            // only first antenna pair for this sample
            rt_cfg.use_only_first_antenna_pair = true;
            mm_cfg.samples_per_slot = 1;
            rt_cfg.calc_tau_mins =
                false; // TAU info is not required for init cell association
            {
              MATX_NVTX_START("asim_loop::allocateDeviceMemForResults",
                              matx::MATX_NVTX_LOG_USER);
              LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                  aerial_emsolver->allocateDeviceMemForResults(
                      aerial_stage.tx_info, aerial_stage.rx_info,
                      selected_tx_indices, selected_rx_indices, rt_cfg,
                      mm_cfg.samples_per_slot, d_all_CFR_results,
                      d_all_tau_mins));
            }
            if (batch) { // for batch=0, beCtrl.setup would do the same
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.batchBoundaryCleanup());
            }
          } else if (sample_within_slot == 0) { // slot boundary
            selected_rx_indices.clear();
            selected_rx_indices.resize(selected_tx_indices.size());

            rt_cfg.use_only_first_antenna_pair =
                beCtrl.use_single_antenna_pair_for_periodic_cfrs();

            mm_cfg.samples_per_slot = 1;
            rt_cfg.calc_tau_mins =
                false; // TAU info is not required for periodic CFR update UEs

            CUDA_CHECK_LAST_ERROR();

            if (beCtrl.periodicCfrUpdateRequired()) {
              {
                MATX_NVTX_START("asim_loop::ueListForPeriodicCfrUpdate",
                                matx::MATX_NVTX_LOG_USER);
                LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                    beCtrl.ueListForPeriodicCfrUpdate(selected_rx_indices));
              }
              {
                MATX_NVTX_START("asim_loop::allocateDeviceMemForResults",
                                matx::MATX_NVTX_LOG_USER);
                LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                    aerial_emsolver->allocateDeviceMemForResults(
                        aerial_stage.tx_info, aerial_stage.rx_info,
                        selected_tx_indices, selected_rx_indices, rt_cfg,
                        mm_cfg.samples_per_slot, d_all_CFR_results,
                        d_all_tau_mins));
              }

              CUDA_CHECK_LAST_ERROR();
            }
          }

          if (selected_tx_indices.size() == 0) {
            LOG(ERROR) << "No RUs found!";
            ret = -1;
            goto cleanup;
          }

          std::string wb_str =
              aerial_stage.tx_info[0].fft_size > 1
                  ? " (wideband CFRs enabled, fft_size=" +
                        std::to_string(aerial_stage.tx_info[0].fft_size) + ")"
                  : " (wideband CFRs disabled)";
          if (rt_cell_association_state) {
            LOG(INFO) << "Computing all links for cell association for batch "
                      << batch << wb_str << "...";
          } else {
            if (mm_cfg.slot_symbol_mode) {
              LOG(INFO) << "Computing sample=" << (sample + 1) << "/"
                        << num_time_steps << ", batch=" << batch
                        << ", slot=" << slot
                        << ", sample_within_slot=" << sample_within_slot
                        << wb_str;
            } else {
              LOG(INFO) << "Computing sample=" << (sample + 1) << "/"
                        << num_time_steps << ", batch=" << batch << wb_str;
            }
          }

          // update aerial_stage's rx_info
          if (state->update_ues() != 0) {
            ret = -1;
            goto cleanup;
          }
          std::vector<emsolver::RayPath> *all_ray_path_results =
              new std::vector<emsolver::RayPath>();
          auto all_CFR_results =
              new std::vector<std::vector<emsolver::d_complex>>{
                  selected_tx_indices.size()};

          /* EM Solver is called here under 3 different scenarios -
1. rt_cell_association_state is true, selected_rx_indices contains all UEs and
d_all_CFR_results is allocated for all UEs, 1 symbol
2. if slot bounday, selected_rx_indices contains UEs for periodic CFR update and
d_all_CFR_results is allocated for perodic CFR update for 1 symbol.
selected_rx_indices will be empty if Periodic CFR update is not scheduled in
this slot
3. else selected_rx_indices contains UEs selected for scheduling and
d_all_CFR_results is allocated for scheduled UEs for 14 symbols In case no UE is
selected for scheduling (possibly due to unassoicated UEs), selected_rx_indices
will be empty TAU will be false for scenario #1 and #2 and will be decided by
calc_tau_mins_ctrl for scenario #3 */

          // NB: incase the selected_rx_indices are not yet sorted by the
          // corresponding rx_IDs, they will be returned sorted by runEMSolver
          if (selected_rx_indices[0].size() == 0) {
            /* There are 2 conditions in which no UEs may be in the
            selected_rx_indices list -
            1. Slot boundary and peridoic CFR update is not scheduled in this
            slot
            2. 14 symbol slot and No UEs selected in this slot (potentially due
            to no UEs associated) */
            LOG(VERBOSE) << "selected_rx_indices is empty - (slot boundary && "
                            "no periodic CFR updates) || (14 symbol slot && no "
                            "UE selected)";
          } else {
            // init cell association or periodic cfrs
            LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(aerial_emsolver->runEMSolver(
                state->get_time_idx(), aerial_stage.tx_info,
                aerial_stage.rx_info, aerial_stage.antenna_info,
                selected_tx_indices, selected_rx_indices, rt_cfg,
                sample_within_slot, mm_cfg.samples_per_slot,
                *all_ray_path_results, d_all_CFR_results, d_all_tau_mins));
          }

          if (rt_cell_association_state) { // batch boundary - scenario #1
            MATX_NVTX_START("asim_loop::batch_boundary_scenario_1",
                            matx::MATX_NVTX_LOG_USER);
            std::vector<matx::tensor_t<aerialsim::ChannelsValueType, 5>> cfr;

            /* CFR memory dimension for each tx is for cell association using
            single antennal pair is rx_indices[tx_idx].size() *
            mm_cfg.samples_per_slot *
            txi.fft_size *
            (rx0.dual_polarized_antenna?2:1) *
            (txi.dual_polarized_antenna?2:1) */
            LOG_AND_GOTO_CLEANUP_IF_FALSE(selected_rx_indices[0].size() ==
                                          beCtrl.getNumUe());
            LOG_AND_GOTO_CLEANUP_IF_FALSE(mm_cfg.samples_per_slot == 1);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(aerial_stage.tx_info[0].fft_size ==
                                          beCtrl.be_config.fft_size);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.rx_info[0].dual_polarized_antenna
                    ? 2
                    : 1 == beCtrl.be_config.ue_panel_nv);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.tx_info[0].dual_polarized_antenna
                    ? 2
                    : 1 == beCtrl.be_config.gnb_panel_nv);
            {
              MATX_NVTX_START("asim_loop::make_boundary_tensor",
                              matx::MATX_NVTX_LOG_USER);
              for (uint32_t i = 0; i < aerial_stage.tx_info.size(); i++) {

                LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                    auto region = beCtrl.DeviceRegion();
                    cfr.push_back(region.make_boundary_tensor<
                                  aerialsim::ChannelsValueType>(
                        d_all_CFR_results[i],
                        {static_cast<int>(
                             selected_rx_indices[i].size()), // num UEs
                         1,                                  // symbol
                         beCtrl.be_config.fft_size,
                         aerial_stage.rx_info[0].dual_polarized_antenna ? 2 : 1,
                         aerial_stage.tx_info[0].dual_polarized_antenna
                             ? 2
                             : 1})););
              }
            }
            cudaDeviceSynchronize();

            /* // dump channel
            {
                std::string ofile = "FE_CH_assoc.mat";
                auto pb = std::make_unique<matx::detail::MatXPybind>();
                auto np = py::module_::import("numpy");
                auto sp = py::module_::import("scipy.io");
                auto td = py::dict{};
                for (int c = 0; c < cfr.size(); c++) {

                    auto np_ten =
            pb->TensorViewToNumpy<matx::tensor_t<aerialsim::ChannelsValueType,
            5>>(cfr[c]); td[std::to_string(c).c_str()] = np_ten;
                }
                auto obj = sp.attr("savemat")("file_name"_a =  ofile, "mdict"_a
            = td);
            } */
            {
              MATX_NVTX_START("asim_loop::initCellAssociation",
                              matx::MATX_NVTX_LOG_USER);
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.initCellAssociation(cfr));
            }
          } else if (sample_within_slot == 0) { // slot boundary
            MATX_NVTX_START("asim_loop::slot_boundary",
                            matx::MATX_NVTX_LOG_USER);
            LOG(DEBUG) << "slot boundary : batch=" << batch << " slot=" << slot
                       << " sample_within_slot=" << sample_within_slot
                       << " , samples_per_slot=" << org_samples_per_slot;

            /* CFR memory dimension for each tx is for periodic cell association
            (using all antenna pairs) is selected_rx_indices[i].size() *
            mm_cfg.samples_per_slot = 1 *
            txi.fft_size *
            rx0.loc_antenna.size() *
            txi.loc_antenna.size() *
            (rx0.dual_polarized_antenna?2:1) *
            (txi.dual_polarized_antenna?2:1) */

            LOG_AND_GOTO_CLEANUP_IF_FALSE(mm_cfg.samples_per_slot == 1);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(aerial_stage.tx_info[0].fft_size ==
                                          beCtrl.be_config.fft_size);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.rx_info[0].loc_antenna.size() *
                    (aerial_stage.rx_info[0].dual_polarized_antenna ? 2 : 1) ==
                beCtrl.be_config.ue_panel_el());
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.tx_info[0].loc_antenna.size() *
                    (aerial_stage.tx_info[0].dual_polarized_antenna ? 2 : 1) ==
                beCtrl.be_config.gnb_panel_el());

            if (selected_rx_indices[0].size()) {
              std::vector<matx::tensor_t<aerialsim::ChannelsValueType, 5>>
                  periodicCfr;
              {
                MATX_NVTX_START("asim_loop::periodicCfr_make_boundary_tensor",
                                matx::MATX_NVTX_LOG_USER);
                for (uint32_t i = 0; i < aerial_stage.tx_info.size(); i++) {

                  const auto [ue_ant_size, ru_ant_size] =
                      ant_size_based_on_antenna_usage(
                          beCtrl.be_config, rt_cfg.use_only_first_antenna_pair,
                          aerial_stage.rx_info[0].dual_polarized_antenna,
                          aerial_stage.tx_info[0].dual_polarized_antenna);

                  auto region = beCtrl.DeviceRegion();
                  periodicCfr.push_back(
                      region.make_boundary_tensor<aerialsim::ChannelsValueType>(
                          d_all_CFR_results[i],
                          {static_cast<int>(
                               selected_rx_indices[i].size()), // num UEs
                           1, beCtrl.be_config.fft_size, ue_ant_size,
                           ru_ant_size}));

                  CUDA_CHECK_LAST_ERROR();
                }
              }

              cudaDeviceSynchronize();
              {
                MATX_NVTX_START("asim_loop::periodicAssociation",
                                matx::MATX_NVTX_LOG_USER);
                LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                    beCtrl.periodicAssociation(periodicCfr,
                                               selected_rx_indices));
              }
              // run eriodic_cqi_update when ENABLE_CQI_CALC_BY_PERIODIC_CFR is
              // defined
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.run_periodic_cqi_update(periodicCfr,
                                                 selected_rx_indices));

              /*d_all_CFR_results contains 1 symb CFRs for UEs chosen for
              periodic association Deallocate that memory as those CFRs have
              been consumed*/
              LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                  aerial_emsolver->deAllocateDeviceMemForResults(
                      rt_cfg, d_all_CFR_results, d_all_tau_mins));
            }

            // No UEs selected in this slot (potentially due to no UEs
            // associated)
            if (beCtrl.getTotalAssociatedUsers() == 0) {
              LOG(WARNING) << "No UEs associated with any cells";
            } else {
              MATX_NVTX_START("asim_loop::ueSelection",
                              matx::MATX_NVTX_LOG_USER);
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.ueSelection(selected_rx_indices));
            }

            /*selected_rx_indices now contains UEs selected for this slot.
            Allocate CFR memory again for getting the first symbol CFRs for the
            new set of UEs. Size of the memory will be either for 1 symb or 14
            symb Hence take the org_samples_per_slot Update calc_tau_mins
            according to calc_tau_mins_ctrl here as memory for TAU is allocated
            along with memory for CFRs and TAU is required for scheduled UEs */

            mm_cfg.samples_per_slot = org_samples_per_slot;
            rt_cfg.use_only_first_antenna_pair = false;
            rt_cfg.calc_tau_mins = calc_tau_mins_ctrl;

            // No UEs selected in this slot (potentially due to no UEs
            // associated)
            if (selected_rx_indices[0].size() == 0) {
              LOG(WARNING) << "No UEs selected";
            } else {

              LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                  aerial_emsolver->allocateDeviceMemForResults(
                      aerial_stage.tx_info, aerial_stage.rx_info,
                      selected_tx_indices, selected_rx_indices, rt_cfg,
                      mm_cfg.samples_per_slot, d_all_CFR_results,
                      d_all_tau_mins));

              LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                  aerial_emsolver->runEMSolver(
                      state->get_time_idx(), aerial_stage.tx_info,
                      aerial_stage.rx_info, aerial_stage.antenna_info,
                      selected_tx_indices, selected_rx_indices, rt_cfg,
                      sample_within_slot, mm_cfg.samples_per_slot,
                      *all_ray_path_results, d_all_CFR_results,
                      d_all_tau_mins));
              cudaDeviceSynchronize();
              if (org_samples_per_slot == 1) {

                /* CFR memory dimension for each tx for second round CFR update
                using all antenna pair is rx_indices[tx_idx].size()*
                mm_cfg.samples_per_slot = 1 *
                txi.fft_size *
                aerial_stage.rx_info[0].loc_antenna.size() *
                aerial_stage.tx_info[0].loc_antenna.size() *
                aerial_stage.rx_info[0].dual_polarized_antenna?2:1 *
                aerial_stage.tx_info[0].dual_polarized_antenna?2:1 */

                LOG_AND_GOTO_CLEANUP_IF_FALSE(
                    aerial_stage.tx_info[0].fft_size ==
                    beCtrl.be_config.fft_size);
                LOG_AND_GOTO_CLEANUP_IF_FALSE(
                    aerial_stage.rx_info[0].loc_antenna.size() *
                        (aerial_stage.rx_info[0].dual_polarized_antenna ? 2
                                                                        : 1) ==
                    beCtrl.be_config.ue_panel_el());
                LOG_AND_GOTO_CLEANUP_IF_FALSE(
                    aerial_stage.tx_info[0].loc_antenna.size() *
                        (aerial_stage.tx_info[0].dual_polarized_antenna ? 2
                                                                        : 1) ==
                    beCtrl.be_config.gnb_panel_el());

                std::vector<matx::tensor_t<aerialsim::ChannelsValueType, 5>>
                    cfr_1_symbols;

                for (uint32_t i = 0; i < aerial_stage.tx_info.size(); i++) {
                  MATX_NVTX_START("asim_loop::cfr_1_symbols",
                                  matx::MATX_NVTX_LOG_USER);
                  LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                      auto region = beCtrl.DeviceRegion();
                      cfr_1_symbols.push_back(region.make_boundary_tensor<
                                              aerialsim::ChannelsValueType>(
                          d_all_CFR_results[i],
                          {static_cast<int>(
                               selected_rx_indices[i].size()), // num UEs
                           1, beCtrl.be_config.fft_size,
                           beCtrl.be_config.ue_panel_el(),
                           beCtrl.be_config.gnb_panel_el()})););
                }

                cudaDeviceSynchronize();

                /* // dump channel
                {
                    std::string ofile = "FE_CH.mat";
                    auto pb = std::make_unique<matx::detail::MatXPybind>();
                    auto np = py::module_::import("numpy");
                    auto sp = py::module_::import("scipy.io");
                    auto td = py::dict{};
                    for (int c = 0; c < cfr_1_symbols.size(); c++) {

                        auto np_ten =
                pb->TensorViewToNumpy<matx::tensor_t<aerialsim::ChannelsValueType,
                5>>(cfr_1_symbols[c]); td[std::to_string(c).c_str()] = np_ten;
                    }
                    auto obj = sp.attr("savemat")("file_name"_a =  ofile,
                "mdict"_a = td);
                }*/
                if (write_cfrs) {
                  MATX_NVTX_START("asim_loop::copyResultsFromDeviceToHost",
                                  matx::MATX_NVTX_LOG_USER);
                  const int num_all_CFR_results =
                      aerial_emsolver->copyResultsFromDeviceToHost(
                          selected_tx_indices, selected_rx_indices, rt_cfg,
                          mm_cfg.samples_per_slot, d_all_CFR_results,
                          all_CFR_results);
                  if (num_all_CFR_results < 0) {
                    LOG(ERROR)
                        << "copyResultsFromDeviceToHost returned an error ("
                        << num_all_CFR_results << ")";
                    ret = -1;
                    goto cleanup;
                  }
                }
                /* If simulation is running for single symbol, call applyTADelay
                 * and schedule */
                if (calc_tau_mins_ctrl) {
                  MATX_NVTX_START("asim_loop::applyTADelay",
                                  matx::MATX_NVTX_LOG_USER);
                  LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.applyTADelay(
                      d_all_tau_mins, cfr_1_symbols, selected_rx_indices));
                  cudaDeviceSynchronize();
                }
                {
                  MATX_NVTX_START("asim_loop::schedule",
                                  matx::MATX_NVTX_LOG_USER);
                  LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.schedule(
                      cfr_1_symbols, selected_rx_indices, false));
                }
                {
                  MATX_NVTX_START("asim_loop::telemetryLogging",
                                  matx::MATX_NVTX_LOG_USER);
                  LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                      beCtrl.telemetryLogging());
                }
              }
            }
          } else if (org_samples_per_slot == 14 &&
                     sample_within_slot == 13) { // last symbol of the slot
            /* CFR memory dimension for each tx for second round CFR update
            using all antenna pair is rx_indices[tx_idx].size()*
            mm_cfg.samples_per_slot = 14 *
            txi.fft_size *
            aerial_stage.rx_info[0].loc_antenna.size() *
            aerial_stage.tx_info[0].loc_antenna.size() *
            aerial_stage.rx_info[0].dual_polarized_antenna?2:1 *
            aerial_stage.tx_info[0].dual_polarized_antenna?2:1 */

            MATX_NVTX_START("asim_loop::last_symbol", matx::MATX_NVTX_LOG_USER);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(aerial_stage.tx_info[0].fft_size ==
                                          beCtrl.be_config.fft_size);
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.rx_info[0].loc_antenna.size() *
                    (aerial_stage.rx_info[0].dual_polarized_antenna ? 2 : 1) ==
                beCtrl.be_config.ue_panel_el());
            LOG_AND_GOTO_CLEANUP_IF_FALSE(
                aerial_stage.tx_info[0].loc_antenna.size() *
                    (aerial_stage.tx_info[0].dual_polarized_antenna ? 2 : 1) ==
                beCtrl.be_config.gnb_panel_el());

            cudaDeviceSynchronize();

            // No UEs selected in this slot (potentially due to no UEs
            // associated)
            if (selected_rx_indices[0].size() == 0) {
              LOG(WARNING) << "No UEs associated! Continue to next sample.";
            } else {

              std::vector<matx::tensor_t<aerialsim::ChannelsValueType, 5>>
                  cfr_14_symbols;
              for (uint32_t i = 0; i < aerial_stage.tx_info.size(); i++) {

                LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                    auto region = beCtrl.DeviceRegion();
                    cfr_14_symbols.push_back(
                        region
                            .make_boundary_tensor<aerialsim::ChannelsValueType>(
                                d_all_CFR_results[i],
                                {static_cast<int>(
                                     selected_rx_indices[i].size()), // num UEs
                                 14,                                 // symbols
                                 beCtrl.be_config.fft_size,
                                 beCtrl.be_config.ue_panel_el(),
                                 beCtrl.be_config.gnb_panel_el()})););
              }
              cudaDeviceSynchronize();
              if (write_cfrs) {
                MATX_NVTX_START("asim_loop::copyResultsFromDeviceToHost",
                                matx::MATX_NVTX_LOG_USER);
                const int num_all_CFR_results =
                    aerial_emsolver->copyResultsFromDeviceToHost(
                        selected_tx_indices, selected_rx_indices, rt_cfg,
                        mm_cfg.samples_per_slot, d_all_CFR_results,
                        all_CFR_results);
                if (num_all_CFR_results < 0) {
                  LOG(ERROR)
                      << "copyResultsFromDeviceToHost returned an error ("
                      << num_all_CFR_results << ")";
                  ret = -1;
                  goto cleanup;
                }
              }
              if (calc_tau_mins_ctrl) {
                LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(beCtrl.applyTADelay(
                    d_all_tau_mins, cfr_14_symbols, selected_rx_indices));
                cudaDeviceSynchronize();
              }
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.schedule(cfr_14_symbols, selected_rx_indices, true));
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.telemetryLogging());
            }
          }

          // release the CFR memory when:
          // requested channel size > 0 && (last sample of samples_per_slot ||
          // batch boundy) used by beCtrl.schedule() and
          // beCtrl.initCellAssociation(), respectively
          if (selected_rx_indices[0].size() > 0 &&
              ((sample_within_slot == mm_cfg.samples_per_slot - 1) ||
               rt_cell_association_state)) {
            MATX_NVTX_START("deAllocateDeviceMemForResults",
                            matx::MATX_NVTX_LOG_USER);

            int32_t status = aerial_emsolver->deAllocateDeviceMemForResults(
                rt_cfg, d_all_CFR_results, d_all_tau_mins);
            if (status != 0) {
              LOG(ERROR) << "Error returned by deAllocateDeviceMemForResults: "
                         << status;
              ret = -1;
              goto cleanup;
            }
          }

          // No UEs selected in this slot (potentially due to no UEs associated)
          if (selected_rx_indices[0].size() == 0) {
            LOG(VERBOSE) << "No UEs associated! Continue to next sample.";
          } else {
            if (!rt_cell_association_state && (write_cirs || write_raypaths)) {
              LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
                  beCtrl.dump_trace("asim_trace.json"));

              MATX_NVTX_START("asim_loop::not_rt_cell_association_state",
                              matx::MATX_NVTX_LOG_USER)
              // Creating local copies to ensure correct behavior with thread
              auto use_only_first_antenna_pair =
                  rt_cfg.use_only_first_antenna_pair;
              auto local_tx_info = aerial_stage.tx_info;
              auto local_rx_info = aerial_stage.rx_info;
              auto local_rx_indices = selected_rx_indices;
              auto local_tx_indices = selected_tx_indices;
              auto local_time_idx = state->get_time_idx();
              auto tid = ray_write_threads.size();
              auto done = std::make_shared<bool>(false);
              thread_running.push_back(done);

              /* Launch a thread for each insertion and delete the vector
               * afterwards */
              ray_write_threads.push_back(
                  std::thread{[write_cirs, write_raypaths, all_ray_path_results,
                               local_tx_info, local_rx_info, local_tx_indices,
                               local_rx_indices, use_only_first_antenna_pair,
                               local_time_idx, state, &insert_cirs_ret,
                               &insert_raypaths_ret, tid,
                               thread_finished = std::move(done)]() mutable {
                    using sc = std::chrono::steady_clock;
                    auto start = sc::now();

                    if (write_cirs) {
                      LOG(DEBUG) << "Adding " << all_ray_path_results->size()
                                 << " ray results to the cirs table";
                      MATX_NVTX_START("asim_loop:insertCIRs",
                                      matx::MATX_NVTX_LOG_USER);
                      insert_cirs_ret = insertCIRs(
                          state->db().settings(), *all_ray_path_results,
                          local_tx_info, local_rx_info, local_tx_indices,
                          local_rx_indices, use_only_first_antenna_pair,
                          local_time_idx);
                    }

                    if (write_raypaths) {
                      LOG(DEBUG) << "Adding " << all_ray_path_results->size()
                                 << " ray results to the raypaths table";
                      MATX_NVTX_START("asim_loop:insertRaypaths",
                                      matx::MATX_NVTX_LOG_USER);
                      insert_raypaths_ret =
                          insertRaypaths(state->db().settings(),
                                         *all_ray_path_results, local_time_idx);
                    }

                    delete all_ray_path_results;
                    auto end = sc::now();
                    auto ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count();
                    LOG(DEBUG) << "insertCIRs and insertRaypaths took " << ms
                               << " ms (thread_id=" << tid << ")";
                    *thread_finished = true;
                  }});
            } else {
              delete all_ray_path_results;
            }

            // write CFRs to clickhouse, if BE needs the CFRs stay only in the
            // device memory then disable the following "if" block only write
            // CFRs if wideband CFRs is enabled and not in cell-association step
            const auto do_write_cfrs =
                write_cfrs && aerial_stage.tx_info[0].fft_size > 1;
            if (!do_write_cfrs) {
              // explicitly free here in case the freeing thread below is not
              // launched
              delete all_CFR_results;
            }
            if (do_write_cfrs &&
                (!mm_cfg.slot_symbol_mode ||
                 (sample_within_slot == mm_cfg.samples_per_slot - 1 &&
                  !rt_cell_association_state))) {

              LOG(DEBUG) << "Adding CFR results to clickhouse db";

              // Creating local copies to ensure correct behavior with thread

              auto use_only_first_antenna_pair =
                  rt_cfg.use_only_first_antenna_pair;
              auto samples_per_slot = mm_cfg.samples_per_slot;

              auto local_tx_info = aerial_stage.tx_info;
              auto local_rx_info = aerial_stage.rx_info;
              auto local_tx_indices = selected_tx_indices;
              auto local_rx_indices = selected_rx_indices;
              auto local_time_idx = state->get_time_idx();

              auto tid = ray_write_threads.size();
              auto done = std::make_shared<bool>(false);
              thread_running.push_back(done);
              ray_write_threads.push_back(
                  std::thread{[all_CFR_results, local_tx_info, local_rx_info,
                               local_tx_indices, local_rx_indices,
                               use_only_first_antenna_pair, samples_per_slot,
                               local_time_idx, state, &insert_cfrs_ret, tid,
                               thread_finished = std::move(done)]() mutable {
                    using sc = std::chrono::steady_clock;
                    auto start = sc::now();
                    MATX_NVTX_START("asim_loop:insertCFRS",
                                    matx::MATX_NVTX_LOG_USER);
                    insert_cfrs_ret = insertCFRs(
                        state->db().settings(), *all_CFR_results, local_tx_info,
                        local_rx_info, local_tx_indices, local_rx_indices,
                        use_only_first_antenna_pair, samples_per_slot,
                        local_time_idx);

                    delete all_CFR_results;
                    auto end = sc::now();
                    auto ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - start)
                            .count();
                    LOG(DEBUG) << "insertCFRs took " << ms
                               << " ms (thread_id=" << tid << ")";
                    *thread_finished = true;
                  }});
            }
          }

          // join threads that have run to completion (erase/iterate idiom) or
          // wait if too many outstanding threads
          auto done_iter = thread_running.begin();
          for (auto thread_iter = ray_write_threads.begin();
               thread_iter != ray_write_threads.end();) {
            const bool too_many_oustanding_threads =
                ray_write_threads.size() >= MAX_OUTSTANDING_DB_THREADS;
            // dereference twice because it is an iterator and shared_ptr
            if (too_many_oustanding_threads || **done_iter) {
              thread_iter->join();
              done_iter = thread_running.erase(done_iter);
              thread_iter = ray_write_threads.erase(thread_iter);
            } else {
              thread_iter++;
              done_iter++;
            }
          }

          // debug.dump(all_ray_results, state->get_time_idx());

          if (state->get_state() ==
                  aerial::sim::simulation_state::ongoing_sim ||
              state->get_state() == aerial::sim::simulation_state::paused_sim) {
            // only increase time sample if not in rt_association step
            if (mm_cfg.slot_symbol_mode) {
              if (!rt_cell_association_state) {
                state->inc_time_idx();
                if (state->get_time_idx() % num_samples_per_batch == 0 &&
                    rt_cell_association_ctrl) {
                  // if the next sample starts a new batch, do cell assocaition
                  // again
                  rt_cell_association_state = true;
                }
              } else { // reset
                rt_cell_association_state = false;
                rt_cfg.use_only_first_antenna_pair = false;
                mm_cfg.samples_per_slot = org_samples_per_slot;
              }
            } else {
              state->inc_time_idx();
            }
          }

          if (insert_cirs_ret != 0 || insert_raypaths_ret != 0 ||
              insert_cfrs_ret != 0) {
            ret = -1;
            goto cleanup;
          }

          // Send progress
          const float progress = float(sample + 1) / float(num_time_steps);
          progressValue.store(progress);

          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
              logGPUMemUsage("[slot " + std::to_string(slot) + " end]"));
          auto slot_end = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch());
          auto diff = slot_end - slot_start;
          LOG(DEBUG) << "Time taken for slot processing = " << diff.count()
                     << " ms";
        }
        auto &stage = state->get_aerial_stage();
        if (insertRUs(state->db().settings(), stage.tx_info,
                      stage.ru_du_info_map) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (insertPanels(state->db().settings(), stage.antenna_info.panels) !=
            0) {
          ret = -1;
          goto cleanup;
        }
        if (insertPatterns(state->db().settings(),
                           stage.antenna_info.patterns) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (write_telemetry &&
            insertTelemetry(state->db().settings(), beCtrl.telemetry) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (insertCsiReport(state->db().settings(), beCtrl.csi_report) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (insertDUs(state->db().settings(), stage.du_info_map) != 0) {
          ret = -1;
          goto cleanup;
        }
      }
    }
  }

  goto cleanup;

// We use a goto cleanup for error handling to simplify cleaning up
// resources/threads from anywhere in the simulation loop above.
cleanup:
  progressValue.store(0.99F); // 99% to indicate to user that we are still
                              // waiting for DB threads to join
  LOG(INFO) << "Waiting for DB ray write threads to complete.  This may take a "
               "while if the simulation is very large.\n";
  for (auto &t : ray_write_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  LOG(INFO) << "Finished computing and storing results in database "
            << state->db().settings().name;

  // Explicitly synchronize stream used by EM solver, to avoid any pending
  // operations or GPU memory cleanup from being deferred until the next
  // simulation starts.
  cudaStreamSynchronize(state->getStream(0));

  // Reset progress only after potentially long running DB threads have joined.
  progressValue.store(-1.0f);
  if (progressThread.joinable()) {
    progressThread.join();
  }

  transition_to_stop_state_if_sim_completed_or_error(*state, ret);

  auto sim_end = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  auto sim_duration = sim_end - sim_start;
  LOG(DEBUG) << "Duration of simulation = " << sim_duration.count() << " ms";

  return ret;
}
//==================================================================================================================================================
ASIM_EXPORT int32_t
handler_play(aerial::sim::simulation_state *state,
             std::function<void(float)> send_progress_func) {
  MATX_NVTX_START("asim_loop::handler_play", matx::MATX_NVTX_LOG_USER);
  int32_t ret = 0;
  int insert_cirs_ret = 0;
  int insert_raypaths_ret = 0;
  int insert_cfrs_ret = 0;
  auto init_state = state->get_state();
  std::vector<std::shared_ptr<bool>> thread_running;
  std::vector<std::thread> ray_write_threads;
  constexpr int MAX_OUTSTANDING_DB_THREADS = 10;
  auto sim_start = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  // handle sim progress
  std::atomic<float> progressValue(0.0f);
  std::thread progressThread;

  if (init_state == aerial::sim::simulation_state::mobility_available ||
      init_state == aerial::sim::simulation_state::paused_sim) {
    if (init_state == aerial::sim::simulation_state::mobility_available) {
      if (state->update_stage_sim_params() != 0) {
        ret = -1;
        goto cleanup;
      }
      if (state->load_deployed_ues() != 0) {
        ret = -1;
        goto cleanup;
      }
      if (state->load_deployed_rus() != 0) {
        ret = -1;
        goto cleanup;
      }

      // clear all tables except db_info
      if (clear_sim_results_from_db(state->db().settings()) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (insertUEs(state->db().settings(), state->get_mobility_users(),
                    state->get_mm_config().scale) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_time_info_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_scenario_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      if (add_materials_to_db(*state) != 0) {
        ret = -1;
        goto cleanup;
      }
      // if (add_world_to_db(*state) != 0) {
      //   ret = -1;
      //   goto cleanup;
      // }
    }

    emsolver::RTConfig rt_cfg = read_rt_config_from_stage(
        state->get_aerial_stage().int_stage, state->simulate_ran);

    AerialStage &aerial_stage = state->get_aerial_stage();
    if (aerial_stage.tx_info.empty()) {
      LOG(ERROR) << "No RUs found.  Please deploy at least one RU and re-run "
                    "the simulation.";
      ret = -1;
      goto cleanup;
    }
    if (aerial_stage.rx_info.empty()) {
      LOG(ERROR) << "No UEs found.  Please deploy at least one UE and re-run "
                    "the simulation.";
      ret = -1;
      goto cleanup;
    }

    const bool enable_training =
        read_enable_training_from_stage(aerial_stage.int_stage);
    if (enable_training &&
        !consistent_tx_rx_info(aerial_stage.tx_info, aerial_stage.rx_info)) {
      ret = -1;
      goto cleanup;
    }

    const std::vector<aerial::sim::mm::user> &users =
        state->get_mobility_users();

    getMobilityParams(state);
    auto mm_cfg = state->get_mm_config();
    const int org_samples_per_slot = mm_cfg.samples_per_slot;

    if (enable_training) {
      const auto scenario_info = config_to_scenario_info(mm_cfg);
      const auto ru_ue_info =
          tx_rx_to_ru_ue_info(aerial_stage.tx_info, aerial_stage.rx_info);
      ret = state->trainer_scenario(scenario_info, ru_ue_info);
      if (ret != 0) {
        ret = -1;
        goto cleanup;
      }
    }

    LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
        state->consume(aerial::sim::simulation_state::sim_play));

    if (!users.empty()) {
      const std::vector<aerial::sim::mm::sample> &samples =
          users.at(0).flattened_samples;

      if (!samples.empty()) {
        if (aerial_stage.tx_info.empty()) {
          LOG(ERROR) << "No RUs found.  Please deploy at least one RU and "
                        "re-run the simulation.";
          ret = -1;
          goto cleanup;
        }
        if (aerial_stage.rx_info.empty()) {
          LOG(ERROR) << "No UEs found.  Please deploy at least one UE and "
                        "re-run the simulation.";
          ret = -1;
          goto cleanup;
        }

        bool rt_cell_association_ctrl =
            false; // SET TO TRUE to enable cell association step (tracing for
                   // all links)
        bool calc_tau_mins_ctrl = false; // SET TO TRUE to enable calculating
                                         // min propagation delays (tau_mins)

        progressThread = std::thread([&progressValue, &send_progress_func]() {
          send_progress_task(progressValue, send_progress_func);
        });

        // initial rt_cell_association_state
        bool rt_cell_association_state =
            rt_cell_association_ctrl &&
            mm_cfg.slot_symbol_mode; // only do cell-association step in
                                     // slot/symbol mode

        // set of selected txs and associated rxs for EMSolver call for each
        // time sample
        auto &selected_tx_indices = state->selected_tx_indices;
        auto &selected_rx_indices = state->selected_rx_indices;

        // assume all txs are selected
        selected_tx_indices.resize(aerial_stage.tx_info.size());
        std::iota(selected_tx_indices.begin(), selected_tx_indices.end(), 0);

        LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
            logGPUMemUsage("[Before EM solver setup]"));
        std::unique_ptr<emsolver::AerialEMSolver> aerial_emsolver;
        try {
          aerial_emsolver = std::make_unique<emsolver::AerialEMSolver>(
              aerial_stage.tx_info, aerial_stage.rx_info,
              aerial_stage.antenna_info, aerial_stage.geometry_info, rt_cfg,
              state->getStream(0));
        } catch (const std::runtime_error &e) {
          LOG(ERROR) << "Problem initializing EM solver: " << e.what() << ".\n"
                     << "Recommend checking your scenario settings and GPU "
                        "memory usage on compute node.";
          ret = -1;
          goto cleanup;
        }
        LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
            aerial_emsolver->registerLogCallback(emsolver_log_callback));
        LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
            logGPUMemUsage("[After EM solver setup]"));
        const auto num_time_steps = samples.size();

        const bool write_cfrs =
            true;
        const bool write_cirs =
            true;
        const bool write_raypaths = true;
        const bool write_telemetry = true;
        const bool write_training_result = true;
        LOG(INFO) << "Opt-in DB tables: write_cfrs=" << write_cfrs
                  << ", write_cirs=" << write_cirs
                  << ", write_raypaths=" << write_raypaths
                  << ", write_telemetry=" << write_telemetry
                  << ", write_training_result=" << write_training_result;

        while ((state->get_state() ==
                aerial::sim::simulation_state::ongoing_sim) &&
               (state->get_time_idx() < num_time_steps)) {
          const auto sample = state->get_time_idx();
          auto batch = asim::time_index_to_batch(
              sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
              mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          const auto slot = asim::time_index_to_slot_within_batch(
              sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
              mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          const auto sample_within_slot =
              asim::time_index_to_sample_within_slot(
                  sample, mm_cfg.slot_symbol_mode, mm_cfg.slots_per_batch,
                  mm_cfg.samples_per_slot, num_time_steps, mm_cfg.batches);
          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
              logGPUMemUsage("[slot " + std::to_string(slot) + " start]"));

          const int num_samples_per_batch =
              mm_cfg.slot_symbol_mode
                  ? mm_cfg.slots_per_batch * mm_cfg.samples_per_slot
                  : num_time_steps / mm_cfg.batches;

          // SELECT RUs AND UEs LISTS
          if (state->get_time_idx() % num_samples_per_batch == 0) {
            // first sample of the first slot
            if (rt_cell_association_state) {
              // reset to all UEs for cell association
              selected_rx_indices.clear();
              selected_rx_indices.resize(selected_tx_indices.size());
              for (int selected_tx_idx = 0;
                   selected_tx_idx < selected_tx_indices.size();
                   selected_tx_idx++) {
                selected_rx_indices[selected_tx_idx].resize(
                    aerial_stage.rx_info.size());
                std::iota(selected_rx_indices[selected_tx_idx].begin(),
                          selected_rx_indices[selected_tx_idx].end(), 0);
              }

              // only first antenna pair for this sample
              rt_cfg.use_only_first_antenna_pair = true;
              mm_cfg.samples_per_slot = 1;
            } else {
              // // lists of selected txs and rxs, should come from the BE cell
              // association
              // // e.g.
              // selected_tx_indices.clear();
              // selected_tx_indices = {0, 10};

              // selected_rx_indices.clear();
              // selected_rx_indices = {{0, 2}, {1, 3, 5}};

              // or choose all links for back-ward compatability with FE
              // standalone applications
              selected_rx_indices.clear();
              selected_rx_indices.resize(selected_tx_indices.size());
              for (int selected_tx_idx = 0;
                   selected_tx_idx < selected_tx_indices.size();
                   selected_tx_idx++) {
                selected_rx_indices[selected_tx_idx].resize(
                    aerial_stage.rx_info.size());
                std::iota(selected_rx_indices[selected_tx_idx].begin(),
                          selected_rx_indices[selected_tx_idx].end(), 0);
              }
            }
          }

          if (selected_tx_indices.size() == 0) {
            LOG(ERROR) << "No RUs found!";
            ret = -1;
            goto cleanup;
          }

          for (int selected_tx_idx = 0;
               selected_tx_idx < selected_tx_indices.size();
               selected_tx_idx++) {
            if (selected_rx_indices[selected_tx_idx].size() == 0) {
              LOG(ERROR) << "No UEs found for Tx "
                         << aerial_stage
                                .tx_info[selected_tx_indices[selected_tx_idx]]
                                .tx_ID
                         << "!";
              ret = -1;
              goto cleanup;
            }
          }

          std::string wb_str =
              aerial_stage.tx_info[0].fft_size > 1
                  ? " (wideband CFRs enabled, fft_size=" +
                        std::to_string(aerial_stage.tx_info[0].fft_size) + ")"
                  : " (wideband CFRs disabled)";
          if (rt_cell_association_state) {
            LOG(INFO) << "Computing all links for cell association for batch "
                      << batch << wb_str << "...";
          } else {
            if (mm_cfg.slot_symbol_mode) {
              LOG(INFO) << "Computing sample=" << (sample + 1) << "/"
                        << num_time_steps << ", batch=" << batch
                        << ", slot=" << slot
                        << ", sample_within_slot=" << sample_within_slot
                        << wb_str;
            } else {
              LOG(INFO) << "Computing sample=" << (sample + 1) << "/"
                        << num_time_steps << ", batch=" << batch << wb_str;
            }
          }

          std::vector<emsolver::RayPath> *all_ray_path_results =
              new std::vector<emsolver::RayPath>();

          std::vector<emsolver::d_complex *> d_all_CFR_results(
              selected_tx_indices.size(), nullptr);
          auto all_CFR_results =
              new std::vector<std::vector<emsolver::d_complex>>{
                  selected_tx_indices.size()};

          std::vector<float *> d_all_tau_mins(selected_tx_indices.size(),
                                              nullptr);

          rt_cfg.calc_tau_mins =
              calc_tau_mins_ctrl &&
              (sample_within_slot ==
               0); // only calculate tau_mins for the first symbol within a slot
          // update aerial_stage's rx_info
          if (state->update_ues() != 0) {
            ret = -1;
            goto cleanup;
          }

          LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
              aerial_emsolver->allocateDeviceMemForResults(
                  aerial_stage.tx_info, aerial_stage.rx_info,
                  selected_tx_indices, selected_rx_indices, rt_cfg,
                  mm_cfg.samples_per_slot, d_all_CFR_results, d_all_tau_mins));

          LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(aerial_emsolver->runEMSolver(
              state->get_time_idx(), aerial_stage.tx_info, aerial_stage.rx_info,
              aerial_stage.antenna_info, selected_tx_indices,
              selected_rx_indices, rt_cfg, sample_within_slot,
              mm_cfg.samples_per_slot, *all_ray_path_results, d_all_CFR_results,
              d_all_tau_mins));

          if (!rt_cell_association_state && (write_cirs || write_raypaths)) {
            // Creating local copies to ensure correct behavior with thread

            auto use_only_first_antenna_pair =
                rt_cfg.use_only_first_antenna_pair;
            auto local_tx_info = aerial_stage.tx_info;
            auto local_rx_info = aerial_stage.rx_info;
            auto local_rx_indices = selected_rx_indices;
            auto local_tx_indices = selected_tx_indices;
            auto local_time_idx = state->get_time_idx();
            auto tid = ray_write_threads.size();
            auto done = std::make_shared<bool>(false);
            thread_running.push_back(done);

            /* Launch a thread for each insertion */
            ray_write_threads.push_back(std::thread{
                [write_cirs, write_raypaths, all_ray_path_results,
                 local_tx_info, local_rx_info, local_tx_indices,
                 local_rx_indices, use_only_first_antenna_pair, local_time_idx,
                 state, &insert_cirs_ret, &insert_raypaths_ret, tid,
                 thread_finished = std::move(done)]() mutable {
                  using sc = std::chrono::steady_clock;
                  auto start = sc::now();

                  if (write_cirs) {
                    LOG(DEBUG) << "Adding " << all_ray_path_results->size()
                               << " ray results to the cirs table";
                    insert_cirs_ret = insertCIRs(
                        state->db().settings(), *all_ray_path_results,
                        local_tx_info, local_rx_info, local_tx_indices,
                        local_rx_indices, use_only_first_antenna_pair,
                        local_time_idx);
                  }

                  if (write_raypaths) {
                    LOG(DEBUG) << "Adding " << all_ray_path_results->size()
                               << " ray results to the raypaths table";
                    insert_raypaths_ret =
                        insertRaypaths(state->db().settings(),
                                       *all_ray_path_results, local_time_idx);
                  }

                  delete all_ray_path_results;
                  auto end = sc::now();
                  auto ms =
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          end - start)
                          .count();
                  LOG(DEBUG) << "insertCIRs and insertRaypaths took " << ms
                             << " ms (thread_id=" << tid << ")";
                  *thread_finished = true;
                }});
          } else {
            delete all_ray_path_results;
          }

          // write CFRs to clickhouse, if BE needs the CFRs stay only in the
          // device memory then disable the following "if" block only write CFRs
          // if wideband CFRs is enabled and not in cell-association step
          const auto do_write_cfrs =
              write_cfrs && aerial_stage.tx_info[0].fft_size > 1;
          if (!do_write_cfrs) {
            // explicitly free here in case the freeing thread below is not
            // launched
            delete all_CFR_results;
          }
          if (do_write_cfrs &&
              (!mm_cfg.slot_symbol_mode ||
               (sample_within_slot == mm_cfg.samples_per_slot - 1 &&
                !rt_cell_association_state))) {
            const int32_t num_all_CFR_results =
                aerial_emsolver->copyResultsFromDeviceToHost(
                    selected_tx_indices, selected_rx_indices, rt_cfg,
                    mm_cfg.samples_per_slot, d_all_CFR_results,
                    all_CFR_results);
            if (num_all_CFR_results < 0) {
              LOG(ERROR) << "copyResultsFromDeviceToHost returned an error ("
                         << num_all_CFR_results << ")";
              ret = -1;
              goto cleanup;
            }

            LOG(DEBUG) << "Adding " << num_all_CFR_results
                       << " CFR results to clickhouse db";

            // Creating local copies to ensure correct behavior with thread
            auto use_only_first_antenna_pair =
                rt_cfg.use_only_first_antenna_pair;
            auto samples_per_slot = mm_cfg.samples_per_slot;
            auto local_time_idx = state->get_time_idx();
            auto local_tx_info = aerial_stage.tx_info;
            auto local_rx_info = aerial_stage.rx_info;
            auto local_rx_indices = selected_rx_indices;
            auto local_tx_indices = selected_tx_indices;

            if (enable_training &&
                sample_within_slot == mm_cfg.samples_per_slot - 1) {
              auto assoc_info = tx_rx_users_to_ru_association_info(
                  users, selected_rx_indices, aerial_stage.tx_info,
                  aerial_stage.rx_info, state->get_time_idx(), mm_cfg.scale);
              const auto ru_ue_info = tx_rx_to_ru_ue_info(aerial_stage.tx_info,
                                                          aerial_stage.rx_info);
              const time_info ti(state->get_time_idx(), batch, slot,
                                 sample_within_slot);
              const auto tr_info = state->trainer_append_cfr(
                  ti, ru_ue_info, assoc_info, *all_CFR_results);
              if (write_training_result &&
                  add_training_result_to_db(state->db().settings(), tr_info) !=
                      0) {
                ret = -1;
                goto cleanup;
              }
            }

            auto tid = ray_write_threads.size();
            auto done = std::make_shared<bool>(false);
            thread_running.push_back(done);
            ray_write_threads.push_back(
                std::thread{[all_CFR_results, local_tx_info, local_rx_info,
                             local_tx_indices, local_rx_indices,
                             use_only_first_antenna_pair, samples_per_slot,
                             local_time_idx, state, &insert_cfrs_ret, tid,
                             thread_finished = std::move(done)]() mutable {
                  using sc = std::chrono::steady_clock;
                  auto start = sc::now();
                  insert_cfrs_ret =
                      insertCFRs(state->db().settings(), *all_CFR_results,
                                 local_tx_info, local_rx_info, local_tx_indices,
                                 local_rx_indices, use_only_first_antenna_pair,
                                 samples_per_slot, local_time_idx);

                  delete all_CFR_results;
                  auto end = sc::now();
                  auto ms =
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          end - start)
                          .count();
                  LOG(DEBUG) << "insertCFRs took " << ms
                             << " ms (thread_id=" << tid << ")";
                  *thread_finished = true;
                }});
          }
          // join threads that have run to completion (erase/iterate idiom) or
          // wait if too many outstanding threads
          auto done_iter = thread_running.begin();
          for (auto thread_iter = ray_write_threads.begin();
               thread_iter != ray_write_threads.end();) {
            const bool too_many_oustanding_threads =
                ray_write_threads.size() >= MAX_OUTSTANDING_DB_THREADS;
            // dereference twice because it is an iterator and shared_ptr
            if (too_many_oustanding_threads || **done_iter == true) {
              thread_iter->join();
              done_iter = thread_running.erase(done_iter);
              thread_iter = ray_write_threads.erase(thread_iter);
            } else {
              thread_iter++;
              done_iter++;
            }
          }
          // debug.dump(all_ray_results, state->get_time_idx());

          if (state->get_state() ==
                  aerial::sim::simulation_state::ongoing_sim ||
              state->get_state() == aerial::sim::simulation_state::paused_sim) {
            // only increase time sample if not in rt_association step
            if (mm_cfg.slot_symbol_mode) {
              if (!rt_cell_association_state) {
                state->inc_time_idx();
                if (state->get_time_idx() % num_samples_per_batch == 0 &&
                    rt_cell_association_ctrl) {
                  // if the next sample starts a new batch, do cell assocaition
                  // again
                  rt_cell_association_state = true;
                }
              } else { // reset
                rt_cell_association_state = false;
                rt_cfg.use_only_first_antenna_pair = false;
                mm_cfg.samples_per_slot = org_samples_per_slot;
              }
            } else {
              state->inc_time_idx();
            }
          }
          {
            MATX_NVTX_START("asim_loop:deAllocateDeviceMemForResults",
                            matx::MATX_NVTX_LOG_USER);
            LOG_AND_GOTO_CLEANUP_IF_NONZERO_RETURN(
                aerial_emsolver->deAllocateDeviceMemForResults(
                    rt_cfg, d_all_CFR_results, d_all_tau_mins));
          }

          if (insert_cirs_ret != 0 || insert_raypaths_ret != 0 ||
              insert_cfrs_ret != 0) {
            ret = -1;
            goto cleanup;
          }

          // Send progress
          const float progress = float(sample + 1) / float(num_time_steps);
          progressValue.store(progress);
          LOG_AND_GOTO_CLEANUP_IF_STATEMENT_THROWS(
              logGPUMemUsage("[slot " + std::to_string(slot) + " end]"));
        }

        auto &stage = state->get_aerial_stage();
        if (insertRUs(state->db().settings(), stage.tx_info,
                      stage.ru_du_info_map) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (insertPanels(state->db().settings(), stage.antenna_info.panels) !=
            0) {
          ret = -1;
          goto cleanup;
        }
        if (insertPatterns(state->db().settings(),
                           stage.antenna_info.patterns) != 0) {
          ret = -1;
          goto cleanup;
        }
        if (insertDUs(state->db().settings(), stage.du_info_map) != 0) {
          ret = -1;
          goto cleanup;
        }
      }
    }
  }

  goto cleanup;

// We use a goto cleanup for error handling to simplify cleaning up
// resources/threads from anywhere in the simulation loop above.
cleanup:
  progressValue.store(0.99F); // 99% to indicate to user that we are still
                              // waiting for DB threads to join
  LOG(INFO) << "Waiting for DB ray write threads to complete.  This may take a "
               "while if the simulation is very large.\n";
  for (auto &t : ray_write_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  LOG(INFO) << "Finished computing and storing results in database "
            << state->db().settings().name;
  transition_to_stop_state_if_sim_completed_or_error(*state, ret);

  // Explicitly synchronize stream used by EM solver, to avoid any pending
  // operations or GPU memory cleanup from being deferred until the next
  // simulation starts.
  cudaStreamSynchronize(state->getStream(0));

  // Reset progress only after potentially long running DB threads have joined.
  progressValue.store(-1.0f);
  if (progressThread.joinable()) {
    progressThread.join();
  }

  auto sim_end = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());
  auto sim_duration = sim_end - sim_start;
  LOG(DEBUG) << "Duration of simulation = " << sim_duration.count() << " ms";

  return ret;
}

ASIM_EXPORT int32_t handler_pause(aerial::sim::simulation_state *state) {
  return state->consume(aerial::sim::simulation_state::sim_pause);
}

ASIM_EXPORT int32_t handler_stop(aerial::sim::simulation_state *state) {
  aerial::sim::simulation_state::state internal_state = state->get_state();

  int32_t ret = 0;
  switch (internal_state) {
  case aerial::sim::simulation_state::paused_sim:
  case aerial::sim::simulation_state::ongoing_sim:
    ret = state->consume(aerial::sim::simulation_state::sim_stop);
    break;
  }

  return ret;
}

ASIM_EXPORT int32_t handler_mobility(aerial::sim::simulation_state *state) {
  aerial::sim::simulation_state::state internal_state = state->get_state();

  if (internal_state == aerial::sim::simulation_state::init ||
      internal_state == aerial::sim::simulation_state::mobility_available) {
    aerial::sim::cnt::getMobilityParams(state);

    std::vector<aerial::sim::mm::user> manual_users;
    if (const auto ret = aerial::sim::cnt::getManualUsers(state, manual_users);
        ret < 0) {
      return ret;
    }

    LOG(DEBUG) << "Manually inserted users: " << manual_users.size();

    const std::vector<float3> &vertices = state->get_mesh_vertices();
    const auto spawn_zone_bb = get_spawn_zone_bounding_box(
        state->get_aerial_stage().int_stage, vertices);
    const auto *spawn_zone_ptr =
        spawn_zone_bb.has_value() ? &spawn_zone_bb.value() : nullptr;
    if (const auto ret =
            state->generate_ue_mobility(spawn_zone_ptr, manual_users);
        ret != 0) {
      return ret;
    }

    aerial::sim::cnt::create_ues(state);
    if (const auto ret = state->load_deployed_ues(); ret != 0) {
      return ret;
    }
    if (const auto ret = createDatabaseAndTables(state->db().settings());
        ret != 0) {
      return ret;
    }
    if (const auto ret = clearTable(state->db().settings(), "ues"); ret != 0) {
      return ret;
    }
    if (const auto ret =
            insertUEs(state->db().settings(), state->get_mobility_users(),
                      state->get_mm_config().scale);
        ret != 0) {
      return ret;
    }
    if (const auto ret = clearTable(state->db().settings(), "time_info");
        ret != 0) {
      return ret;
    }
    if (const auto ret = add_time_info_to_db(*state); ret != 0) {
      return ret;
    }
    if (const auto ret =
            state->consume(aerial::sim::simulation_state::sim_mobility);
        ret != 0) {
      return ret;
    }
  } else {
    LOG(ERROR) << "Unable to generate mobility.  You may need to stop any "
                  "active simulations first.";
    return -1;
  }

  return 0;
}
} // namespace cnt
} // namespace sim
} // namespace aerial
