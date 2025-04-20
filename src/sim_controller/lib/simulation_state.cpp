/***************************************************************************
 # Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto. Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#include "simulation_state.hpp"
#include "aerial_emsolver_api.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/matrix4d.h"
#include "logger.hpp"
#include <sstream>
#include <iomanip>
#include <boost/algorithm/string/replace.hpp>
#include <cuda_runtime.h>
#include "aerial_emsolver_api.h"
#include "clickhouse.hpp"
#include "shared_types.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace { // begin anonymous namespace
[[nodiscard]] int32_t
panel_name_to_index(const std::string &target_panel_name,
                    const std::vector<emsolver::AntennaPanel> &panels) {
  auto it = std::find_if(panels.cbegin(), panels.cend(),
                         [&target_panel_name](const auto &panel) {
                           return panel.panel_name == target_panel_name;
                         });
  bool not_found = it == panels.end();
  if (not_found) {
    LOG(ERROR) << "Did not find panel name: " << target_panel_name
               << " in list of antenna panels.  Check your panels and UE/RU "
                  "panel assignments.";
  }
  return not_found ? -1 : std::distance(panels.cbegin(), it);
}
} // end anonymous namespace

namespace aerial::sim {

void simulation_state_log_callback(const std::string &level,
                                   const std::string &message) {
  if (level.find("ERROR") != std::string::npos) {
    LOG(ERROR) << message;
  } else if (level.find("WARN") != std::string::npos) {
    LOG(WARNING) << message;
  } else if (level.find("DEBUG") != std::string::npos) {
    LOG(DEBUG) << message;
  } else {
    LOG(WARNING) << "unknown trainer log level " << level;
  }
}

simulation_state::simulation_state(
    const std::vector<float3> &ext_vertices,
    const std::vector<int3> &ext_triangles_outdoor,
    const std::vector<int3> &ext_triangles_indoor, pxr::UsdStageRefPtr stage,
    bool simulate_ran)
    : mm_state(std::make_unique<aerial::sim::mm::mobility_state>(
          ext_vertices, ext_triangles_outdoor, ext_triangles_indoor)),
      aerial_stage(stage), internal_state(init), time_idx(0),
      simulate_ran(simulate_ran),
      tr(std::make_unique<trainer>(
          aerial::sim::simulation_state_log_callback)) {
  for (auto &s : streams) {
    ASIM_CUDA_CHECK(cudaStreamCreate(&s));
  }
}

simulation_state::~simulation_state() {
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
}

int32_t simulation_state::trainer_scenario(const scenario_info &si,
                                           const ru_ue_info &rui) {
  return tr->scenario(si, rui);
}

training_info simulation_state::trainer_append_cfr(
    const time_info &ti, const ru_ue_info &info,
    std::vector<ru_association_info> &ru_association_infos,
    const std::vector<std::vector<cuda::std::complex<float>>>
        &all_cfr_results) {
  return tr->append_cfr(ti, info, ru_association_infos, all_cfr_results);
}

int32_t simulation_state::build_tx_info(
    const pxr::UsdPrim &tx_node, const emsolver::AntennaPanel &tx_panel,
    const pxr::SdfPath &tx_panel_path, emsolver::TXInfo &txi,
    std::unordered_map<int, ru_du_info> &ru_du_info_map, const int &tx_id,
    const int &du_id, const bool &du_manual_assign,
    const float &subcarrier_spacing, const int &fft_size,
    const float &radiated_power_w, const int &panel_id,
    const double &height_from_cylinder_base, const double &mech_azimuth_deg,
    const double &mech_tilt_deg) {

  auto translate_attr = tx_node.GetAttribute(pxr::TfToken("xformOp:translate"));
  pxr::GfVec3d translation;
  translate_attr.Get(&translation);

  // convert deg to rad
  const float mech_azimuth_rads =
      static_cast<float>(mech_azimuth_deg) * M_PI_F / 180.0;
  const float mech_tilt_rads =
      static_cast<float>(mech_tilt_deg) * M_PI_F / 180.0;

  const float cos_alpha = std::cos(mech_azimuth_rads);
  const float sin_alpha = std::sin(mech_azimuth_rads);
  const float cos_beta = std::cos(mech_tilt_rads);
  const float sin_beta = std::sin(mech_tilt_rads);

  // (1) translation[2] is at center of RU cylinder/prim
  // (2) height_from_cylinder_base is in meters
  // (3) height_from_cylinder_center is half the height of the RU cylinder
  // (4) tx_center_z is calculating the center of the antenna panel, not the RU
  // cylinder (5) the antenna panel is at the top of the RU cylinder, where the
  // rays originate from
  const auto meters_per_unit =
      pxr::UsdGeomGetStageMetersPerUnit(get_aerial_stage().int_stage);
  const auto scale_height = 1.0f / meters_per_unit;
  const auto height_from_cylinder_center =
      static_cast<float>(height_from_cylinder_base * scale_height / 2.0);
  const auto tx_center_z =
      static_cast<float>(translation[2] + height_from_cylinder_center);

  emsolver::Matrix4x4 xform = {cos_alpha * cos_beta,
                               -sin_alpha,
                               cos_alpha * sin_beta,
                               static_cast<float>(translation[0]),
                               sin_alpha * cos_beta,
                               cos_alpha,
                               sin_alpha * sin_beta,
                               static_cast<float>(translation[1]),
                               -sin_beta,
                               0.0,
                               cos_beta,
                               tx_center_z,
                               0.0,
                               0.0,
                               0.0,
                               1.0};

  txi.tx_center = make_float3(translation[0], translation[1], tx_center_z);

  txi.Ttx = xform;
  txi.tx_ID = tx_id;
  ru_du_info_map[tx_id] = {du_id, static_cast<bool>(du_manual_assign)};

  txi.panel_id.push_back(panel_id);
  txi.height = height_from_cylinder_base;
  txi.mech_azimuth_deg = mech_azimuth_deg;
  txi.mech_tilt_deg = mech_tilt_deg;
  txi.carrier_freq = tx_panel.reference_freq;
  txi.subcarrier_spacing = subcarrier_spacing;
  txi.fft_size = fft_size;
  txi.radiated_power = radiated_power_w;
  txi.antenna_names = tx_panel.antenna_names;
  txi.antenna_pattern_indices = tx_panel.antenna_pattern_indices;
  txi.num_loc_antenna_horz = tx_panel.num_loc_antenna_horz;
  txi.num_loc_antenna_vert = tx_panel.num_loc_antenna_vert;

  std::vector<float3> antennas;
  std::vector<std::pair<int, int>> antennas_ij;

  if (tx_panel.num_loc_antenna_horz * tx_panel.num_loc_antenna_vert >
      MAX_NUM_TX_ANTENNAS) {
    LOG(ERROR)
        << "The total number antennas at RU node "
        << tx_node.GetPath().GetString() << " (using panel "
        << tx_panel_path.GetString()
        << ") is larger than the maximum allowed number (currently set to "
        << MAX_NUM_TX_ANTENNAS
        << ")! Choose another panel or reduce the antenna numbers before "
           "running the EM Solver again";
    return -1;
  }

  const auto scale_loc =
      0.01f / meters_per_unit; // spacing in cm, scale to unit

  for (int i = 0; i < tx_panel.num_loc_antenna_horz; i++) {
    float loc_i = -(int(tx_panel.num_loc_antenna_horz) - 1) / 2.0 *
                      tx_panel.antenna_spacing_horz +
                  i * tx_panel.antenna_spacing_horz;
    for (int j = 0; j < tx_panel.num_loc_antenna_vert; j++) {
      float loc_j = -(int(tx_panel.num_loc_antenna_vert) - 1) / 2.0 *
                        tx_panel.antenna_spacing_vert +
                    j * tx_panel.antenna_spacing_vert;
      antennas.push_back(
          make_float3(0.0f, loc_i * scale_loc,
                      loc_j * scale_loc)); // spacing in m, scale to unit
      antennas_ij.push_back(std::make_pair(i, j));
    }
  }

  for (auto &antenna : antennas) {
    antenna =
        make_float3(xform.m[0][0] * antenna.x + xform.m[0][1] * antenna.y +
                        xform.m[0][2] * antenna.z + xform.m[0][3],
                    xform.m[1][0] * antenna.x + xform.m[1][1] * antenna.y +
                        xform.m[1][2] * antenna.z + xform.m[1][3],
                    xform.m[2][0] * antenna.x + xform.m[2][1] * antenna.y +
                        xform.m[2][2] * antenna.z + xform.m[2][3]);
  }

  // antenna rotation angles
  txi.antenna_rotation_angles.push_back(
      make_float3(tx_panel.antenna_roll_angle_first_polz, mech_tilt_rads,
                  mech_azimuth_rads));

  txi.dual_polarized_antenna = tx_panel.dual_polarized;
  if (tx_panel.dual_polarized) {
    txi.antenna_rotation_angles.push_back(
        make_float3(tx_panel.antenna_roll_angle_second_polz, mech_tilt_rads,
                    mech_azimuth_rads));
  }

  txi.loc_antenna = antennas;
  txi.ij_antenna = antennas_ij;

  return 0;
}

ASIM_EXPORT int32_t simulation_state::load_deployed_rus() {
    // Clear existing data
    get_aerial_stage().tx_info.clear();
    get_aerial_stage().du_info_map.clear();
    get_aerial_stage().ru_du_info_map.clear();

    // Hardcoded DUs (Digital Units)
    std::vector<du_info> dus = {
        {.id = 1, .subcarrier_spacing = 30.0, .fft_size = 4096, .num_antennas = 4, .max_channel_bandwidth = 100.0, .center=make_float3(3400.91582f,3400.91582f, 15000.0f)}
        // {.id = 2, .subcarrier_spacing = 30.0, .fft_size = 4096}
    };

    // Hardcoded RUs (Radio Units)
    std::vector<emsolver::TXInfo> rus = {
        {
            .tx_ID = 1,  // RU ID
            .tx_center = make_float3(5000.0f, 5000.0f, 4000.0f),  // Position (mm)
    
            // Transformation matrix (identity matrix as default)
            .Ttx = {
                1.0f, 0.0f, 0.0f, 0.0f,  // Row 0
                0.0f, 1.0f, 0.0f, 0.0f,  // Row 1
                0.0f, 0.0f, 1.0f, 0.0f,  // Row 2
                0.0f, 0.0f, 0.0f, 1.0f   // Row 3
            },
    
            // Panel configuration (single panel with ID 0)
            .panel_id = {0},
            .height = 1.5f,  // Height in meters
            .mech_azimuth_deg = 175.18913f,  // Mechanical azimuth (degrees)
            .mech_tilt_deg = 0.0f,           // Mechanical tilt (degrees)
    
            // RF parameters
            .carrier_freq = 3600.0f,       // 3.6 GHz
            .carrier_bandwidth = 100.0f,   // 100 MHz
            .subcarrier_spacing = 30.0f,   // 30 kHz
            .fft_size = 4096,              // FFT size
            .radiated_power = 1.2f,        // 1 Watt (30 dBm)
    
            // Antenna configuration (4x half-wave dipoles)
            .antenna_names = {"halfwave_dipole", "halfwave_dipole", "halfwave_dipole", "halfwave_dipole"},
            .antenna_pattern_indices = {0, 0, 0, 0},  // All use pattern index 0
            .dual_polarized_antenna = true,
    
            // Antenna rotation (roll, tilt, azimuth in radians)
            .antenna_rotation_angles = {
                make_float3(0.0f, 0.0f, 3.057f),  // Polarization 1
                make_float3(M_PI_2, 0.0f, 3.057f) // Polarization 2 (if dual-polarized)
            },
    
            // Antenna array layout (2x2 grid)
            .num_loc_antenna_horz = 2,
            .num_loc_antenna_vert = 1,
            .loc_antenna = {
                make_float3(-0.05f, 0.0f, 0.05f),  // Antenna 1
                make_float3(0.05f, 0.0f, 0.05f),   // Antenna 2
                make_float3(-0.05f, 0.0f, -0.05f),  // Antenna 3
                make_float3(0.05f, 0.0f, -0.05f)    // Antenna 4
            },
            .ij_antenna = {
                {0, 0},  // Antenna 1 (i,j indices)
                {1, 0},   // Antenna 2
                {0, 1},   // Antenna 3
                {1, 1}    // Antenna 4
            }
        }
    };

    // Add DUs to the map
    for (const auto& du : dus) {
        get_aerial_stage().du_info_map[du.id] = du;
    }

    // Add RUs and their DU associations
    for (const auto& ru : rus) {
        // Assign RUs to DUs in round-robin fashion
        int du_id = (ru.tx_ID - 1) % dus.size() + 1;
        get_aerial_stage().ru_du_info_map[ru.tx_ID] = {du_id, true}; // true = manual assignment
        get_aerial_stage().tx_info.push_back(ru);
        
        // Add RU index to DU's list
        get_aerial_stage().du_info_map[du_id].tx_info_idx.push_back(
            get_aerial_stage().tx_info.size() - 1);
    }

    // Log deployment
    LOG(INFO) << "Deployed " << get_aerial_stage().tx_info.size() << " RUs and " 
              << get_aerial_stage().du_info_map.size() << " DUs with hardcoded values";

    return 0;
}

int32_t simulation_state::build_rx_info(unsigned int k,
                                        const emsolver::AntennaPanel &rx_panel,
                                        const int &rx_panel_id,
                                        const pxr::SdfPath &rx_panel_path) {
  const std::vector<aerial::sim::mm::user> &users = get_mobility_users();

  unsigned int s = get_time_idx();

  const float3 point = users.at(k).flattened_samples.at(s).point;

  const auto meters_per_unit =
      pxr::UsdGeomGetStageMetersPerUnit(get_aerial_stage().int_stage);
  const auto scale_height = 1.0f / meters_per_unit;
  const pxr::GfVec3d translation(
      point.x, point.y, point.z + users.at(k).params.height * scale_height);

  const float3 orientation = users.at(k).flattened_samples.at(s).orientation;

  // get mech_az and mech_tilt of the user
  float mech_az, mech_el;
  if (orientation.z > 1.0f - 1e-5f) {
    mech_el = M_PI_F / 2.0f;
    mech_az = 0.0f;
  } else if (orientation.z < -1.0f + 1e-5f) {
    mech_el = -M_PI_F / 2.0f;
    mech_az = 0.0f;
  } else {
    mech_el = M_PI_F / 2.0f - std::acos(orientation.z);
    mech_az = std::atan2(orientation.y, orientation.x);
  }

  const float cos_alpha = std::cos(mech_az);
  const float sin_alpha = std::sin(mech_az);
  const float ue_params_mech_tilt_rads =
      users.at(k).params.mech_tilt_degs * M_PI_F / 180.0;

  const float mech_tilt_rads =
      -mech_el + ue_params_mech_tilt_rads; // add the panel's tilt angle
  const float cos_beta = std::cos(mech_tilt_rads);
  const float sin_beta = std::sin(mech_tilt_rads);

  // get xfrom of the panel
  emsolver::Matrix4x4 xform = {cos_alpha * cos_beta,
                               -sin_alpha,
                               cos_alpha * sin_beta,
                               static_cast<float>(translation[0]),
                               sin_alpha * cos_beta,
                               cos_alpha,
                               sin_alpha * sin_beta,
                               static_cast<float>(translation[1]),
                               -sin_beta,
                               0.0,
                               cos_beta,
                               static_cast<float>(translation[2]),
                               0.0,
                               0.0,
                               0.0,
                               1.0};

  get_aerial_stage().rx_info.at(k).rx_center =
      make_float3(translation[0], translation[1], translation[2]);

  get_aerial_stage().rx_info.at(k).Trx = xform;
  get_aerial_stage().rx_info.at(k).rx_ID = users.at(k).uID;
  get_aerial_stage().rx_info.at(k).radiated_power =
      users.at(k).params.radiated_power_w;
  get_aerial_stage().rx_info.at(k).panel_id.push_back(rx_panel_id);

  get_aerial_stage().rx_info.at(k).antenna_names = rx_panel.antenna_names;
  get_aerial_stage().rx_info.at(k).antenna_pattern_indices =
      rx_panel.antenna_pattern_indices;

  get_aerial_stage().rx_info.at(k).num_loc_antenna_horz =
      rx_panel.num_loc_antenna_horz;
  get_aerial_stage().rx_info.at(k).num_loc_antenna_vert =
      rx_panel.num_loc_antenna_vert;

  std::vector<float3> antennas;
  std::vector<std::pair<int, int>> antennas_ij;

  if (rx_panel.num_loc_antenna_horz * rx_panel.num_loc_antenna_vert >
      MAX_NUM_RX_ANTENNAS) {
    LOG(ERROR)
        << "The total number of UE panel antennas (at "
        << rx_panel_path.GetString()
        << ") is larger than the maximum allowed number (currently set to "
        << MAX_NUM_RX_ANTENNAS
        << ")! Choose another panel or reduce the antenna numbers before "
           "running the EM Solver again";
    return -1;
  }

  const auto scale_loc =
      0.01f / meters_per_unit; // spacing in cm, scale to unit

  for (int i = 0; i < rx_panel.num_loc_antenna_horz; i++) {
    float loc_i = -(int(rx_panel.num_loc_antenna_horz) - 1) / 2.0 *
                      rx_panel.antenna_spacing_horz +
                  i * rx_panel.antenna_spacing_horz;
    for (int j = 0; j < rx_panel.num_loc_antenna_vert; j++) {
      float loc_j = -(int(rx_panel.num_loc_antenna_vert) - 1) / 2.0 *
                        rx_panel.antenna_spacing_vert +
                    j * rx_panel.antenna_spacing_vert;
      antennas.push_back(
          make_float3(0.0f, loc_i * scale_loc,
                      loc_j * scale_loc)); // spacing in m, scale to unit
      antennas_ij.push_back(std::make_pair(i, j));
    }
  }

  for (auto &antenna : antennas) {
    antenna =
        make_float3(xform.m[0][0] * antenna.x + xform.m[0][1] * antenna.y +
                        xform.m[0][2] * antenna.z + xform.m[0][3],
                    xform.m[1][0] * antenna.x + xform.m[1][1] * antenna.y +
                        xform.m[1][2] * antenna.z + xform.m[1][3],
                    xform.m[2][0] * antenna.x + xform.m[2][1] * antenna.y +
                        xform.m[2][2] * antenna.z + xform.m[2][3]);
  }

  // antenna rotation angles
  get_aerial_stage().rx_info.at(k).antenna_rotation_angles.push_back(
      make_float3(rx_panel.antenna_roll_angle_first_polz, mech_tilt_rads,
                  mech_az));

  get_aerial_stage().rx_info.at(k).dual_polarized_antenna =
      rx_panel.dual_polarized;
  if (rx_panel.dual_polarized) {
    get_aerial_stage().rx_info.at(k).antenna_rotation_angles.push_back(
        make_float3(rx_panel.antenna_roll_angle_second_polz, mech_tilt_rads,
                    mech_az));
  }

  get_aerial_stage().rx_info.at(k).loc_antenna = antennas;
  get_aerial_stage().rx_info.at(k).ij_antenna = antennas_ij;

  return 0;
}

int32_t simulation_state::update_ues() {
  const std::vector<aerial::sim::mm::user> &users = get_mobility_users();

  for (unsigned int k = 0; k < users.size(); k++) {
    const auto rx_panel_id = users.at(k).params.panel;
    const auto &rx_panel =
        get_aerial_stage().antenna_info.panels.at(rx_panel_id);

    if (build_rx_info(k, rx_panel, rx_panel_id,
                      pxr::SdfPath(rx_panel.panel_name)) != 0) {
      return -1;
    }
  }

  return 0;
}

int32_t simulation_state::generate_ue_mobility(
    const aerial::sim::mm::bounding_box_info *spawn_zone,
    std::vector<aerial::sim::mm::user> &prior_users) {
  return mm_state->generate_ue_mobility(spawn_zone, prior_users);
}

int32_t simulation_state::add_manual_user(
    const std::vector<float3> &waypoints, unsigned int ID,
    aerial::sim::mm::waypoint_config wp_cfg,
    const std::vector<aerial::sim::mm::waypoint_params> &wp_params_list,
    std::vector<aerial::sim::mm::user> &prior_users) {
  return mm_state->add_manual_user(waypoints, ID, wp_cfg, wp_params_list,
                                   prior_users);
}

int32_t simulation_state::load_deployed_ues() {
  get_aerial_stage().rx_info.clear();

  auto aerial_visibility = pxr::TfToken("aerialVisibility");

  const std::vector<aerial::sim::mm::user> &users = get_mobility_users();
  if (get_mm_config().users != users.size()) {
    LOG(ERROR) << __func__ << ":" << __LINE__ << " cfg.users ("
               << get_mm_config().users << ") != users.size (" << users.size()
               << ").  Stop any active simulations, and re-generate UEs.";
    return -1;
  }

  for (unsigned int k = 0; k < users.size(); k++) {
    std::ostringstream label;

    label << "/UEs/ue_" << std::setw(4) << std::setfill('0') << users.at(k).uID;

    auto node =
        get_aerial_stage().int_stage->GetPrimAtPath(pxr::SdfPath(label.str()));

    if (!node.IsValid()) {
      LOG(ERROR) << __func__ << ":" << __LINE__ << " prim " << label.str()
                 << " is not valid.  Stop any active simulations, and "
                    "re-generate UEs.";
      return -1;
    }

    if (auto attr = node.GetAttribute(aerial_visibility)) {
      bool visibility = false;
      attr.Get(&visibility);
      if (!visibility) {
        LOG(VERBOSE) << "Skipping " << node.GetPath()
                     << " due to visiblity attribute set to invisible";
        continue;
      }
    }

    if (!node.GetAttribute(pxr::TfToken("aerial:ue:user_id"))) {
      LOG(ERROR) << "Couldn't read user_id for node "
                 << node.GetPath().GetString();
      return -1;
    }

    aerial::sim::mm::user_equipment params{};
    params.height =
        get_mm_config()
            .height_ue; // height (in meters) read earlier
                        // /Scenario.sim:ue:height in getMobilityParams()

    double radiated_power_dbm = 0.0;
    if (auto attr = node.GetAttribute(pxr::TfToken("aerial:ue:radiated_power"));
        !attr || !attr.Get(&radiated_power_dbm)) {
      LOG(ERROR) << "Couldn't read radiated power for node "
                 << node.GetPath().GetString();
      return -1;
    }

    params.radiated_power_w =
        std::pow(10.0, 0.1 * (radiated_power_dbm -
                              30)); // convert dBm to W in linear scale

    if (auto attr = node.GetAttribute(pxr::TfToken("aerial:ue:mech_tilt"));
        !attr || !attr.Get(&params.mech_tilt_degs)) {

      LOG(ERROR) << "Couldn't read mech. tilt for node "
                 << node.GetPath().GetString();
      return -1;
    }

    pxr::TfToken panel_type;
    if (auto attr = node.GetAttribute(pxr::TfToken("aerial:ue:panel_type"));
        !attr || !attr.Get(&panel_type)) {
      LOG(ERROR) << "Couldn't read panel type for node "
                 << node.GetPath().GetString();
      return -1;
    }

    auto rx_panel_path =
        pxr::SdfPath("/Panels").AppendPath(pxr::SdfPath(panel_type));
    LOG(DEBUG) << "Found UE panel node at: " << rx_panel_path;

    if (auto attr = node.GetAttribute(pxr::TfToken("aerial:ue:bler_target"));

        !attr || !attr.Get(&(params.bler_target))) {
      LOG(ERROR) << "Couldn't read BLER target for node "
                 << node.GetPath().GetString();
      return -1;
    }

    get_aerial_stage().rx_info.push_back(emsolver::RXInfo{});
    const auto rx_panel_id = panel_name_to_index(
        panel_type.GetString(), get_aerial_stage().antenna_info.panels);
    if (rx_panel_id < 0) {
      return -1;
    }

    params.panel = rx_panel_id;
    if (mm_state->deploy_user_equipment(k, params) < 0) {
      LOG(ERROR) << "Failed to deploy user equipment for user " << k;
      return -1;
    }

    const auto &rx_panel =
        get_aerial_stage().antenna_info.panels.at(rx_panel_id);

    if (build_rx_info(k, rx_panel, params.panel, rx_panel_path) != 0) {
      return -1;
    }
  }

  return 0;
}

int32_t simulation_state::update_stage_sim_params() {
  omniClientLiveWaitForPendingUpdates();

  get_aerial_stage().geometry_info.building_mesh.clear();
  get_aerial_stage().geometry_info.terrain_mesh.clear();
  get_aerial_stage().geometry_info.material_dict.clear();

  if (buildMaterialDictionary(get_aerial_stage().int_stage,
                              get_aerial_stage().geometry_info) != 0) {
    return -1;
  }
  if (getStageMeshBuildings(get_aerial_stage().int_stage,
                            get_aerial_stage().geometry_info) != 0) {
    return -1;
  }

  return 0;
}

int32_t simulation_state::consume(const trigger input) {
  int32_t ret = 0;
  switch (internal_state) {
  case simulation_state::init:

    switch (input) {
    case simulation_state::sim_mobility:
      internal_state = mobility_available;
      reset_time_idx();
      ret = clear_raytracing_results(db().settings());
      break;
    }

    break;

  case simulation_state::mobility_available:
    switch (input) {
    case simulation_state::sim_play:
      internal_state = ongoing_sim;
      break;
    case simulation_state::sim_mobility:
      reset_time_idx();
      ret = clear_raytracing_results(db().settings());
      break;
    }
    break;

  case simulation_state::ongoing_sim:
    switch (input) {
    case simulation_state::sim_pause:
      internal_state = paused_sim;
      break;
    case simulation_state::sim_stop:
      internal_state = mobility_available;
      reset_time_idx();
      break;
    }
    break;

  case simulation_state::paused_sim:

    switch (input) {
    case simulation_state::sim_play:
      internal_state = ongoing_sim;
      break;
    case simulation_state::sim_stop:
      internal_state = mobility_available;
      reset_time_idx();
      break;
    }
    break;
  }

  return ret;
}
} // namespace aerial::sim
