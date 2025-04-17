#include "cnt_utils.hpp"

#include <iostream>
#include <stdexcept>

#include "pxr/base/gf/matrix4d.h"
#include <cuda_runtime.h>

#include "logger.hpp"
#include "json_loader_for_user.hpp"
#include "pxr/usd/usdGeom/xformOp.h"
#include "pxr/usd/usdGeom/metrics.h"
#include "mobility_api.hpp"
#include "cnt_utils.hpp"
#include "simulation_state.hpp"
#include <OmniClient.h>
#include "nlohmann/json.hpp"
#include <mutex>
#include "pxr/usd/usdGeom/xformCommonAPI.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3d.h"
using namespace pxr;

namespace {
double get_scs_from_stage(pxr::UsdStageRefPtr stage) {
  double scs_khz = 0.0;
  bool warn = false;
  for (const auto &du :
       stage->GetPrimAtPath(pxr::SdfPath("/DUs")).GetAllChildren()) {
    double this_scs_khz = 0.0;
    if (du.GetAttribute(pxr::TfToken("aerial:du:subcarrier_spacing"))
            .Get(&this_scs_khz) &&
        this_scs_khz > 0) {
      if (scs_khz > 0.0 && scs_khz != this_scs_khz) {
        warn = true;
        LOG(WARNING) << du.GetName()
                     << " has different subcarrier spacing than others ("
                     << this_scs_khz << "khz != " << scs_khz << "khz)";
      }
      scs_khz = std::max(this_scs_khz, scs_khz);
    }
  }
  if (scs_khz == 0.0) {
    scs_khz = 30;
    LOG(WARNING) << "Did not find DU with valid subcarrier spacing so "
                    "defaulting to 30khz";
  }
  if (warn) {
    LOG(WARNING) << "Found different DU subcarrier spacings so using max "
                 << scs_khz << "khz";
  }

  return scs_khz;
}
} // namespace

namespace aerial::sim::cnt {
ASIM_EXPORT std::pair<aerial::sim::simulation_state *, int32_t>
setupMainState(UsdStageRefPtr stage) {
  std::vector<int3> triangles_outdoor;
  std::vector<int3> triangles_indoor;
  std::vector<float3> vertices;

  UsdGeomXformCache xcache;

  auto node = stage->GetPrimAtPath(SdfPath("/World/mobility_domain"));

  GfMatrix4d xform = xcache.GetLocalToWorldTransform(node).GetTranspose();

  if (auto attr = node.GetAttribute(TfToken("xformOp:translate"))) {
    GfVec3d T;
    if (!attr.Get(&T)) {
      LOG(VERBOSE) << "T not read correctly for node: "
                   << node.GetPath().GetString();
    }
  }

  if (auto attr = node.GetAttribute(TfToken("points"))) {
    // VtValue points;
    VtArray<GfVec3f> points;
    if (!attr.Get(&points)) {
      LOG(ERROR) << "Couldn't get 'points' attribute from "
                 << node.GetPath().GetString();
      return std::make_pair(nullptr, -1);
    }

    if (auto attr = node.GetAttribute(TfToken("faceVertexIndices"))) {
      VtArray<int> indices;
      if (!attr.Get(&indices)) {
        LOG(ERROR) << "Couldn't get 'faceVertexIndices' attribute from "
                   << node.GetPath().GetString();
        return std::make_pair(nullptr, -1);
      }
      if (auto attr = node.GetAttribute(TfToken("faceVertexCounts"))) {
        VtArray<int> counts;
        if (!attr.Get(&counts)) {
          LOG(ERROR) << "Couldn't get 'faceVertexCounts' attribute from "
                     << node.GetPath().GetString();
          return std::make_pair(nullptr, -1);
        }

        HdMeshTopology topology(TfToken("none"), TfToken("rightHanded"), counts,
                                indices);
        HdMeshUtil meshUtil(&topology, node.GetPath());
        VtVec3iArray new_indices;
        VtIntArray primitiveParams;
        meshUtil.ComputeTriangleIndices(&new_indices, &primitiveParams);
        VtArray<int> mobility_indices;

        if (auto attr = node.GetAttribute(TfToken("primvars:MobilityType"))) {
          attr.Get(&mobility_indices);
        } else {
          LOG(DEBUG) << "primvars:mobility_type not found";
        }
        auto num_indoor_triangles = std::accumulate(mobility_indices.begin(),
                                                    mobility_indices.end(), 0);
        LOG(DEBUG) << "num indoor triangles: " << num_indoor_triangles
                   << ", num mobility indices: " << mobility_indices.size();
        triangles_indoor.reserve(num_indoor_triangles);
        triangles_outdoor.reserve(new_indices.size() - num_indoor_triangles);

        for (int i = 0; i < new_indices.size(); i++) {
          int3 triangle_index = make_int3(new_indices[i][0], new_indices[i][1],
                                          new_indices[i][2]);
          if ((num_indoor_triangles > 0) && (mobility_indices[i] == 1)) {
            triangles_indoor.push_back(triangle_index);
            num_indoor_triangles--;
          } else {
            triangles_outdoor.push_back(triangle_index);
          }

          // sanity check to ensure that element is a triangle
          if (counts[i] != 3) {
            LOG(ERROR) << "Mobility mesh element " << i
                       << " is not a triangle (has vertex count != 3)";
            return std::make_pair(nullptr, -1);
          }
        }

        vertices.reserve(points.size());

        for (int i = 0; i < points.size(); i++) {
          auto vtx0 = points[i];
          GfVec4d vtx0_4(vtx0[0], vtx0[1], vtx0[2], 1.0);
          GfVec4d Tv0 = xform * vtx0_4;
          vertices.push_back(make_float3(Tv0[0], Tv0[1], Tv0[2]));
        }
      } else {
        LOG(ERROR) << "Couldn't get 'faceVertexCounts' attribute from "
                   << node.GetPath().GetString();
        return std::make_pair(nullptr, -1);
      }
    } else {
      LOG(ERROR) << "Couldn't get 'faceVertexIndices' attribute from "
                 << node.GetPath().GetString();
      return std::make_pair(nullptr, -1);
    }
  } else {
    LOG(ERROR) << "Couldn't get 'points' attribute from "
               << node.GetPath().GetString();
    return std::make_pair(nullptr, -1);
  }

  try {
    constexpr auto simulate_ran_default_value = false;

    return std::make_pair(new aerial::sim::simulation_state(
                              vertices, triangles_outdoor, triangles_indoor,
                              stage, simulate_ran_default_value),
                          0);
  } catch (const std::runtime_error &e) {
    LOG(ERROR) << "simulation state constructor threw exception " << e.what();
    return std::make_pair(nullptr, -1);
  }
}

ASIM_EXPORT void getRANSimAttribute(aerial::sim::simulation_state *state) {
  UsdStageRefPtr stage = state->get_aerial_stage().int_stage;
  omniClientLiveProcess();

  {
    auto node = stage->GetPrimAtPath(SdfPath("/Scenario"));
    auto attr = node.GetAttribute(TfToken("sim:is_full"));
    if (attr) {
      attr.Get(&(state->simulate_ran));
    }
  }
}

ASIM_EXPORT void getMobilityParams(aerial::sim::simulation_state *state) {
  aerial::sim::mm::config cfg{};

  UsdStageRefPtr stage = state->get_aerial_stage().int_stage;

  omniClientLiveProcess();

  auto node = stage->GetPrimAtPath(SdfPath("/Scenario"));

 // node.GetAttribute(TfToken("sim:duration")).Get(&(cfg.duration));
  //node.GetAttribute(TfToken("sim:interval")).Get(&(cfg.interval));
  cfg.duration = 100.0f; //hardcoded values
  cfg.interval = 0.1f; //hardcoded values 
  node.GetAttribute(TfToken("sim:num_users")).Get(&(cfg.users));
  node.GetAttribute(TfToken("sim:perc_indoor_procedural_ues"))
      .Get(&(cfg.percentage_indoor_users));

  node.GetAttribute(TfToken("sim:batches")).Get(&cfg.batches);
  node.GetAttribute(TfToken("sim:slots_per_batch")).Get(&(cfg.slots_per_batch));
  node.GetAttribute(TfToken("sim:samples_per_slot"))
      .Get(&(cfg.samples_per_slot));

  if (cfg.batches == 0) {
    cfg.batches = 1;
    LOG(WARNING) << "Did not find attribute sim:batches, so setting to "
                    "default value of "
                 << cfg.batches;
  }

  const bool no_duration_interval = cfg.duration == 0 || cfg.interval == 0;
  cfg.slot_symbol_mode = no_duration_interval;
  if (no_duration_interval) {
    LOG(INFO) << "Did not find attributes sim:duration/sim:interval with a "
                 "positive value so "
                 "using slot/symbol instead";
    if (cfg.samples_per_slot == 0) {
      cfg.samples_per_slot = 1;
    }
    if (cfg.slots_per_batch == 0) {
      cfg.slots_per_batch = 1;
    }

    const auto scs_khz =
        get_scs_from_stage(state->get_aerial_stage().int_stage);
    constexpr float subframe_time_in_seconds = 1.0e-3f;
    const float slots_per_subframe = scs_khz / 15;
    const float slot_time_in_seconds =
        subframe_time_in_seconds / slots_per_subframe;
    cfg.duration = cfg.slots_per_batch * slot_time_in_seconds;
    cfg.interval = slot_time_in_seconds / cfg.samples_per_slot;
  } else {
    LOG(INFO) << "Found attributes sim:duration/sim:interval so "
                 "using default value of 1 sample per slot";
    cfg.samples_per_slot = 1;
    cfg.slots_per_batch = 0;
  }

  cfg.speed = make_float2(0, 0);

  node.GetAttribute(TfToken("sim:ueMinSpeed")).Get(&(cfg.speed.x));
  node.GetAttribute(TfToken("sim:ueMaxSpeed")).Get(&(cfg.speed.y));
  node.GetAttribute(TfToken("sim:is_seeded")).Get(&(cfg.is_seeded));
  node.GetAttribute(TfToken("sim:seed")).Get(&(cfg.seed));

  TfToken gnb_panel;
  TfToken ue_panel;

  node.GetAttribute(TfToken("sim:gnb:panel_type")).Get(&gnb_panel);
  node.GetAttribute(TfToken("sim:ue:panel_type")).Get(&ue_panel);
  node.GetAttribute(TfToken("sim:ue:height")).Get(&(cfg.height_ue));

  LOG(DEBUG) << "Scenario users: " << cfg.users << ", batches: " << cfg.batches
             << ", slot/symbol mode: " << cfg.slot_symbol_mode
             << ", slots_per_batch: " << cfg.slots_per_batch
             << ", samples_per_slot: " << cfg.samples_per_slot
             << ", duration (per batch): " << cfg.duration
             << ", interval: " << cfg.interval
             << ", ue_min_speed=" << cfg.speed.x
             << ", ue_max_speed=" << cfg.speed.y
             << ", is_seeded=" << cfg.is_seeded << ", seed=" << cfg.seed
             << std::endl;

  cfg.panel_gnb = gnb_panel.GetString();
  cfg.panel_ue = ue_panel.GetString();

  cfg.scale = UsdGeomGetStageMetersPerUnit(stage);

  // stage->GetMetadata(pxr::TfToken("metersPerUnit"), &(cfg.scale));

  cfg.scale = 1 / cfg.scale;
  LOG(VERBOSE) << "scale=" << cfg.scale;

  state->set_mm_config(cfg);
}

ASIM_EXPORT int32_t getManualUsers(aerial::sim::simulation_state *state,
                                   std::vector<aerial::sim::mm::user> &users) {
  static const TfToken TOKEN_USER_ID("aerial:ue:user_id");
  static const TfToken TOKEN_MANUAL("aerial:ue:manual");
  static const TfToken TOKEN_WAYPOINTS("aerial:ue:waypoints");
  static const TfToken TOKEN_WAYPOINT_SPEED("aerial:ue:waypoint_speed");
  static const TfToken TOKEN_WAYPOINT_PAUSE(
      "aerial:ue:waypoint_pause_duration");
  static const TfToken TOKEN_WAYPOINT_AZIMUTH(
      "aerial:ue:waypoint_azimuth_offset");
  static const TfToken TOKEN_HEIGHT("height");
  static const TfToken TOKEN_RADIUS("radius");
  static const TfToken TOKEN_NUM_USERS("sim:num_users");

  const auto &cfg = state->get_mm_config();
  UsdStageRefPtr stage = state->get_aerial_stage().int_stage;
  omniClientLiveProcess();

  int32_t ret = 0;
  unsigned int num_manual_users = 0;

  // loading from JSON
  aerial::sim::utils::JsonUserLoader loader(
      "src/sim_controller/lib/user_config.json");
  bool jsonLoaded = loader.loadUsers();

  if (jsonLoaded) {
    for (const auto &user : loader.getUsers()) {
      double offset_z_height = 0.0;
      double offset_z_radius = 0.0;
      double offset_z = offset_z_height / 2 + offset_z_radius;

      std::vector<aerial::sim::mm::waypoint_params> wp_params_list;
      std::vector<float3> waypoint_vec;

      auto waypoint_config =
          user.waypoint_config == "user_defined"
              ? aerial::sim::mm::waypoint_config::user_defined
              : aerial::sim::mm::waypoint_config::none;

      if (!user.waypoints.empty()) {
        waypoint_vec.reserve(user.waypoints.size());
        wp_params_list.reserve(user.waypoints.size());

        for (const auto &wp : user.waypoints) {
          waypoint_vec.push_back(make_float3(wp.x, wp.y, wp.z));
          wp_params_list.push_back(
              {static_cast<float>(wp.speed_mps * cfg.scale), wp.stop_sec,
               static_cast<float>(wp.azimuth_offset_rad * M_PI_F / 180.0f)});
        }
      } else if (user.has_position) {
        float3 point =
            make_float3(user.pos_x, user.pos_y, user.pos_z - offset_z);
        waypoint_vec.push_back(point);
      }

      if ((ret = state->add_manual_user(waypoint_vec, user.id, waypoint_config,
                                        wp_params_list, users)) != 0) {
        return ret;
      }
      num_manual_users += 1;
    }
  } else {
    // Fall back to original USD stage traversal
    auto prims = stage->Traverse();

    for (auto k = prims.begin(); k != prims.end(); k++) {
      if (!k->IsValid()) {
        continue;
      }

      if (k->HasProperty(TOKEN_USER_ID)) {
        unsigned int buffer;
        bool is_manual;

        k->GetAttribute(TOKEN_USER_ID).Get(&buffer);
        k->GetAttribute(TOKEN_MANUAL).Get(&is_manual);

        if (is_manual) {
          unsigned int ID = buffer;

          double offset_z_height;
          double offset_z_radius;

          k->GetAttribute(TOKEN_HEIGHT).Get(&offset_z_height);
          k->GetAttribute(TOKEN_RADIUS).Get(&offset_z_radius);

          double offset_z = offset_z_height / 2 + offset_z_radius;

          pxr::VtArray<pxr::GfVec3f> waypoints;
          if (const auto wps = k->GetAttribute(TOKEN_WAYPOINTS);
              wps.IsValid()) {
            if (!wps.Get(&waypoints)) {
              LOG(DEBUG) << "Did not find aerial:ue:waypoints attribute for UE " << ID;
            }
          }

          pxr::VtArray<float> waypoint_attributes_speed_list{};
          pxr::VtArray<float> waypoint_attributes_pause_duration_list{};
          pxr::VtArray<float> waypoint_attributes_azimuth_offset_list{};

          const auto wp_attributes_speed =
              k->GetAttribute(TOKEN_WAYPOINT_SPEED);
          const auto wp_attributes_pause_duration =
              k->GetAttribute(TOKEN_WAYPOINT_PAUSE);
          const auto wp_attributes_azimuth_offset =
              k->GetAttribute(TOKEN_WAYPOINT_AZIMUTH);

          if (wp_attributes_speed.IsValid()) {
            wp_attributes_speed.Get(&waypoint_attributes_speed_list);
          }
          if (wp_attributes_pause_duration.IsValid()) {
            wp_attributes_pause_duration.Get(
                &waypoint_attributes_pause_duration_list);
          }
          if (wp_attributes_azimuth_offset.IsValid()) {
            wp_attributes_azimuth_offset.Get(
                &waypoint_attributes_azimuth_offset_list);
          }

          std::vector<aerial::sim::mm::waypoint_params> wp_params_list;
          if (not waypoints.empty()) {
            std::vector<float3> waypoint_vec;
            waypoint_vec.reserve(waypoints.size());
            for (const auto &wp : waypoints) {
              waypoint_vec.push_back(make_float3(wp[0], wp[1], wp[2]));
            }

            if (not waypoint_attributes_speed_list.empty() &&
                not waypoint_attributes_pause_duration_list.empty() &&
                not waypoint_attributes_azimuth_offset_list.empty()) {

              for (uint32_t wp_idx = 0;
                   wp_idx < waypoint_attributes_speed_list.size(); wp_idx++) {
                wp_params_list.push_back(aerial::sim::mm::waypoint_params{
                    static_cast<float>(waypoint_attributes_speed_list[wp_idx] *
                                       cfg.scale),
                    static_cast<float>(
                        waypoint_attributes_pause_duration_list[wp_idx]),
                    static_cast<float>(
                        waypoint_attributes_azimuth_offset_list[wp_idx] *
                        M_PI_F / 180.0)});
              }
            }

            constexpr auto user_defined_waypoints =
                aerial::sim::mm::waypoint_config::user_defined;
            if ((ret = state->add_manual_user(waypoint_vec, ID,
                                              user_defined_waypoints,
                                              wp_params_list, users)) != 0) {
              return ret;
            }
          } else {
            pxr::GfVec3d translate{};
            pxr::GfVec3f rotate{};
            pxr::GfVec3f scale{};
            pxr::GfVec3f pivot{};
            pxr::UsdGeomXformCommonAPI::RotationOrder rot{};
            pxr::UsdGeomXformCommonAPI xform_api(*k);
            if (const auto r = xform_api.GetXformVectors(&translate, &rotate,
                                                         &scale, &pivot, &rot,
                                                         pxr::UsdTimeCode(0));
                !r) {
              LOG(ERROR) << "Failed to call GetXformVectors";
              return -1;
            }

            float3 point = make_float3(translate[0], translate[1],
                                       translate[2] - offset_z);
            std::vector<float3> waypoint_vec = {point};
            constexpr auto no_user_defined_waypoints =
                aerial::sim::mm::waypoint_config::none;
            if ((ret = state->add_manual_user(waypoint_vec, ID,
                                              no_user_defined_waypoints,
                                              wp_params_list, users)) != 0) {
              return ret;
            }
          }
          num_manual_users += 1;
        }
      }
    }
  }

  auto copy = cfg;
  auto node = stage->GetPrimAtPath(SdfPath("/Scenario"));
  node.GetAttribute(TOKEN_NUM_USERS).Get(&(copy.users));

  if (copy.users < num_manual_users) {
    copy.users = num_manual_users;
    node.GetAttribute(TOKEN_NUM_USERS).Set(copy.users);
  }

  state->set_mm_config(copy);
  return ret;
}

} // namespace aerial::sim::cnt
