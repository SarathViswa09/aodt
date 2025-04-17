#include "json_loader_for_user.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace aerial {
namespace sim {
namespace utils {

JsonUserLoader::JsonUserLoader(const std::string &config_path)
    : config_path_(config_path) {}

bool JsonUserLoader::loadMobilityConfig(MobilityParams &config) {
  try {
    std::ifstream ifs(config_path_);
    if (!ifs.is_open()) {
      error_ = "Could not open config file at " + config_path_;
      return false;
    }

    nlohmann::json config_data;
    ifs >> config_data;

    // Extract mobility parameters
    const auto &mobility = config_data["mobility_config"];

    config.duration = mobility["duration"].get<float>();
    config.interval = mobility["interval"].get<float>();
    config.num_users = mobility["num_users"].get<unsigned int>();
    config.percentage_indoor_users =
        mobility["percentage_indoor_users"].get<float>();
    config.batches = mobility["batches"].get<unsigned int>();
    config.slots_per_batch = mobility["slots_per_batch"].get<unsigned int>();
    config.samples_per_slot = mobility["samples_per_slot"].get<unsigned int>();
    config.min_speed = mobility["min_speed"].get<float>();
    config.max_speed = mobility["max_speed"].get<float>();
    config.is_seeded = mobility["is_seeded"].get<bool>();
    config.seed = mobility["seed"].get<unsigned int>();
    config.height_ue = mobility["height_ue"].get<float>();
    config.panel_gnb = mobility["panel_gnb"].get<std::string>();
    config.panel_ue = mobility["panel_ue"].get<std::string>();
    config.scale = mobility["scale"].get<float>();

    return true;
  } catch (const std::exception &e) {
    error_ = std::string("Error processing mobility config: ") + e.what();
    return false;
  }
}

bool JsonUserLoader::loadUsers() {
  try {
    std::ifstream ifs(config_path_);
    if (!ifs.is_open()) {
      error_ = "Could not open config file at " + config_path_;
      return false;
    }

    nlohmann::json user_data;
    ifs >> user_data;

    for (const auto &user_json : user_data["users"]) {
      UserConfig user;
      user.id = user_json["id"].get<unsigned int>();
      user.waypoint_config = user_json["waypoint_config"].get<std::string>();

      if (user_json.find("waypoints") != user_json.end()) {
        for (const auto &wp_json : user_json["waypoints"]) {
          WaypointData wp;
          wp.x = wp_json["point"]["x"].get<float>();
          wp.y = wp_json["point"]["y"].get<float>();
          wp.z = wp_json["point"]["z"].get<float>();
          wp.speed_mps = wp_json["speed_mps"].get<float>();
          wp.stop_sec = wp_json["stop_sec"].get<float>();
          wp.azimuth_offset_rad = wp_json["azimuth_offset_rad"].get<float>();
          user.waypoints.push_back(wp);
        }
        user.has_position = false;
      } else if (user_json.find("position") != user_json.end()) {
        user.has_position = true;
        user.pos_x = user_json["position"]["x"].get<float>();
        user.pos_y = user_json["position"]["y"].get<float>();
        user.pos_z = user_json["position"]["z"].get<float>();
      }

      users_.push_back(user);
    }
    return true;
  } catch (const std::exception &e) {
    error_ = std::string("Error processing user config: ") + e.what();
    return false;
  }
}

const std::vector<UserConfig> &JsonUserLoader::getUsers() const {
  return users_;
}

std::string JsonUserLoader::getError() const { return error_; }

} // namespace utils
} // namespace sim
} // namespace aerial
