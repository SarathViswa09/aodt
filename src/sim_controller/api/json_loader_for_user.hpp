#pragma once

#include <vector>
#include <string>

namespace aerial {
namespace sim {
namespace utils {

struct WaypointData {
  float x;
  float y;
  float z;
  float speed_mps;
  float stop_sec;
  float azimuth_offset_rad;
};

struct UserConfig {
  unsigned int id;
  std::string waypoint_config;
  std::vector<WaypointData> waypoints;
  bool has_position;
  float pos_x, pos_y, pos_z;
};

// Mobility Params
struct MobilityParams {
  float duration;
  float interval;
  unsigned int num_users;
  float percentage_indoor_users;
  unsigned int batches;
  unsigned int slots_per_batch;
  unsigned int samples_per_slot;
  float min_speed;
  float max_speed;
  bool is_seeded;
  unsigned int seed;
  float height_ue;
  std::string panel_gnb;
  std::string panel_ue;
  float scale;
};

class JsonUserLoader {
public:
  JsonUserLoader(const std::string &config_path);

  //users
  bool loadUsers();
  // Mobility
  bool loadMobilityConfig(MobilityParams &config);
  const std::vector<UserConfig> &getUsers() const;
  std::string getError() const;

private:
  std::string config_path_;
  std::vector<UserConfig> users_;
  std::string error_;
};

} // namespace utils
} // namespace sim
} // namespace aerial
