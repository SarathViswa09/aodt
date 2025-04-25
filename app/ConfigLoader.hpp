#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <pxr/base/gf/vec3d.h>

struct WaypointConfig {
    pxr::GfVec3d point;
    double speed_mps;
    double stop_sec;
    double azimuth_offset_rad;
};

struct DUConfig {
    uint32_t id;
    double subcarrier_spacing;
    uint32_t fft_size;
    uint32_t num_antennas;
    double max_channel_bandwidth;
    pxr::GfVec3d position;
};

struct RUConfig {
    uint32_t cell_id;
    uint32_t du_id;
    bool du_manual_assign;
    double radiated_power;
    double height;
    double mech_azimuth;
    double mech_tilt;
    std::string panel_type;
    pxr::GfVec3d position;
};

struct UEConfig {
    uint32_t user_id;
    bool manual;
    double radiated_power;
    double mech_tilt;
    std::string panel_type;
    double bler_target;
    pxr::GfVec3d position;
    std::string waypoint_config;
    bool is_indoor;
    std::vector<WaypointConfig> waypoints;
};

struct StageConfig {
    std::vector<DUConfig> dus;
    std::vector<RUConfig> rus;
    std::vector<UEConfig> ues;

    static StageConfig& instance();
    void loadFromFile(const std::string& filename = "app/config.json");
};
