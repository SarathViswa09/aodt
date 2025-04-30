#include "ConfigLoader.hpp"
#include <fstream>

using json = nlohmann::json;

StageConfig& StageConfig::instance() {
    static StageConfig config;
    static bool loaded = false;
    if (!loaded) {
        config.loadFromFile();
        loaded = true;
    }
    return config;
}

void StageConfig::loadFromFile(const std::string& filename) {
    std::ifstream f(filename);
    json data = json::parse(f);

    dus.clear();
    for (const auto& du : data["DUs"]) {
        dus.push_back(DUConfig{
            du.at("id"),
            du.at("subcarrier_spacing"),
            du.at("fft_size"),
            du.at("num_antennas"),
            du.at("max_channel_bandwidth"),
            pxr::GfVec3d(
                du["position"]["x"],
                du["position"]["y"],
                du["position"]["z"])
        });
    }

    rus.clear();
    for (const auto& ru : data["RUs"]) {
        rus.push_back(RUConfig{
            ru.at("cell_id"),
            ru.at("du_id"),
            ru.at("du_manual_assign"),
            ru.at("radiated_power"),
            ru.at("height"),
            ru.at("mech_azimuth"),
            ru.at("mech_tilt"),
            ru.at("panel_type"),
            pxr::GfVec3d(
                ru["position"]["x"],
                ru["position"]["y"],
                ru["position"]["z"])
        });
    }

    ues.clear();
    for (const auto& ue_json : data["UEs"]) {
        UEConfig u{
            ue_json.at("user_id"),
            ue_json.at("manual"),
            ue_json.at("radiated_power"),
            ue_json.at("mech_tilt"),
            ue_json.at("panel_type"),
            ue_json.at("bler_target"),
            pxr::GfVec3d(
                ue_json["position"]["x"],
                ue_json["position"]["y"],
                ue_json["position"]["z"]),
            ue_json.at("waypoint_config"),
            ue_json.at("is_indoor"),
            {}
        };

        if (ue_json.contains("waypoints")) {
            for (const auto& wpt : ue_json["waypoints"]) {
                u.waypoints.push_back(WaypointConfig{
                    pxr::GfVec3d(
                        wpt["point"]["x"],
                        wpt["point"]["y"],
                        wpt["point"]["z"]),
                    wpt["speed_mps"],
                    wpt["stop_sec"],
                    wpt["azimuth_offset_rad"]
                });
            }
        }

        ues.push_back(std::move(u));
    }
}

