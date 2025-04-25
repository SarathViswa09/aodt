#include "stage_utils.hpp"
#include "ConfigLoader.hpp"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usd/editTarget.h>
#include <pxr/usd/sdf/layer.h>
#include <iostream>
#include <sstream>
#include <iomanip>

PXR_NAMESPACE_USING_DIRECTIVE

void addSimpleDUAndRU(UsdStageRefPtr stage) {
    using SC = StageConfig;
    auto& cfg = SC::instance();

    // --- DUs ---
    const SdfPath dusPath("/DUs");
    if (!stage->GetPrimAtPath(dusPath))
        stage->DefinePrim(dusPath, TfToken("Xform"));

    for (const auto& d : cfg.dus) {
        std::ostringstream name;
        name << "du_" << std::setw(4) << std::setfill('0') << d.id;
        SdfPath duPath = dusPath.AppendChild(TfToken(name.str()));
        UsdPrim duPrim = stage->DefinePrim(duPath, TfToken("Xform"));

        duPrim.CreateAttribute(TfToken("aerial:du:id"), SdfValueTypeNames->UInt).Set(d.id);
        duPrim.CreateAttribute(TfToken("aerial:du:subcarrier_spacing"), SdfValueTypeNames->Double).Set(d.subcarrier_spacing);
        duPrim.CreateAttribute(TfToken("aerial:du:fft_size"), SdfValueTypeNames->UInt).Set(d.fft_size);
        duPrim.CreateAttribute(TfToken("aerial:du:num_antennas"), SdfValueTypeNames->UInt).Set(d.num_antennas);
        duPrim.CreateAttribute(TfToken("aerial:du:max_channel_bandwidth"), SdfValueTypeNames->Double).Set(d.max_channel_bandwidth);

        UsdGeomXformCommonAPI(duPrim).SetTranslate(d.position);
    }

    // --- RUs ---
    const SdfPath rusPath("/RUs");
    if (!stage->GetPrimAtPath(rusPath))
        stage->DefinePrim(rusPath, TfToken("Xform"));

    for (const auto& r : cfg.rus) {
        std::ostringstream name;
        name << "ru_" << std::setw(4) << std::setfill('0') << r.cell_id;
        SdfPath ruPath = rusPath.AppendChild(TfToken(name.str()));
        UsdPrim ruPrim = stage->DefinePrim(ruPath, TfToken("Xform"));

        ruPrim.CreateAttribute(TfToken("aerial:gnb:cell_id"), SdfValueTypeNames->UInt).Set(r.cell_id);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:du_id"), SdfValueTypeNames->UInt).Set(r.du_id);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:du_manual_assign"), SdfValueTypeNames->Bool).Set(r.du_manual_assign);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:radiated_power"), SdfValueTypeNames->Double).Set(r.radiated_power);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:height"), SdfValueTypeNames->Double).Set(r.height);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:mech_azimuth"), SdfValueTypeNames->Double).Set(r.mech_azimuth);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:mech_tilt"), SdfValueTypeNames->Double).Set(r.mech_tilt);
        ruPrim.CreateAttribute(TfToken("aerial:gnb:panel_type"), SdfValueTypeNames->Token).Set(TfToken(r.panel_type));

        UsdGeomXformCommonAPI(ruPrim).SetTranslate(r.position);
    }

    // --- UEs ---
    const SdfPath uesPath("/UEs");

    if (!stage->GetPrimAtPath(uesPath)) {
        stage->DefinePrim(uesPath, TfToken("Xform"));
    } 

    else {
    // Remove all existing UEs properly
    auto uesPrim = stage->GetPrimAtPath(uesPath);

    for (const auto& child : uesPrim.GetChildren()) {
        stage->RemovePrim(child.GetPath());
    }
}
    for (const auto& u : cfg.ues) {
        std::ostringstream name;
        name << "ue_" << std::setw(4) << std::setfill('0') << u.user_id;
        SdfPath uePath = uesPath.AppendChild(TfToken(name.str()));
        UsdPrim uePrim = stage->DefinePrim(uePath, TfToken("Xform"));

        uePrim.CreateAttribute(TfToken("aerial:ue:user_id"), SdfValueTypeNames->UInt).Set(u.user_id);
        uePrim.CreateAttribute(TfToken("aerial:ue:manual"), SdfValueTypeNames->Bool).Set(u.manual);
        uePrim.CreateAttribute(TfToken("aerial:ue:radiated_power"), SdfValueTypeNames->Double).Set(u.radiated_power);
        uePrim.CreateAttribute(TfToken("aerial:ue:mech_tilt"), SdfValueTypeNames->Double).Set(u.mech_tilt);
        uePrim.CreateAttribute(TfToken("aerial:ue:panel_type"), SdfValueTypeNames->Token).Set(TfToken(u.panel_type));
        uePrim.CreateAttribute(TfToken("aerial:ue:bler_target"), SdfValueTypeNames->Double).Set(u.bler_target);

        UsdGeomXformCommonAPI(uePrim).SetTranslate(u.position);

        if (u.waypoint_config == "user_defined") {
            pxr::VtArray<pxr::GfVec3f> points;
            pxr::VtArray<float> speeds, pauses, azimuths;
            for (auto& w : u.waypoints) {
                points.push_back(pxr::GfVec3f(w.point[0], w.point[1], w.point[2]));
                speeds.push_back(static_cast<float>(w.speed_mps));
                pauses.push_back(static_cast<float>(w.stop_sec));
                azimuths.push_back(static_cast<float>(w.azimuth_offset_rad));
            }

            uePrim.CreateAttribute(TfToken("aerial:ue:waypoints"), SdfValueTypeNames->Point3fArray).Set(points);
            uePrim.CreateAttribute(TfToken("aerial:ue:waypoint_speed"), SdfValueTypeNames->FloatArray).Set(speeds);
            uePrim.CreateAttribute(TfToken("aerial:ue:waypoint_pause_duration"), SdfValueTypeNames->FloatArray).Set(pauses);
            uePrim.CreateAttribute(TfToken("aerial:ue:waypoint_azimuth_offset"), SdfValueTypeNames->FloatArray).Set(azimuths);
        }
    }
}
