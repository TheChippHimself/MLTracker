"""Relational features for poses (distances and offsets between keypoints)."""

import math
from posesense.pose.landmarks import LM
from posesense.types import Landmark


def extract_relations(lms: list[Landmark], scale: float) -> dict[str, float]:
    """Extract relation features normalized by the given scale.
    
    Assumes lms has already undergone aspect-ratio correction, so x and y are in the same units.
    """
    
    def dist(a: Landmark, b: Landmark) -> float:
        return math.hypot(a.x - b.x, a.y - b.y) / scale
        
    def dy(a: Landmark, b: Landmark) -> float:
        return (a.y - b.y) / scale
        
    def dx(a: Landmark, b: Landmark) -> float:
        return (a.x - b.x) / scale
        
    mid_hip_x = (lms[LM.LEFT_HIP].x + lms[LM.RIGHT_HIP].x) / 2.0
    
    # 1. Vertical offset of each wrist relative to its shoulder
    wrist_y_rel_shoulder_l = dy(lms[LM.LEFT_WRIST], lms[LM.LEFT_SHOULDER])
    wrist_y_rel_shoulder_r = dy(lms[LM.RIGHT_WRIST], lms[LM.RIGHT_SHOULDER])
    
    # 2. Vertical offset of each wrist relative to the nose
    wrist_y_rel_nose_l = dy(lms[LM.LEFT_WRIST], lms[LM.NOSE])
    wrist_y_rel_nose_r = dy(lms[LM.RIGHT_WRIST], lms[LM.NOSE])
    
    # 3. Horizontal distance of each wrist from body midline (mid_hip_x)
    wrist_x_midline_l = abs(lms[LM.LEFT_WRIST].x - mid_hip_x) / scale
    wrist_x_midline_r = abs(lms[LM.RIGHT_WRIST].x - mid_hip_x) / scale
    
    # 4. Distance from each wrist to opposite shoulder
    wrist_to_opp_shoulder_l = dist(lms[LM.LEFT_WRIST], lms[LM.RIGHT_SHOULDER])
    wrist_to_opp_shoulder_r = dist(lms[LM.RIGHT_WRIST], lms[LM.LEFT_SHOULDER])
    
    # 5. Distance from each wrist to same-side hip
    wrist_to_hip_l = dist(lms[LM.LEFT_WRIST], lms[LM.LEFT_HIP])
    wrist_to_hip_r = dist(lms[LM.RIGHT_WRIST], lms[LM.RIGHT_HIP])
    
    # 6. Wrist-to-wrist distance
    wrist_to_wrist = dist(lms[LM.LEFT_WRIST], lms[LM.RIGHT_WRIST])
    
    # 7. Vertical offset between hip and knee on each side
    hip_knee_dy_l = dy(lms[LM.LEFT_KNEE], lms[LM.LEFT_HIP])
    hip_knee_dy_r = dy(lms[LM.RIGHT_KNEE], lms[LM.RIGHT_HIP])
    
    # 8. Vertical offset between knee and ankle on each side
    knee_ankle_dy_l = dy(lms[LM.LEFT_ANKLE], lms[LM.LEFT_KNEE])
    knee_ankle_dy_r = dy(lms[LM.RIGHT_ANKLE], lms[LM.RIGHT_KNEE])
    
    # 9. Shoulder-line tilt angle (from horizontal, mapped to vertical offset)
    shoulder_tilt = dy(lms[LM.RIGHT_SHOULDER], lms[LM.LEFT_SHOULDER])
    
    # 10. Torso inclination angle (mid-shoulder to mid-hip from vertical)
    mid_shoulder_x = (lms[LM.LEFT_SHOULDER].x + lms[LM.RIGHT_SHOULDER].x) / 2.0
    mid_shoulder_y = (lms[LM.LEFT_SHOULDER].y + lms[LM.RIGHT_SHOULDER].y) / 2.0
    mid_hip_y = (lms[LM.LEFT_HIP].y + lms[LM.RIGHT_HIP].y) / 2.0
    
    dx_torso = (mid_shoulder_x - mid_hip_x) / scale
    dy_torso = (mid_hip_y - mid_shoulder_y) / scale
    torso_inclination = math.atan2(dx_torso, dy_torso) * 180 / math.pi if dy_torso != 0 else 0.0
    
    # 11. Left/right arm-raised indicators (continuous)
    # Using relative height of elbow and wrist above shoulder
    arm_raised_l = - (dy(lms[LM.LEFT_WRIST], lms[LM.LEFT_SHOULDER]) + dy(lms[LM.LEFT_ELBOW], lms[LM.LEFT_SHOULDER]))
    arm_raised_r = - (dy(lms[LM.RIGHT_WRIST], lms[LM.RIGHT_SHOULDER]) + dy(lms[LM.RIGHT_ELBOW], lms[LM.RIGHT_SHOULDER]))

    return {
        "wrist_y_rel_shoulder_l": wrist_y_rel_shoulder_l,
        "wrist_y_rel_shoulder_r": wrist_y_rel_shoulder_r,
        "wrist_y_rel_nose_l": wrist_y_rel_nose_l,
        "wrist_y_rel_nose_r": wrist_y_rel_nose_r,
        "wrist_x_midline_l": wrist_x_midline_l,
        "wrist_x_midline_r": wrist_x_midline_r,
        "wrist_to_opp_shoulder_l": wrist_to_opp_shoulder_l,
        "wrist_to_opp_shoulder_r": wrist_to_opp_shoulder_r,
        "wrist_to_hip_l": wrist_to_hip_l,
        "wrist_to_hip_r": wrist_to_hip_r,
        "wrist_to_wrist": wrist_to_wrist,
        "hip_knee_dy_l": hip_knee_dy_l,
        "hip_knee_dy_r": hip_knee_dy_r,
        "knee_ankle_dy_l": knee_ankle_dy_l,
        "knee_ankle_dy_r": knee_ankle_dy_r,
        "shoulder_tilt": shoulder_tilt,
        "torso_inclination": torso_inclination,
        "arm_raised_l": arm_raised_l,
        "arm_raised_r": arm_raised_r,
    }
