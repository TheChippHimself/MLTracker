import math
import socket
import struct

import numpy as np

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
# MediaPipe under-reports depth. >1.0 exaggerates arms reaching toward/away from
# the camera. Keep at 1.0 unless forward reaches look too flat; the old value of
# 2.0 was compensating for the q_between over-rotation bug and distorts the arms.
ARM_Z_SCALE = 1.0

# Send /VMC/Ext/Tra/Pos hand/head/pelvis trackers in addition to the bones.
# Set to False to test pure FK (bones only).
SEND_TRACKERS = True

# How the Hips *position* is interpreted by the receiver.
#   "offset"   : send only how far the hips moved from standing (0 when standing).
#   "absolute" : send the full hip height in metres.
# If your avatar floats a full leg-length above the floor, use "offset".
# If it sinks into the floor, use "absolute".
HIPS_POSITION_MODE = "offset"
HIPS_Y_SCALE = 1.0  # scale crouch depth to your avatar's size if needed

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------
# Quaternion helpers, (x, y, z, w) order, same convention as Unity
# ----------------------------------------------------------------------------
def q_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def q_between(u, v):
    """Shortest-arc quaternion rotating direction u onto direction v."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm < 1e-6 or v_norm < 1e-6:
        return IDENTITY.copy()
    u = u / u_norm
    v = v / v_norm
    d = float(np.dot(u, v))
    if d > 1.0 - 1e-9:
        return IDENTITY.copy()
    if d < -1.0 + 1e-9:
        # 180 degrees: any axis perpendicular to u works
        axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis) < 0.01:
            axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    c = np.cross(u, v)
    # u and v are unit vectors here, so w = 1 + dot(u, v).
    # (The old code used |u|*|v| from BEFORE normalising, i.e. the bone length
    # in metres, which over-rotated every bone.)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def look_rotation(forward, up):
    """Emulates Unity's Quaternion.LookRotation(forward, up)."""
    f_norm = np.linalg.norm(forward)
    if f_norm < 1e-6:
        return IDENTITY.copy()
    f = forward / f_norm
    u = up - np.dot(up, f) * f
    if np.linalg.norm(u) < 1e-3:
        u = np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, f) * f
        if np.linalg.norm(u) < 1e-3:
            u = np.array([0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)
    r = np.cross(u, f)

    m00, m01, m02 = r[0], u[0], f[0]
    m10, m11, m12 = r[1], u[1], f[1]
    m20, m21, m22 = r[2], u[2], f[2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def _osc_str(s: str) -> bytes:
    """OSC string: UTF-8 + at least one NUL, padded to a multiple of 4 bytes."""
    b = s.encode("utf-8") + b"\0"
    return b + b"\0" * (-len(b) % 4)


class VMCSender:
    """Broadcasts body tracking data over the network using the VMC (OSC) Protocol."""

    def __init__(self, ip="127.0.0.1", port=39539):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bone_states = {}  # For smoothing: {bone_name: (qx, qy, qz, qw)}
        self.smoothing = 0.85
        self._hip_drop = None   # smoothed vertical hips->lowest-foot distance
        self._leg_ref = None    # smoothed standing leg length

    # ------------------------------------------------------------------ utils
    def _slerp(self, q1, q2, t):
        dot = np.sum(q1 * q2)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q1 + t * (q2 - q1)
            return q / np.linalg.norm(q)
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = (s0 * q1) + (s1 * q2)
        return q / np.linalg.norm(q)

    def send_bone(self, name: str, x: float, y: float, z: float,
                  qx: float, qy: float, qz: float, qw: float):
        q_target = np.array([qx, qy, qz, qw], dtype=float)

        if self.smoothing > 0.0 and name in self.bone_states:
            q_prev = self.bone_states[name]
            q_smooth = self._slerp(q_prev, q_target, 1.0 - self.smoothing)
        else:
            q_smooth = q_target

        self.bone_states[name] = q_smooth
        qx, qy, qz, qw = q_smooth

        msg = (_osc_str("/VMC/Ext/Bone/Pos") + _osc_str(",sfffffff") + _osc_str(name)
               + struct.pack('>7f', x, y, z, qx, qy, qz, qw))
        self.sock.sendto(msg, (self.ip, self.port))

    def send_tracker(self, name: str, x: float, y: float, z: float,
                     qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        msg = (_osc_str("/VMC/Ext/Tra/Pos") + _osc_str(",sfffffff") + _osc_str(name)
               + struct.pack('>7f', x, y, z, qx, qy, qz, qw))
        self.sock.sendto(msg, (self.ip, self.port))

    # ------------------------------------------------------------ main solver
    def solve_pose(self, lm):
        """
        Turn MediaPipe world landmarks into VMC data.

        Returns (bones, trackers):
          bones    = {name: (pos_xyz, quat_xyzw)}   (local rotations, Hips is root)
          trackers = {name: (pos_xyz, quat_xyzw)}
        Split out from broadcast_pose so it can be unit-tested without a socket.
        """
        def get_pos(idx):
            # MediaPipe: X right(+), Y down(+), Z toward camera(-)
            # VMC/Unity:  X right(+), Y up(+),   Z forward(+)
            return np.array([-lm[idx].x, -lm[idx].y, -lm[idx].z], dtype=float)

        # --- Landmarks ---
        nose = get_pos(0)
        ear_l, ear_r = get_pos(7), get_pos(8)
        shoulder_l, shoulder_r = get_pos(11), get_pos(12)
        elbow_l, elbow_r = get_pos(13), get_pos(14)
        wrist_l, wrist_r = get_pos(15), get_pos(16)
        hip_l, hip_r = get_pos(23), get_pos(24)
        knee_l, knee_r = get_pos(25), get_pos(26)
        ankle_l, ankle_r = get_pos(27), get_pos(28)

        pelvis = (hip_l + hip_r) / 2.0
        shoulder_mid = (shoulder_l + shoulder_r) / 2.0

        # --- Global (world) rotations of each bone, assuming a T-pose rest ---
        # Head
        head_right = ear_r - ear_l
        head_forward = nose - ((ear_l + ear_r) / 2.0)
        head_up = np.cross(head_forward, head_right)
        q_global_head = look_rotation(head_forward, head_up)

        # Hips
        body_right = hip_r - hip_l
        body_up = shoulder_mid - pelvis
        body_forward = np.cross(body_right, body_up)
        q_global_hips = look_rotation(body_forward, body_up)

        # Spine
        q_global_spine = q_between([0, 1, 0], shoulder_mid - pelvis)

        # Arms (optionally exaggerate depth)
        zs = np.array([1.0, 1.0, ARM_Z_SCALE])
        l_upper_arm = (elbow_l - shoulder_l) * zs
        l_lower_arm = (wrist_l - elbow_l) * zs
        r_upper_arm = (elbow_r - shoulder_r) * zs
        r_lower_arm = (wrist_r - elbow_r) * zs

        q_global_LUA = q_between([-1, 0, 0], l_upper_arm)
        q_global_LLA = q_between([-1, 0, 0], l_lower_arm)
        q_global_RUA = q_between([1, 0, 0], r_upper_arm)
        q_global_RLA = q_between([1, 0, 0], r_lower_arm)

        # Legs
        l_upper_leg = knee_l - hip_l
        l_lower_leg = ankle_l - knee_l
        r_upper_leg = knee_r - hip_r
        r_lower_leg = ankle_r - knee_r

        q_global_LUL = q_between([0, -1, 0], l_upper_leg)
        q_global_LLL = q_between([0, -1, 0], l_lower_leg)
        q_global_RUL = q_between([0, -1, 0], r_upper_leg)
        q_global_RLL = q_between([0, -1, 0], r_lower_leg)

        # --- Global -> local (parent^-1 * child) ---
        q_local_spine = q_mult(q_inv(q_global_hips), q_global_spine)
        q_local_head = q_mult(q_inv(q_global_spine), q_global_head)

        q_local_LUA = q_mult(q_inv(q_global_spine), q_global_LUA)
        q_local_LLA = q_mult(q_inv(q_global_LUA), q_global_LLA)
        q_local_RUA = q_mult(q_inv(q_global_spine), q_global_RUA)
        q_local_RLA = q_mult(q_inv(q_global_RUA), q_global_RLA)

        q_local_LUL = q_mult(q_inv(q_global_hips), q_global_LUL)
        q_local_LLL = q_mult(q_inv(q_global_LUL), q_global_LLL)
        q_local_RUL = q_mult(q_inv(q_global_hips), q_global_RUL)
        q_local_RLL = q_mult(q_inv(q_global_RUL), q_global_RLL)

        # --- Hip height ---
        # Real height of the hips above the lowest foot. This lets the avatar
        # actually crouch (feet stay planted) instead of folding its legs in the
        # air, and still ignores depth noise because it only uses the Y axis.
        l_leg_len = np.linalg.norm(l_upper_leg) + np.linalg.norm(l_lower_leg)
        r_leg_len = np.linalg.norm(r_upper_leg) + np.linalg.norm(r_lower_leg)
        leg_length = (l_leg_len + r_leg_len) / 2.0

        raw_drop = pelvis[1] - min(ankle_l[1], ankle_r[1])
        raw_drop = float(np.clip(raw_drop, 0.05, leg_length))

        a = 0.3  # light smoothing for height only
        self._hip_drop = raw_drop if self._hip_drop is None else self._hip_drop + a * (raw_drop - self._hip_drop)
        self._leg_ref = leg_length if self._leg_ref is None else self._leg_ref + 0.02 * (leg_length - self._leg_ref)

        hip_height = self._hip_drop  # absolute metres, hips above floor
        if HIPS_POSITION_MODE == "offset":
            hips_y = min(0.0, hip_height - self._leg_ref) * HIPS_Y_SCALE
        else:
            hips_y = hip_height * HIPS_Y_SCALE

        bones = {
            "Head": ((0, 0, 0), q_local_head),
            "Spine": ((0, 0, 0), q_local_spine),
            "LeftUpperArm": ((0, 0, 0), q_local_LUA),
            "LeftLowerArm": ((0, 0, 0), q_local_LLA),
            "RightUpperArm": ((0, 0, 0), q_local_RUA),
            "RightLowerArm": ((0, 0, 0), q_local_RLA),
            "LeftUpperLeg": ((0, 0, 0), q_local_LUL),
            "LeftLowerLeg": ((0, 0, 0), q_local_LLL),
            "RightUpperLeg": ((0, 0, 0), q_local_RUL),
            "RightLowerLeg": ((0, 0, 0), q_local_RLL),
            "Hips": ((pelvis[0], hips_y, pelvis[2]), q_global_hips),
        }

        # --- Trackers (absolute positions, hips at hip_height) ---
        # Hand targets follow the same (optionally z-scaled) arm chain as the FK bones.
        hand_l = shoulder_l + (wrist_l - shoulder_l) * zs
        hand_r = shoulder_r + (wrist_r - shoulder_r) * zs
        trackers = {
            "Head": ((nose[0], nose[1] + hip_height, nose[2]), q_global_head),
            "LeftHand": ((hand_l[0], hand_l[1] + hip_height, hand_l[2]), IDENTITY),
            "RightHand": ((hand_r[0], hand_r[1] + hip_height, hand_r[2]), IDENTITY),
            "Pelvis": ((pelvis[0], hip_height, pelvis[2]), q_global_hips),
        }
        return bones, trackers

    def broadcast_pose(self, pose_result):
        # world_landmarks keep true 3D proportions (metres, origin at the hips).
        if not pose_result.world_landmarks:
            return

        bones, trackers = self.solve_pose(pose_result.world_landmarks)

        for name, (pos, q) in bones.items():
            self.send_bone(name, pos[0], pos[1], pos[2], *q)

        if SEND_TRACKERS:
            for name, (pos, q) in trackers.items():
                self.send_tracker(name, pos[0], pos[1], pos[2], *q)