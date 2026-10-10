"""Feature extraction pipeline."""

from posesense.config import FeaturesConfig
from posesense.exceptions import FeatureSchemaError
from posesense.pose.landmarks import LM
from posesense.types import PoseResult, Landmark
from posesense.features.normalize import aspect_ratio_correction, get_center, get_scale
from posesense.features.angles import calculate_angle
from posesense.features.relational import extract_relations

class FeatureExtractor:
    """Extracts a normalized 1D feature vector from a pose result."""

    # Group A target landmarks
    GROUP_A_LMS = [
        LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
        LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_HIP, LM.RIGHT_HIP,
        LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_ANKLE, LM.RIGHT_ANKLE
    ]

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config
        
        from posesense.features.schema import SCHEMA_VERSION
        if self.config.schema_version != SCHEMA_VERSION:
            raise FeatureSchemaError(
                f"Config requested schema {self.config.schema_version}, "
                f"but code only supports {SCHEMA_VERSION}"
            )

    def extract(self, result: PoseResult) -> dict[str, float]:
        features: dict[str, float] = {}

        lms = list(result.world_landmarks if self.config.coordinate_source == "world" and result.world_landmarks else result.landmarks)

        # Aspect ratio correction for image landmarks
        if self.config.coordinate_source == "image":
            lms = aspect_ratio_correction(lms, result.frame_width, result.frame_height)

        center_x, center_y = get_center(lms, self.config)
        scale = get_scale(lms, self.config)

        # 1. Group A: Coordinate Features (Only target 13 landmarks)
        for lm_enum in self.GROUP_A_LMS:
            lm = lms[lm_enum]
            name = lm_enum.name.lower()
            features[f"{name}_x"] = (lm.x - center_x) / scale
            features[f"{name}_y"] = (lm.y - center_y) / scale
            if self.config.use_z:
                features[f"{name}_z"] = lm.z / scale
            # We don't output visibilities in the final feature vector per spec Group A

        # 2. Group B: Angle Features
        angles = self._extract_angles(lms)
        features.update(angles)

        # 3. Group C: Relational Features
        relations = extract_relations(lms, scale)
        features.update(relations)

        return features

    def _extract_angles(self, lms: list[Landmark]) -> dict[str, float]:
        return {
            "angle_left_elbow": calculate_angle(lms[LM.LEFT_SHOULDER], lms[LM.LEFT_ELBOW], lms[LM.LEFT_WRIST]),
            "angle_right_elbow": calculate_angle(lms[LM.RIGHT_SHOULDER], lms[LM.RIGHT_ELBOW], lms[LM.RIGHT_WRIST]),
            "angle_left_shoulder": calculate_angle(lms[LM.LEFT_HIP], lms[LM.LEFT_SHOULDER], lms[LM.LEFT_ELBOW]),
            "angle_right_shoulder": calculate_angle(lms[LM.RIGHT_HIP], lms[LM.RIGHT_SHOULDER], lms[LM.RIGHT_ELBOW]),
            "angle_left_hip": calculate_angle(lms[LM.LEFT_SHOULDER], lms[LM.LEFT_HIP], lms[LM.LEFT_KNEE]),
            "angle_right_hip": calculate_angle(lms[LM.RIGHT_SHOULDER], lms[LM.RIGHT_HIP], lms[LM.RIGHT_KNEE]),
            "angle_left_knee": calculate_angle(lms[LM.LEFT_HIP], lms[LM.LEFT_KNEE], lms[LM.LEFT_ANKLE]),
            "angle_right_knee": calculate_angle(lms[LM.RIGHT_HIP], lms[LM.RIGHT_KNEE], lms[LM.RIGHT_ANKLE]),
        }

    def get_feature_names(self) -> list[str]:
        dummy_lms = tuple(Landmark(0.5, 0.5, 0.0, 1.0, 1.0) for _ in range(33))
        dummy_result = PoseResult(landmarks=dummy_lms, world_landmarks=None, timestamp_ms=0, frame_width=100, frame_height=100)
        return list(self.extract(dummy_result).keys())
