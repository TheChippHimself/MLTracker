"""Quality validation for detected poses."""

from __future__ import annotations

import math

from posesense.config import QualityConfig
from posesense.pose.landmarks import LM
from posesense.types import PoseResult, QualityReport


class QualityGate:
    """Validates pose results against quality thresholds."""

    def __init__(self, config: QualityConfig) -> None:
        """Initialize the quality gate.

        Args:
            config: Quality thresholds for visibility, margins, etc.
        """
        self.config = config

    def evaluate(self, result: PoseResult) -> QualityReport:
        """Evaluate a pose result and return a detailed report.

        Args:
            result: The detected pose landmarks.

        Returns:
            A QualityReport indicating pass/fail and reasons if failed.
        """
        reasons: list[str] = []
        missing: list[int] = []

        # 1. Check required landmarks
        for idx in self.config.required_landmarks:
            lm = result.landmarks[idx]
            if lm.visibility < self.config.min_visibility or lm.presence < self.config.min_presence:
                missing.append(idx)
                reasons.append(f"Landmark {LM(idx).name} not visible")

        # 2. Check edge margins
        margin = self.config.edge_margin
        for i, lm in enumerate(result.landmarks):
            # Only care about visible landmarks
            if lm.visibility > 0.1 and lm.presence > 0.1:
                if lm.x < margin or lm.x > (1.0 - margin) or lm.y < margin or lm.y > (1.0 - margin):
                    reasons.append(f"Landmark {LM(i).name} too close to edge")
                    # Don't flood the reasons list
                    if len(reasons) > 3:
                        break

        # 3. Check torso length (prevent tiny/distant false positives)
        l_shoulder = result.landmarks[LM.LEFT_SHOULDER]
        r_shoulder = result.landmarks[LM.RIGHT_SHOULDER]
        l_hip = result.landmarks[LM.LEFT_HIP]
        r_hip = result.landmarks[LM.RIGHT_HIP]

        # Ensure all torso landmarks are visible before computing
        torso_visible = all(
            lm.visibility >= self.config.min_visibility and lm.presence >= self.config.min_presence
            for lm in (l_shoulder, r_shoulder, l_hip, r_hip)
        )

        if torso_visible:
            # Midpoints
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
            mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            mid_hip_x = (l_hip.x + r_hip.x) / 2
            mid_hip_y = (l_hip.y + r_hip.y) / 2

            torso_length = math.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)

            if torso_length < self.config.min_torso_length_norm:
                reasons.append(f"Person too far (torso length {torso_length:.2f} < {self.config.min_torso_length_norm})")
        else:
            reasons.append("Torso not fully visible")

        # Basic score calculation (1.0 - fraction of issues)
        score = 1.0
        if reasons:
            score -= min(0.2 * len(reasons), 0.9)  # Cap at 0.1 for failed

        return QualityReport(
            ok=len(reasons) == 0,
            score=max(0.0, score),
            missing=tuple(missing),
            reasons=tuple(reasons),
        )
