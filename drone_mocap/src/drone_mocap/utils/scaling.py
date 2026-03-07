from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PixelScaler:
    """
    Converts pixel measurements to real-world units (meters).

    Calibration is done by supplying a known real-world distance and its
    pixel length in the video frame (e.g. athlete height, lane width,
    a calibration pole placed in the scene).

    Usage
    -----
    # From athlete height:
    scaler = PixelScaler.from_known_distance(pixel_dist=820.0, real_dist_m=1.78)

    # Convert a position offset:
    delta_m = scaler.px_to_m(delta_px)

    # Compute joint velocities in m/s:
    vels = scaler.joint_velocities_m_per_s(xy_smooth, fps, joint_indices)
    """

    px_per_meter: float

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_known_distance(cls, pixel_dist: float, real_dist_m: float) -> "PixelScaler":
        """
        Create a scaler from a single known reference distance.

        Args:
            pixel_dist:  Length of the reference object in pixels.
                         Measure in the first frame using, e.g., the vertical
                         distance from the top of the head keypoint to the heel
                         keypoint when the athlete is standing upright.
            real_dist_m: True length of the same object in metres
                         (e.g. 1.78 for an athlete who is 178 cm tall).
        """
        if pixel_dist <= 0 or real_dist_m <= 0:
            raise ValueError("pixel_dist and real_dist_m must both be positive.")
        return cls(px_per_meter=pixel_dist / real_dist_m)

    @classmethod
    def from_keypoint_height(
        cls,
        xy: np.ndarray,
        head_idx: int,
        heel_idx: int,
        real_height_m: float,
    ) -> "PixelScaler":
        """
        Derive scale directly from keypoint positions.

        Args:
            xy:            (J, 2) pixel array for a reference frame where the
                           athlete is standing upright and fully visible.
            head_idx:      Keypoint index for the top of the head
                           (MediaPipe: 0 = nose; use 11/12 shoulder as fallback).
            heel_idx:      Keypoint index for the heel (29 = L_HEEL, 30 = R_HEEL).
            real_height_m: Athlete's standing height in metres.
        """
        pixel_height = float(np.linalg.norm(xy[heel_idx] - xy[head_idx]))
        return cls.from_known_distance(pixel_height, real_height_m)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def px_to_m(self, px_value: float | np.ndarray) -> float | np.ndarray:
        """Convert a pixel distance / displacement to metres."""
        return px_value / self.px_per_meter

    def px_per_s_to_m_per_s(self, px_per_s: float | np.ndarray) -> float | np.ndarray:
        """Convert a pixel-per-second speed to metres per second."""
        return px_per_s / self.px_per_meter

    # ------------------------------------------------------------------
    # Segment kinematics
    # ------------------------------------------------------------------

    def joint_velocities_m_per_s(
        self,
        xy_smooth: np.ndarray,
        fps: float,
        joint_indices: dict[str, int],
    ) -> dict[str, np.ndarray]:
        """
        Compute the speed (scalar, m/s) of each named joint over time.

        Velocity is estimated via central finite differences on the smoothed
        pixel trajectory, then converted to m/s.  Edge frames use forward /
        backward differences automatically via numpy.gradient.

        Args:
            xy_smooth:     (T, J, 2) smoothed pixel coordinates.
            fps:           Video frame rate in Hz.
            joint_indices: Mapping of human-readable name → keypoint index,
                           e.g. {"knee": 26, "ankle": 28}.

        Returns:
            dict of name → (T,) speed arrays in m/s.
        """
        velocities: dict[str, np.ndarray] = {}
        for name, j in joint_indices.items():
            xy_j = xy_smooth[:, j, :].astype(float)  # (T, 2)
            # Central difference: px per frame
            dxy = np.gradient(xy_j, axis=0)           # (T, 2)
            speed_px_per_frame = np.linalg.norm(dxy, axis=1)  # (T,)
            velocities[name] = speed_px_per_frame * fps / self.px_per_meter
        return velocities

    def segment_length_m(
        self,
        xy_smooth: np.ndarray,
        idx_prox: int,
        idx_dist: int,
    ) -> np.ndarray:
        """
        Compute the instantaneous length of a body segment in metres over time.

        Args:
            xy_smooth: (T, J, 2) smoothed pixel coordinates.
            idx_prox:  Proximal keypoint index (e.g. hip).
            idx_dist:  Distal keypoint index (e.g. knee).

        Returns:
            (T,) array of segment lengths in metres.
        """
        diff = xy_smooth[:, idx_prox, :] - xy_smooth[:, idx_dist, :]
        px_lengths = np.linalg.norm(diff, axis=1)
        return px_lengths / self.px_per_meter
