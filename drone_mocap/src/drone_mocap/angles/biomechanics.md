# Biomechanical Definitions and Angle Computation
This document defines the biomechanical conventions used to compute lower-limb joint angles from 2D markerless pose data. 
The system estimates where the hip, knee, and ankle joint angles are for both left and right limbs in a sagittal plane projection.

## Coordinate System and Assumptions
- Motion occurs primarily in the sagittal plane
- The camera view is approximately perpendicular to the plane of motion
- The image's vertical axis is used as the global reference
- Joint centers are approximated by pose keypoints
- No correction for soft tissue artifact is applied

## 1. Knee Flexion Angle
The knee flexion angle is defined as the angle between the thigh and shank segments:
- Thigh: Hip → Knee
- Shank: Ankle → Knee

### Mathematical Formula 
θ_knee = arccos( (v_thigh · v_shank) / (|v_thigh||v_shank|) )

## 2. Hip Flexion Angle
Hip flexion is defined as the angle between the thigh segment and the global vertical axis. This definition is used due to the absence of trunk landmarks in the pose data.
Segment vector:
- Thigh: Knee → Hip
Reference Axis
- Global vertical axis of the image frame:
    v_vertical = [0, 1]
This vector corresponds to the image's vertical axis.

### Mathematical Formula
θ_hip = arccos( (v_thigh · v_vertical) / (|v_thigh||v_vertical|) )

## 3. Ankle Angle 
The ankle angle is defined as the angle between the shank segment and the global vertical axis.
Segment Vector:
- Shank: Knee → Ankle

### Mathematical Formula
θ_ankle = arccos( (v_shank · v_vertical) / (|v_shank||v_vertical|) )

Note: The true ankle dorsiflexion and plantarflexion(Dorsiflexion is the upward motion decreasing the angle, plantar flexion is pointing the toes downward) cannot be computed without foot landmarks. 
This metric represents the shank's inclination relative to vertical.

## Angle Sign Convention
- Knee flexion decreases from ~180° with increasing flexion
- Hip and ankle angles are positive when the distal segment rotates anteriorly(towards the front)
- All angles are reported in degrees

## Neutral Reference Definition
- Full knee extension (~180°) is considered the neutral knee position.
- Hip neutral corresponds to the thigh aligned with the global vertical axis.
- Ankle neutral corresponds to the shank aligned with the global vertical axis.

## Possible Results and Ranges
- Knee flexion during normal gait: ~0° to 60°
- Knee flexion during sit-to-stand: up to ~100°
- Hip flexion during gait: ~10° to 40°
- Ankle: Shank inclination during stance: ~5° to 20°
    - This angle could be 0 if standing still.
    - Values outside these ranges may indicate pose estimation error or non-sagittal motion.

## Left and Right Limb Consistency
- It is assumed that identical geometric definitions are used for left and right limbs
- No coordinate mirroring is applied
- Comparisons assume consistent camera orientation across trials

## Data Quality Considerations
- Pose estimation jitter may introduce noise in angle trajectories
- Smoothing will be applied in the system and in post-processing
- Frames with missing or unreliable keypoints should be excluded

## Limitations
- Analysis is restricted to sagittal-plane motion
- Out-of-plane motion cannot be captured with 2D pose data
- Accuracy depends on pose estimation performance
- We would need more than one camera, a point of view

## Future Improvements
- Extension to 3D pose estimation
- Inclusion of foot landmarks
- Validation against motion capture systems
