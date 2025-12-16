# Visual Odometry Improvements Guide

This document outlines methods to reduce trajectory drift in our visual odometry pipeline, ordered from **easiest to hardest** to implement.

---

## Current Pipeline Limitations

Our pipeline chains frame-to-frame transforms:

```
T_total = ΔT₀₁ × ΔT₁₂ × ΔT₂₃ × ...
```

**Problem:** Small errors accumulate over thousands of frames, causing significant drift (~37m MAE on KITTI sequence 00).

---

## Improvement 1: LiDAR ICP Odometry ⭐ Easiest

**Difficulty:** 🟢 Easy | **Expected Improvement:** 37m → 10-15m MAE

### Why It Helps

- Uses ALL LiDAR points (100k+) instead of sparse keypoints
- LiDAR depth is more accurate than stereo (~2cm vs ~10-50cm)
- Direct point cloud registration is robust

### Implementation

```python
import open3d as o3d
import numpy as np

def lidar_icp_odometry(pc_prev, pc_curr, voxel_size=0.5):
    """
    Estimate transform between consecutive LiDAR scans using ICP.

    Args:
        pc_prev: Previous point cloud (N, 3) or (N, 4)
        pc_curr: Current point cloud (N, 3) or (N, 4)
        voxel_size: Downsampling voxel size in meters

    Returns:
        4x4 transformation matrix (curr → prev)
    """
    # Convert to Open3D format
    pcd_prev = o3d.geometry.PointCloud()
    pcd_curr = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(pc_prev[:, :3])
    pcd_curr.points = o3d.utility.Vector3dVector(pc_curr[:, :3])

    # Downsample for speed
    pcd_prev = pcd_prev.voxel_down_sample(voxel_size)
    pcd_curr = pcd_curr.voxel_down_sample(voxel_size)

    # Estimate normals for point-to-plane ICP
    pcd_prev.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    pcd_curr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        pcd_curr, pcd_prev,
        max_correspondence_distance=2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )

    return result.transformation
```

### Integration into Pipeline

```python
def visual_odometry_with_lidar_icp(handler, use_lidar_odom=True, ...):
    # ... existing setup code ...

    pc_prev = None
    for i in range(num_frames - 1):
        # Get current point cloud
        pc_curr = next(handler.pcs)

        if use_lidar_odom and pc_prev is not None:
            # Use LiDAR ICP instead of visual odometry
            Tmat = lidar_icp_odometry(pc_prev, pc_curr)
        else:
            # Fallback to visual odometry
            rmat, tvec, _, _ = estimate_motion(matches, kp0, kp1, k_left, depth)
            Tmat = np.eye(4)
            Tmat[:3, :3] = rmat
            Tmat[:3, 3] = tvec.T

        T_tot = T_tot @ np.linalg.inv(Tmat)
        pc_prev = pc_curr
```

### Requirements

```bash
pip install open3d
```

---

## Improvement 2: Sensor Fusion (Visual + LiDAR) 🔄 Moderate

**Difficulty:** 🟡 Moderate | **Expected Improvement:** 37m → 8-12m MAE

### Why It Helps

- Combines strengths of both sensors
- Visual: good for rotation, texture-rich areas
- LiDAR: good for translation, structure

### Implementation: Simple Weighted Fusion

```python
def fuse_transforms(T_visual, T_lidar, w_visual=0.3, w_lidar=0.7):
    """
    Weighted fusion of visual and LiDAR transforms.

    Args:
        T_visual: 4x4 transform from visual odometry
        T_lidar: 4x4 transform from LiDAR ICP
        w_visual: Weight for visual estimate
        w_lidar: Weight for LiDAR estimate

    Returns:
        Fused 4x4 transform
    """
    from scipy.spatial.transform import Rotation, Slerp

    # Extract rotations
    R_visual = Rotation.from_matrix(T_visual[:3, :3])
    R_lidar = Rotation.from_matrix(T_lidar[:3, :3])

    # Spherical interpolation for rotation
    rotations = Rotation.concatenate([R_visual, R_lidar])
    slerp = Slerp([0, 1], rotations)
    R_fused = slerp(w_lidar)  # Interpolate toward LiDAR

    # Linear interpolation for translation
    t_fused = w_visual * T_visual[:3, 3] + w_lidar * T_lidar[:3, 3]

    # Construct fused transform
    T_fused = np.eye(4)
    T_fused[:3, :3] = R_fused.as_matrix()
    T_fused[:3, 3] = t_fused

    return T_fused
```

### Implementation: Extended Kalman Filter (More Sophisticated)

```python
class OdometryEKF:
    """Extended Kalman Filter for fusing visual and LiDAR odometry."""

    def __init__(self):
        # State: [x, y, z, roll, pitch, yaw]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 0.1

        # Process noise (motion uncertainty)
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    def predict(self, delta_pose, uncertainty):
        """Prediction step with motion model."""
        self.state += delta_pose
        self.covariance += self.Q * uncertainty

    def update(self, measurement, R):
        """Update step with measurement."""
        H = np.eye(6)  # Direct observation

        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        innovation = measurement - self.state
        self.state += K @ innovation
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

        return self.state

# Usage in pipeline
ekf = OdometryEKF()
for i in range(num_frames - 1):
    # Get both estimates
    T_visual = estimate_motion(...)  # Visual odometry
    T_lidar = lidar_icp_odometry(...)  # LiDAR ICP

    # Convert to pose deltas
    delta_visual = transform_to_pose(T_visual)
    delta_lidar = transform_to_pose(T_lidar)

    # Predict with LiDAR (more trusted)
    ekf.predict(delta_lidar, uncertainty=0.1)

    # Update with visual measurement
    R_visual = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Visual uncertainty
    fused_pose = ekf.update(delta_visual, R_visual)
```

---

## Improvement 3: Loop Closure Detection 🔁 Moderate-Hard

**Difficulty:** 🟠 Moderate-Hard | **Expected Improvement:** 37m → 5-10m MAE

### Why It Helps

- Detects when robot returns to a previously visited location
- Corrects accumulated drift by adding constraints
- Essential for long trajectories

### Implementation: Position-Based Loop Closure

```python
def detect_loop_closure_position(trajectory, current_frame, min_gap=100, threshold=10.0):
    """
    Detect loop closure based on estimated position proximity.

    Args:
        trajectory: Current trajectory estimates (N, 3, 4)
        current_frame: Current frame index
        min_gap: Minimum frame gap to consider (avoid recent frames)
        threshold: Distance threshold for loop detection (meters)

    Returns:
        loop_frame: Frame index of detected loop, or None
    """
    current_pos = trajectory[current_frame, :3, 3]

    for j in range(0, current_frame - min_gap):
        past_pos = trajectory[j, :3, 3]
        distance = np.linalg.norm(current_pos - past_pos)

        if distance < threshold:
            return j

    return None


def correct_trajectory_loop(trajectory, loop_start, loop_end):
    """
    Distribute loop closure error across trajectory.

    Args:
        trajectory: Full trajectory (N, 3, 4)
        loop_start: Earlier frame in the loop
        loop_end: Later frame (should be close to loop_start)

    Returns:
        Corrected trajectory
    """
    # Calculate position error at loop closure
    error = trajectory[loop_end, :3, 3] - trajectory[loop_start, :3, 3]

    # Distribute error linearly across frames
    for k in range(loop_start, loop_end + 1):
        alpha = (k - loop_start) / (loop_end - loop_start)
        trajectory[k, :3, 3] -= alpha * error

    return trajectory
```

### Implementation: LiDAR-Based Loop Closure (More Robust)

```python
def detect_loop_closure_lidar(current_pc, keyframe_pcs, keyframe_indices,
                               fitness_threshold=0.5, max_distance=2.0):
    """
    Detect loop closure by matching current LiDAR scan against keyframes.

    Args:
        current_pc: Current point cloud
        keyframe_pcs: List of keyframe point clouds
        keyframe_indices: Frame indices of keyframes
        fitness_threshold: Minimum ICP fitness to accept match

    Returns:
        (loop_frame, transform) or (None, None)
    """
    pcd_curr = o3d.geometry.PointCloud()
    pcd_curr.points = o3d.utility.Vector3dVector(current_pc[:, :3])
    pcd_curr = pcd_curr.voxel_down_sample(0.5)

    best_fitness = 0
    best_match = None
    best_transform = None

    for kf_pc, kf_idx in zip(keyframe_pcs, keyframe_indices):
        pcd_kf = o3d.geometry.PointCloud()
        pcd_kf.points = o3d.utility.Vector3dVector(kf_pc[:, :3])
        pcd_kf = pcd_kf.voxel_down_sample(0.5)

        # Try ICP alignment
        result = o3d.pipelines.registration.registration_icp(
            pcd_curr, pcd_kf,
            max_correspondence_distance=max_distance
        )

        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_match = kf_idx
            best_transform = result.transformation

    if best_fitness > fitness_threshold:
        return best_match, best_transform

    return None, None
```

---

## Improvement 4: Bundle Adjustment 📐 Hard

**Difficulty:** 🔴 Hard | **Expected Improvement:** 37m → 3-8m MAE

### Why It Helps

- Jointly optimizes ALL camera poses and 3D points
- Minimizes reprojection error globally
- Produces geometrically consistent reconstruction

### Concept

```
Standard VO:     Frame i → Frame i+1 (local, greedy)

Bundle Adjustment:
    Frame 0 ──┬── Frame 1 ──┬── Frame 2 ──┬── Frame 3
              │             │             │
              └──── 3D Points ────────────┘

    Optimize all poses + all points to minimize:
    Σ ||project(Pose_j, Point_i) - observed_pixel_ij||²
```

### Implementation with g2o or GTSAM

```python
# Using gtsam library
import gtsam

def bundle_adjustment(observations, initial_poses, initial_points, K):
    """
    Perform bundle adjustment using GTSAM.

    Args:
        observations: List of (frame_idx, point_idx, pixel_u, pixel_v)
        initial_poses: Initial camera poses (N, 4, 4)
        initial_points: Initial 3D points (M, 3)
        K: Camera intrinsic matrix

    Returns:
        Optimized poses and points
    """
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # Camera calibration
    cal = gtsam.Cal3_S2(K[0,0], K[1,1], 0, K[0,2], K[1,2])

    # Add pose priors for first pose (anchor)
    pose0 = gtsam.Pose3(gtsam.Rot3(initial_poses[0][:3,:3]),
                        initial_poses[0][:3,3])
    graph.add(gtsam.PriorFactorPose3(
        gtsam.symbol('x', 0), pose0,
        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
    ))

    # Add initial pose estimates
    for i, pose in enumerate(initial_poses):
        gtsam_pose = gtsam.Pose3(gtsam.Rot3(pose[:3,:3]), pose[:3,3])
        initial_values.insert(gtsam.symbol('x', i), gtsam_pose)

    # Add initial point estimates
    for j, point in enumerate(initial_points):
        initial_values.insert(gtsam.symbol('l', j), gtsam.Point3(*point))

    # Add projection factors
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    for frame_idx, point_idx, u, v in observations:
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
            gtsam.Point2(u, v), noise,
            gtsam.symbol('x', frame_idx),
            gtsam.symbol('l', point_idx),
            cal
        ))

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()

    # Extract optimized poses
    optimized_poses = []
    for i in range(len(initial_poses)):
        pose = result.atPose3(gtsam.symbol('x', i))
        T = np.eye(4)
        T[:3,:3] = pose.rotation().matrix()
        T[:3,3] = pose.translation()
        optimized_poses.append(T)

    return np.array(optimized_poses)
```

### Requirements

```bash
pip install gtsam
# or
pip install g2o-python
```

---

## Improvement 5: Full SLAM System 🗺️ Expert

**Difficulty:** 🔴🔴 Expert | **Expected Improvement:** 37m → 1-5m MAE

For production-quality results, consider using established SLAM libraries:

| System                                                  | Type   | Sensors                 | Notes                          |
| ------------------------------------------------------- | ------ | ----------------------- | ------------------------------ |
| [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)    | Visual | Mono/Stereo/RGB-D + IMU | State-of-the-art visual SLAM   |
| [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)        | LiDAR  | LiDAR + IMU             | Tightly-coupled LiDAR-inertial |
| [LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) | LiDAR  | LiDAR only              | Classic LiDAR odometry         |
| [KISS-ICP](https://github.com/PRBonn/kiss-icp)          | LiDAR  | LiDAR only              | Simple, robust LiDAR odometry  |

---

## Summary

| Improvement       | Difficulty       | Expected MAE | Implementation Time |
| ----------------- | ---------------- | ------------ | ------------------- |
| LiDAR ICP         | 🟢 Easy          | 10-15m       | 1-2 hours           |
| Sensor Fusion     | 🟡 Moderate      | 8-12m        | 2-4 hours           |
| Loop Closure      | 🟠 Moderate-Hard | 5-10m        | 4-8 hours           |
| Bundle Adjustment | 🔴 Hard          | 3-8m         | 1-2 days            |
| Full SLAM         | 🔴🔴 Expert      | 1-5m         | Integration effort  |

## Recommended Path

1. **Start with LiDAR ICP** — Quick win, significant improvement
2. **Add simple loop closure** — Fix long-term drift
3. **Consider KISS-ICP** — If you want a production solution with minimal effort

---

## References

- [ICP Algorithm](https://en.wikipedia.org/wiki/Iterative_closest_point)
- [Open3D Documentation](http://www.open3d.org/docs/release/)
- [GTSAM Tutorial](https://gtsam.org/tutorials/intro.html)
- [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
