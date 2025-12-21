● Modern SLAM Pipeline Architecture

┌─────────────────────────────────────────────────────────────────────────────────┐
│ MODERN SLAM SYSTEM PIPELINE │
├─────────────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ Camera │ │ LiDAR │ │ IMU │ ← SENSORS │
│ └────┬────┘ └────┬────┘ └────┬────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────────────────────────────┐ │
│ │ FRONTEND (Real-time) │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Feature Extraction/Tracking │ │ ← Detect & match features │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Frame-to-Frame Odometry │ │ ← Estimate motion (what we built) │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ IMU Pre-integration │ │ ← Fuse IMU between frames │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Keyframe Decision │ │ ← "Is this frame important?" │
│ │ └───────────────┬───────────────┘ │ │
│ └──────────────────┼──────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────┐ │
│ │ LOCAL MAPPING (Near real-time) │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Local Bundle Adjustment │ │ ← Optimize recent N keyframes │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Map Point Culling │ │ ← Remove bad map points │
│ │ └───────────────┬───────────────┘ │ │
│ └──────────────────┼──────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────┐ │
│ │ LOOP CLOSING (Background) │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Place Recognition │ │ ← "Have I been here before?" │
│ │ │ (DBoW2, NetVLAD, ScanContext)│ │ │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Loop Constraint Estimation │ │ ← Compute relative pose │
│ │ └───────────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ ┌───────────────────────────────┐ │ │
│ │ │ Pose Graph Optimization │ │ ← Correct ALL poses │
│ │ └───────────────────────────────┘ │ │
│ └─────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────┐ │
│ │ OUTPUT │ │
│ │ • Drift-corrected trajectory │ │
│ │ • 3D Map (point cloud/mesh) │ │
│ │ • Real-time pose estimate │ │
│ └─────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────────────────┘

---

The Three Main Threads

┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ TRACKING │ │ LOCAL MAPPING │ │ LOOP CLOSING │
│ (Frontend) │ │ (Backend) │ │ (Backend) │
├────────────────┤ ├────────────────┤ ├────────────────┤
│ • Real-time │ │ • Near RT │ │ • Background │
│ • Every frame │ │ • Keyframes │ │ • When needed │
│ • Fast │ │ • Medium │ │ • Slow OK │
│ │ │ │ │ │
│ Input: │ │ Input: │ │ Input: │
│ Raw sensors │ │ Keyframes │ │ All keyframes │
│ │ │ │ │ │
│ Output: │ │ Output: │ │ Output: │
│ Current pose │ │ Local map │ │ Corrected map │
│ Keyframe? │ │ Refined poses │ │ Drift-free │
└────────────────┘ └────────────────┘ └────────────────┘
│ │ │
└───────────────────┴────────────────────┘
│
Shared Map & Poses

---

Example: LIO-SAM Pipeline (LiDAR-Inertial SLAM)

                      IMU (400Hz)
                          │
                          ▼
              ┌───────────────────────┐
              │   IMU Pre-integration │
              │   (predict motion     │
              │    between LiDAR)     │
              └───────────┬───────────┘
                          │
      LiDAR (10Hz)        │
          │               │
          ▼               ▼

┌───────────────────────────────────┐
│ Feature Extraction │
│ • Edge points (high curvature) │
│ • Planar points (low curvature) │
└───────────────┬───────────────────┘
│
▼
┌───────────────────────────────────┐
│ LiDAR Odometry │
│ • Scan-to-scan matching │
│ • IMU prediction as initial │
└───────────────┬───────────────────┘
│
▼
┌───────────────────────────────────┐
│ LiDAR Mapping │
│ • Scan-to-map matching │
│ • Local map (sliding window) │
└───────────────┬───────────────────┘
│
▼
┌───────────────────────────────────┐
│ Factor Graph Optimization │
│ • IMU factors │
│ • LiDAR odometry factors │
│ • GPS factors (if available) │
│ • Loop closure factors │
└───────────────┬───────────────────┘
│
▼
Optimized Trajectory

---

Factor Graph (Heart of Modern SLAM)

Instead of sequential estimation, modern SLAM uses factor graphs:

Factor Graph Visualization:

      Pose1 ──── Pose2 ──── Pose3 ──── Pose4 ──── Pose5
        │    odom   │   odom   │   odom   │   odom   │
        │          │          │          │          │
       IMU        IMU        IMU        IMU        IMU
        │          │          │          │          │
        └─── GPS   └─── GPS                    Loop─┘
                                                │
                                                │
      Pose1 ◄────────────────────────────────────┘

Legend:
○ Pose node (variable to optimize)
│ Factor (constraint between variables)

Factors:
• Odometry: "Pose2 is HERE relative to Pose1"
• IMU: "Acceleration/rotation was THIS"
• GPS: "Absolute position is approximately HERE"
• Loop: "Pose5 ≈ Pose1" (revisited location)

---

What We Built vs Full SLAM

| Component           | What We Built | Full SLAM System     |
| ------------------- | ------------- | -------------------- |
| Feature extraction  | ✅ SIFT       | ORB (faster)         |
| Frame-to-frame odom | ✅ PnP        | PnP + motion model   |
| LiDAR odometry      | ✅ Basic ICP  | LOAM/KISS-ICP        |
| Sensor fusion       | ✅ EKF        | Factor graph         |
| IMU integration     | ❌ No         | ✅ Pre-integration   |
| Keyframes           | ❌ No         | ✅ Yes               |
| Local BA            | ❌ No         | ✅ Sliding window    |
| Loop closure        | ❌ No         | ✅ Place recognition |
| Pose graph opt      | ❌ No         | ✅ g2o/GTSAM         |
| GPS fusion          | ❌ No         | ✅ As factor         |

---

Popular Modern SLAM Systems

| System    | Type                   | Key Features                 |
| --------- | ---------------------- | ---------------------------- |
| ORB-SLAM3 | Visual/Visual-Inertial | Multi-map, IMU, loop closure |
| LIO-SAM   | LiDAR-Inertial         | Factor graph, GPS, robust    |
| LOAM      | LiDAR                  | Edge/planar features         |
| KISS-ICP  | LiDAR                  | Simple, robust ICP           |
| VINS-Mono | Visual-Inertial        | Monocular + IMU              |
| R3LIVE    | LiDAR-Visual-Inertial  | All sensors fused            |

---

Simplified Code Structure

class ModernSLAM:
def **init**(self):
self.keyframes = []
self.map_points = []
self.factor_graph = FactorGraph()
self.place_recognizer = DBoW2()

      def process_frame(self, image, lidar, imu_data):
          # 1. TRACKING (real-time)
          features = self.extract_features(image)
          pose_estimate = self.track(features, lidar, imu_data)

          # 2. KEYFRAME DECISION
          if self.is_keyframe(pose_estimate):
              kf = self.create_keyframe(image, lidar, pose_estimate)
              self.keyframes.append(kf)

              # 3. LOCAL MAPPING (parallel thread)
              self.local_mapping_thread.add(kf)

              # 4. LOOP CLOSING (parallel thread)
              self.loop_closing_thread.check(kf)

          return pose_estimate

      def local_mapping(self, keyframe):
          # Optimize recent keyframes
          recent_kfs = self.get_recent_keyframes(n=10)
          self.bundle_adjustment(recent_kfs)

      def loop_closing(self, keyframe):
          # Check for loop
          match = self.place_recognizer.query(keyframe)
          if match:
              # Compute loop constraint
              relative_pose = self.compute_relative_pose(keyframe, match)

              # Add to factor graph and optimize ALL poses
              self.factor_graph.add_loop_factor(keyframe, match, relative_pose)
              self.factor_graph.optimize()  # Corrects drift!

---

Key Takeaway

Your Pipeline: Modern SLAM:

Sensors → Odometry → EKF Sensors → Frontend → Local BA → Loop Closure
│ │
Still drifts Factor Graph Optimization
│
Drift Corrected!

The factor graph + loop closure combination is what makes modern SLAM drift-free. Everything else (better features, IMU, etc.) just reduces the drift rate.
