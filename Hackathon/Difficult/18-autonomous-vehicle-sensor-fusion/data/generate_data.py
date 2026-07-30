#!/usr/bin/env python3
"""Generate synthetic autonomous vehicle sensor fusion data."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_timesteps = 2000
obstacle_types = ["Pedestrian", "Cyclist", "Car", "Truck", "None"]
sensor_failure_rate = 0.05  # 5% dropout per sensor

records = []
for t in range(n_timesteps):
    # Ground truth
    gt = np.random.choice(obstacle_types, p=[0.15, 0.10, 0.25, 0.10, 0.40])
    
    # Camera features (object class probabilities - softmax output)
    camera_base = np.zeros(5)
    if gt != "None":
        gt_idx = obstacle_types.index(gt)
        camera_base[gt_idx] = np.random.uniform(0.7, 0.95)
        # Confuse with similar classes
        if gt in ["Car", "Truck"]:
            alt_idx = obstacle_types.index("Truck" if gt == "Car" else "Car")
            camera_base[alt_idx] = np.random.uniform(0.0, 0.2)
        elif gt in ["Pedestrian", "Cyclist"]:
            alt_idx = obstacle_types.index("Cyclist" if gt == "Pedestrian" else "Pedestrian")
            camera_base[alt_idx] = np.random.uniform(0.0, 0.15)
        # Remaining probability to others
        remaining = 1.0 - camera_base.sum()
        for i in range(5):
            if camera_base[i] == 0:
                camera_base[i] = np.random.uniform(0, remaining * 0.3)
        camera_base = camera_base / camera_base.sum()  # renormalize
    else:
        # No obstacle - uniform-ish distribution
        camera_base = np.random.dirichlet(np.ones(5) * 2)
    
    camera_feat = camera_base + np.random.normal(0, 0.02, 5)
    camera_feat = np.clip(camera_feat, 0, 1)
    camera_feat = camera_feat / camera_feat.sum()
    
    # Camera dropout
    if np.random.random() < sensor_failure_rate:
        camera_feat = np.full(5, np.nan)
    
    # LiDAR features (x, y, z of nearest obstacle)
    if gt == "None":
        lidar_feat = np.array([np.random.uniform(50, 100),
                               np.random.uniform(-10, 10),
                               np.random.uniform(0, 0.5)])
    else:
        # Distance based on obstacle type
        dist_ranges = {"Pedestrian": (5, 30), "Cyclist": (5, 40),
                       "Car": (5, 80), "Truck": (5, 100)}
        d_min, d_max = dist_ranges[gt]
        dist = np.random.uniform(d_min, d_max)
        angle = np.random.uniform(-30, 30)
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        z = np.random.uniform(0.5, 2.5) if gt == "Pedestrian" else np.random.uniform(1.0, 3.0)
        lidar_feat = np.array([x, y, z])
        # Add noise
        lidar_noise = np.array([0.1, 0.05, 0.02]) * dist / 10
        lidar_feat += np.random.normal(0, lidar_noise)
    
    # LiDAR dropout
    if np.random.random() < sensor_failure_rate:
        lidar_feat = np.full(3, np.nan)
    
    # Radar features (speed, distance, angle)
    if gt == "None":
        radar_feat = np.array([np.random.uniform(-1, 1),
                               np.random.uniform(80, 120),
                               np.random.uniform(-5, 5)])
    else:
        speed_ranges = {"Pedestrian": (1, 6), "Cyclist": (10, 30),
                        "Car": (20, 80), "Truck": (10, 60)}
        speed = np.random.uniform(*speed_ranges[gt])
        dist_ranges_r = {"Pedestrian": (5, 30), "Cyclist": (5, 40),
                         "Car": (10, 100), "Truck": (10, 120)}
        d_min, d_max = dist_ranges_r[gt]
        dist = np.random.uniform(d_min, d_max)
        angle = np.random.uniform(-30, 30) + np.random.normal(0, 2)
        radar_feat = np.array([speed, dist, angle])
        # Radar is more noise-resistant but less accurate
        radar_feat += np.random.normal(0, [0.5, 0.5, 1.0])
    
    # Radar dropout
    if np.random.random() < sensor_failure_rate:
        radar_feat = np.full(3, np.nan)
    
    row = {"Timestep": t, "ObstacleType": gt}
    row.update({f"Camera_{i}": camera_feat[i] for i in range(5)})
    row.update({f"LiDAR_{'xyz'[i]}": lidar_feat[i] for i in range(3)})
    row.update({f"Radar_{['Speed','Dist','Angle'][i]}": radar_feat[i] for i in range(3)})
    records.append(row)

df = pd.DataFrame(records)
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sensor_data.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} sensor readings -> {out_path}")
print(f"Obstacle distribution:")
print(df["ObstacleType"].value_counts())
