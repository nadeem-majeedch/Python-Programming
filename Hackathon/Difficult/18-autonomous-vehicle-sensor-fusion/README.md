# Problem 18: Autonomous Vehicle Sensor Fusion

## Domain
Multi-modal Data Fusion

## Problem Statement
An autonomous vehicle is equipped with three types of sensors: Camera, LiDAR, and Radar. Each sensor provides complementary information about the vehicle's surroundings, but each has limitations. Cameras provide rich visual data but struggle in poor lighting. LiDAR provides precise 3D geometry but has limited range in rain. Radar works in all conditions but has lower resolution. Sensors can also fail intermittently (dropouts).

Your task is to fuse data from these heterogeneous sensors to classify obstacles as Pedestrian, Cyclist, Car, Truck, or None. You must handle sensor failures gracefully and provide uncertainty estimates for your predictions.

## Objectives
1. Achieve >90% classification accuracy when all sensors are operating
2. Maintain >80% accuracy when any single sensor fails
3. Handle conflicting sensor readings and quantify prediction uncertainty
4. Compare early fusion, late fusion, and hybrid fusion strategies

## Dataset
- 2000 synchronized time steps from 3 sensors
- Camera: object class probabilities for 5 classes (10-dimensional features)
- LiDAR: x,y,z coordinates of nearest object (3 features)
- Radar: speed, distance, angle (3 features)
- Ground truth: obstacle type (Pedestrian/Cyclist/Car/Truck/None)
- Sensors have different noise levels and intermittent dropouts
- CSV at `data/sensor_data.csv`

## Success Criteria
- **All sensors**: Accuracy > 90%
- **One sensor failure**: Accuracy > 80%
- **Graceful degradation**: Performance degrades smoothly as sensors drop out
- **Uncertainty**: Reasonable confidence scores that correlate with correctness

## Starter Code
`starter_code.py` loads synchronized data and implements a simple voting ensemble. You must extend with proper fusion strategies.

## Constraints
- Must handle missing sensor data (NaN values)
- Must evaluate under multiple sensor failure scenarios
- Must report per-class performance and confusion matrix
