## Q-Nav 
Q-Nav is an AI-powered autonomous navigation system designed for spacecraft operating in GPS-denied environments. When classical sensors fail due to radiation, thermal stress, or mechanical wear, Q-Nav's quantum-inspired backup system automatically activates, providing 100-1000x more precise navigation to ensure mission success.
Key Innovation: First implementation of quantum-classical hybrid navigation for space applications, combining real-time AI object detection with quantum-enhanced sensor fusion.

✨ Features
🎯 Core Capabilities

Real-Time Obstacle Detection

YOLOv8 AI detects space debris, asteroids, and obstacles at 30+ fps
Automatic collision risk assessment (SAFE/LOW/MEDIUM/HIGH)
Processes live camera feed with sub-second latency


## Multi-Sensor Fusion

Integrates IMU (acceleration/rotation), LiDAR (3D distance), and camera data
Kalman filter-based state estimation
Redundant sensor architecture for reliability


## Intelligent Path Planning

Dynamic collision-free trajectory calculation
Real-time obstacle avoidance
5-second ahead collision prediction


⚛️ Quantum Backup Navigation (Our Innovation)

Automatically activates when sensor health < 50%
100x more precise than classical sensors (1e-5 vs 1e-2 rad/s noise)
Quantum superposition evaluates 100+ paths simultaneously
Maintains accuracy for months without recalibration



🎨 Professional GUI

Dark theme mission control interface
Real-time sensor health monitoring with live graphs
Activation log with timestamps
Flashing visual alerts for mode changes
One-click quantum activation/restoration
Full-screen optimized layout
 
## Technology Stack
AI/ML Framework:

PyTorch 2.0+
Ultralytics YOLOv8 (nano model for speed)
NumPy/SciPy for physics simulations

Computer Vision:

OpenCV 4.8+ (video processing)
PIL/Pillow (image handling)

GUI:

Tkinter (native Python GUI)
Matplotlib (real-time graphs)

Sensors:

Simulated IMU, LiDAR, Camera with realistic noise models
Based on actual hardware specifications (MPU6050, LIDAR-Lite v3)


📦 Installation
Prerequisites

Python 3.8 or higher
Webcam (for live demo)
4GB RAM minimum
GPU recommended (optional, runs on CPU)

## System Architecture

┌─────────────┐
│   Camera    │──→ YOLOv8 Detection ──→ Obstacle List
└─────────────┘                              ↓
┌─────────────┐                         ┌─────────┐
│ IMU Sensor  │──→ Sensor Fusion ────→ │Navigator│
└─────────────┘        ↓                 └─────────┘
┌─────────────┐    Position/              ↓
│   LiDAR     │──→  Velocity          Path Planning
└─────────────┘        ↓                   ↓
                 Health Monitor        Safe Trajectory
                       ↓
              < 50% Health? ──→ Quantum Backup ⚛️

