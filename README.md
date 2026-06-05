# CARLA Driver Monitoring System (DMS)

A real-time Driver Monitoring System (DMS) built on the CARLA simulator and Intel RealSense cameras. The framework integrates gaze estimation, face anti-spoofing, and driver action recognition into a unified simulation environment for intelligent transportation and driver behavior research.


## Overview

This project extends the CARLA simulator with an Intel RealSense RGB-D camera and multiple AI-based driver monitoring modules. It enables real-time analysis of driver behavior inside a simulated vehicle environment while simultaneously recording CARLA and RealSense video streams.

The system integrates:

- Driver Gaze Estimation (TGGNet)
- Face Anti-Spoofing (RGB-Depth-IR)
- Driver Action Recognition (Vigi-CLIP)
- Intel RealSense RGB-D Sensing
- CARLA Driving Simulation
- Real-Time Visualization and Recording

---

## Features

### Driver Gaze Estimation

- MediaPipe face landmark extraction
- Graph-based gaze prediction using TGGNet
- Real-time gaze vector visualization
- Gaze compass and eye tracking overlay

### Face Anti-Spoofing

- RGB, depth, and infrared fusion
- Real-time liveness verification
- Deep learning-based spoof detection
- Confidence visualization

### Driver Action Recognition

- ViFi-CLIP-based action recognition
- Real-time driver distraction detection
- Temporal video understanding
- 16 driver behavior classes

Supported actions include:

- Safe Driving
- Texting
- Phone Conversation
- Drinking
- Smoking
- GPS Operation
- Radio Adjustment
- Talking to Passenger
- Fatigue and Drowsiness
- Reaching Behind
- And More

### CARLA Integration

- Manual vehicle control using Logitech G29 steering wheel
- Multi-camera support
- Weather control
- Real-time simulation

### Intel RealSense Integration

- RGB stream
- Depth stream
- Infrared stream
- Video recording

---

## Driver Monitoring Modes

The system supports four operational modes:

| Mode | Description |
|--------|-------------|
| Gaze | Driver gaze estimation |
| Anti-Spoofing | Face liveness verification |
| Action | Driver activity recognition |
| ALL | Run all modules simultaneously |

Press **M** to switch between modes.

---

## Controls

| Key | Function |
|------|----------|
| M | Switch DMS mode |
| T | Start/Stop video recording |
| P | Toggle autopilot |
| TAB | Change camera |
| C | Change weather |
| H | Help |
| ESC | Exit |

---

## System Architecture

```text
Intel RealSense Camera
        │
        ├── RGB Stream
        ├── Depth Stream
        └── IR Stream
                │
                ▼
        Driver Monitoring System
        ├── Gaze Estimation
        ├── Face Anti-Spoofing
        └── Action Recognition
                │
                ▼
      Real-Time Visualization
                │
                ▼
           CARLA Simulator
```

## Requirements

### Software

- CARLA 0.9.16
- Unreal Engine 5.5 (optional)
- Python 3.8+

### Hardware

- Intel RealSense Camera
- Logitech G29 Steering Wheel (optional)

### Dependencies

```bash
pip install carla
pip install pyrealsense2
pip install opencv-python
pip install pygame
pip install mediapipe
pip install torch
pip install torch-geometric
pip install networkx
pip install numpy
```

## Running the System

Start the CARLA server:

```bash
./CarlaUE5.sh
```

Launch the Driver Monitoring System:

```bash
python carla_realsense_dms.py
```

Run with autopilot:

```bash
python carla_realsense_dms.py --autopilot
```

Custom resolution:

```bash
python carla_realsense_dms.py --res 2560x1440
```

---

## Output

Recorded videos are automatically saved to:

```text
output/
├── carla_YYYYMMDD_HHMMSS.mp4
└── realsense_rgb_YYYYMMDD_HHMMSS.mp4
```

---

## Research Applications

- Driver Vigilance Monitoring
- Driver Distraction Detection
- Human Behavior Analysis
- Human-Machine Interaction
- Autonomous Driving Research
- Intelligent Transportation Systems
- Digital Twin Driver Simulation

---

