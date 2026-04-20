# Project Success Tiers and Criteria

This document defines the technical development roadmap, ensuring the AI and vision systems are established before moving into custom hardware and active stability.

## Pass (Accelerated AI Baseline)

* **Vision:** Single RGB video stream using the Intel RealSense D435.

*  **Hardware Acceleration:** Full integration of the Hailo 8 AI HAT plus to offload processing from the CPU.

*  **AI Logic:** Real-time human detection using the YOLOv8n model.

*  **Performance:** Achieving a consistent ≥ 30 FPS detection rate on a stationary test bench.

*  **Outcome:** Successful identification and logging of a human target with high frame rates.

---

## Credit (ROS 2 Integration and Thermal Vision)

*  **Environment Setup:** Installation of ROS 2 Humble on Ubuntu 22.04.

*  **Communication:** Using standard ROS 2 packages to view the camera feed and AI bounding boxes in a basic visualiser like RViz.

*  **Thermal Integration:** Adding a thermal camera feed into the ROS 2 environment for dual-spectrum capability.

*  **AI Tuning:** Testing the system with custom datasets for SAR-specific postures such as lying down.

*  **Outcome:** A functional system where the AI and camera communicate via ROS 2 topics with dual-spectrum (RGB + thermal) capability.

---

## Distinction (Mechanical Design and Mobile Streaming)

*  **Mechanical Design:** Completion of a 3D CAD design for a custom ground recon drone, including assembly steps and manufacturing plans.

*  **Mobile Streaming:** Live video feed from the unit viewed on a remote laptop or phone via a shared WiFi network using ROS 2 transport tools.

*  **Outcome:** A complete design roadmap for the drone body with verified remote streaming capability.

---

## High Distinction (Custom PCB and Physical Assembly)

*  **Physical Manufacturing:** Fabrication of the ground recon drone chassis based on the Distinction tier CAD designs.

*  **System Integration:** Mounting the AI vision system onto the chassis and integrating all sensors (RGB, Thermal, and NPU).

*  **Wire Management:** Implementation of a clean wiring loom or a simplified PCB hub to manage power distribution from LiPo batteries to the Pi 5 and motors.

*  **Night Vision Hardware:** Integration of auxiliary LED and IR LED lighting to assist cameras in low light.

*  **Outcome:** A ruggedised mobile prototype with professional-grade power management and enhanced low-light visibility.

---

## Perfect (Active Stability and Target Mapping)

*  **Active Stability:** Implementation of motor compensation logic to prevent the drone from tipping forward during acceleration to keep the camera level.

*  **Target Mapping:** The system generates a simple 2D coordinate log or marker system showing where each survivor was detected relative to the starting point.

*  **Automation:** Basic safety features such as the drone stopping automatically or triggering an alert when the AI detects a person.

*  **Outcome:** A fully realised search and rescue tool that maintains balance and provides the operator with a list of found locations.
