# Project Success Tiers and Criteria

## Pass (Accelerated AI Baseline)

* **Vision:** Intel RealSense D435 dual-stream with live IR/RGB display toggle.
* **Hardware Acceleration:** NCNN ARM NEON inference on Raspberry Pi 5 — no NPU required.
* **AI Logic:** YOLOv8n-pose (NCNN) detects humans including lying-down postures at ≥ 10 FPS.
* **Performance:** Threaded inference decoupled from display maintains consistent 10–15 FPS at 320×320.
* **Logging:** Detections logged to `detections.csv` with timestamp, confidence, distance, and bounding box; video recording supported.
* **Outcome:** Human targets identified and logged in IR and RGB modes with real-time depth distance at ≥ 10 FPS.

---

## Credit (Dual-Spectrum AI and Remote Monitoring)

* **Thermal Integration:** Victor 328B thermal camera overlaid on IR feed with FOV-aligned projection and hotspot detection.
* **Dual Detection:** IR model runs primary detection; thermal blob analysis and 90°-rotated inference act as fallbacks for obscured or lying-down persons.
* **Temperature Readout:** Scene-calibrated temperature estimate displayed per detected person from thermal data.
* **Remote Monitoring:** Detection events and camera feed accessible over SSH from a remote laptop on the same network.
* **Outcome:** Dual-spectrum human detection operational with thermal confirmation and remote access — without ROS 2, which requires Ubuntu 22.04 and is incompatible with the Pi OS environment used on the Raspberry Pi 5.

---

## Distinction (Mechanical Design and Mobile Streaming)

* **Mechanical Design:** 3D CAD design of a custom ground recon drone with assembly and manufacturing plans.
* **Mobile Streaming:** Live combined IR + thermal feed streamed over WiFi to a remote laptop or phone via a lightweight HTTP/WebSocket server — chosen over ROS 2 transport for deployment practicality on Pi OS.
* **Outcome:** Complete drone body design with verified remote streaming capability.

---

## High Distinction (Custom PCB and Physical Assembly)

* **Physical Manufacturing:** Fabrication of the ground recon drone chassis based on the Distinction tier CAD designs.
* **System Integration:** Mounting the AI vision system onto the chassis with RGB, thermal, and IR sensors integrated.
* **Wire Management:** Clean wiring loom or PCB hub managing LiPo power distribution to Pi 5 and motors.
* **Night Vision Hardware:** IR and auxiliary LEDs added for low-light camera support.
* **Outcome:** Ruggedised mobile prototype with professional power management and low-light capability.

---

## Perfect (Active Stability and Target Mapping)

* **Active Stability:** Motor compensation logic keeps the camera level during acceleration.
* **High FPS:** Inference pipeline optimised to achieve consistent ≥ 30 FPS at 320×320 resolution.
* **Target Mapping:** 2D coordinate log marks where each person was detected relative to the starting point.
* **Automation:** Drone halts and triggers an alert automatically when a person is detected.
* **Outcome:** Fully realised SAR tool with active balance, 30+ FPS detection, and a logged map of found locations.
