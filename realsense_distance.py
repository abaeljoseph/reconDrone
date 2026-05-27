#!/usr/bin/env python3
"""
Intel RealSense D435 - Depth / Infrared / Colour Stream with Closest Object Distance
- Green crosshair on the closest object with distance label in metres
Press Q to quit.

Usage:
  python realsense_distance.py                   # colour mode (default)
  python realsense_distance.py --mode depth      # colourised depth map
  python realsense_distance.py --mode ir         # infrared mode
"""

import argparse
import pyrealsense2 as rs
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["depth", "ir", "colour"], default="colour",
                    help="Camera mode: 'depth', 'ir', or 'colour'")
args = parser.parse_args()

pipeline = rs.pipeline()
config   = rs.config()

if args.mode == "ir":
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8,  15)
    config.enable_stream(rs.stream.depth,       640, 480, rs.format.z16, 15)
    win_title = "RealSense D435 — Infrared"
elif args.mode == "colour":
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  15)
    win_title = "RealSense D435 — Colour"
else:
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    win_title = "RealSense D435 — Depth"

align = rs.align(rs.stream.color) if args.mode == "colour" else None

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 2)

pipeline.start(config)
cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
print(f"Streaming in {args.mode.upper()} mode — press Q to quit")

try:
    while True:
        frames      = pipeline.wait_for_frames(5000)
        aligned     = align.process(frames) if align else frames
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            continue

        # Build display image
        if args.mode == "ir":
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue
            display  = cv2.cvtColor(np.asanyarray(ir_frame.get_data()), cv2.COLOR_GRAY2BGR)
        elif args.mode == "colour":
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue
            display = np.asanyarray(color_frame.get_data())
        else:
            display = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Find closest non-zero pixel using depth data (available in both modes)
        depth_image = np.asanyarray(depth_frame.get_data())
        valid_mask  = depth_image > 0
        if valid_mask.any():
            cy, cx     = np.unravel_index(
                np.where(valid_mask, depth_image, 65535).argmin(),
                depth_image.shape
            )
            distance_m = depth_frame.get_distance(int(cx), int(cy))

            cv2.drawMarker(display, (int(cx), int(cy)), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(display, f"Closest: {distance_m:.3f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(win_title, display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stream stopped.")
