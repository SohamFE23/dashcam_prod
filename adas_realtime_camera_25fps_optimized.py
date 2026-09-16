"""
ADAS REALTIME CAMERA INFERENCE
Raspberry Pi + USB/Pi Camera
Model: YOLOv11x Segmentation with Perimeter Zone Detection
Parameters: DO NOT CHANGE
"""

import cv2
import numpy as np
import time
from collections import deque

from ultralytics import YOLO
from picamera2 import Picamera2

# ============================================================
# 1. MODEL & PATHS
# ============================================================
WEIGHTS_PATH = "/home/soham/Dashcam/best.onnx"

# ============================================================
# 2. CLASS-TO-COLOR MAPPING (BGR format) - DUPLICATED FOR SAFETY
# ============================================================
class_colors = {
    0: (0, 0, 255),        # Car = Red
    1: (0, 165, 255),      # Bus = Orange
    2: (0, 255, 0),        # Full Dashed White = Green
    3: (255, 0, 255),      # Pedestrian = Magenta
    4: (255, 0, 0),        # Truck = Blue
    5: (0, 255, 255)       # Yellow Lane = Yellow
}

class_names = {
    0: "Bus",
    1: "Car", 
    2: "Dashed Lane",
    3: "Pedestrian",
    4: "Truck",
    5: "Yellow Lane"
}

# ============================================================
# 3. CLASS-TO-COLOR MAPPING (BGR format) - DUPLICATED FOR SAFETY (BLOCK 41 - FULLY CORRECT)
# ============================================================
class_colors = {
    0: (0, 0, 255),        # Car = Red
    1: (0, 165, 255),      # Bus = Orange
    2: (0, 255, 0),        # Full Dashed White = Green
    3: (255, 0, 255),      # Pedestrian = Magenta
    4: (255, 0, 0),        # Truck = Blue
    5: (0, 255, 255)       # Yellow Lane = Yellow
}

class_names = {
    0: "Bus",
    1: "Car", 
    2: "Dashed Lane",
    3: "Pedestrian",
    4: "Truck",
    5: "Yellow Lane"
}

# ============================================================
# 4. PERIMETER & RADAR SETUP (BLOCK 41 - FULLY CORRECT NO MISTAKE)
# ============================================================
raw_pts = [
    [440, 430],
    [548, 270],
    [710, 268],
    [828, 411],
    [712, 398],
    [632, 396],
    [553, 405],
    [440, 430]
]

# ============================================================
# 5. INITIALIZE MODEL
# ============================================================

print("Loading YOLO ONNX model...")

# best.onnx is used exactly as supplied.
# The Pi 4 has no CUDA GPU, so inference runs on CPU.
model = YOLO(WEIGHTS_PATH, task="detect")

device_mode = "CPU"

# ------------------------------------------------------------
# SPEED SETTINGS
# ------------------------------------------------------------
# 320 is much faster than the previous 1024 input.
# The display/camera still runs at 25 FPS; YOLO inference
# runs continuously in a separate worker and always uses
# the newest available camera frame.
INFERENCE_SIZE = 320
CONF_THRESHOLD = 0.25
MAX_DETECTIONS = 20

print(f"Device Mode: {device_mode}")
print(f"YOLO inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")

# 6. CAMERA SETUP - RASPBERRY PI CSI CAMERA
# ============================================================

# Camera/display target
w = 640
h = 480
camera_fps = 25

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (w, h),
        "format": "RGB888"
    },
    controls={
        "FrameRate": camera_fps
    },
    buffer_count=4
)

picam2.configure(camera_config)
picam2.start()

time.sleep(2)

print("Camera started successfully")
print(f"Camera Resolution: {w} x {h} @ {camera_fps} FPS")


# ============================================================
# CAMERA WRAPPER
# ============================================================

class PicameraCapture:

    def isOpened(self):
        return True

    def read(self):
        try:
            frame = picam2.capture_array("main")

            # Picamera2 gives RGB888; OpenCV uses BGR.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return True, frame

        except Exception as e:
            print(f"Camera capture error: {e}")
            return False, None

    def release(self):
        try:
            picam2.stop()
        except Exception:
            pass


cap = PicameraCapture()

# 7. POLYGON SCALING (FULLY CORRECT BLOCK 41)
# ============================================================
# Reference image = 1280 x 720
scale_x = w / 1280.0
scale_y = h / 720.0

ZONE_POINTS = np.array([
    [int(x * scale_x), int(y * scale_y)]
    for x, y in raw_pts
], dtype=np.int32)

poly_center_x = (
    np.min(ZONE_POINTS[:, 0]) +
    np.max(ZONE_POINTS[:, 0])
) // 2

print(f" Polygon scaling applied: {scale_x:.2f}x, {scale_y:.2f}x")
print(f" Polygon center X: {poly_center_x}")

# ============================================================
# 8. RADAR DIMENSIONS
# ============================================================
RADAR_W, RADAR_H = 600, 900
SCALE_X_RADAR = RADAR_W / w
SCALE_Y_RADAR = RADAR_H / h

# ============================================================
# 9. FPS COUNTERS
# ============================================================

# Display FPS
fps_deque = deque(maxlen=30)

# YOLO inference FPS
inference_fps_deque = deque(maxlen=30)

last_display_time = time.time()
last_inference_time = time.time()

# 10. MAIN LOOP - LOW-LATENCY CAMERA + BACKGROUND YOLO
# ============================================================

# IMPORTANT:
# The camera/display loop is kept separate from YOLO inference.
# This prevents a slow YOLO model from freezing the camera window.
#
# If YOLO takes longer than one frame, old inference results are
# displayed until the next result is ready. New camera frames are
# never allowed to build up a long queue.

import threading

latest_frame = None
latest_result = None
latest_result_frame = None

frame_lock = threading.Lock()
result_lock = threading.Lock()

stop_event = threading.Event()

frame_idx = 0
inference_count = 0


def inference_worker():
    global latest_frame
    global latest_result
    global latest_result_frame
    global inference_count

    local_last_time = time.time()

    while not stop_event.is_set():

        # Get newest frame only.
        with frame_lock:
            if latest_frame is None:
                frame_for_inference = None
            else:
                frame_for_inference = latest_frame.copy()

        if frame_for_inference is None:
            time.sleep(0.001)
            continue

        try:
            # Fast CPU inference.
            result = model(
                frame_for_inference,
                conf=CONF_THRESHOLD,
                imgsz=INFERENCE_SIZE,
                max_det=MAX_DETECTIONS,
                verbose=False
            )[0]

            now = time.time()
            dt = now - local_last_time

            if dt > 0:
                inference_fps_deque.append(1.0 / dt)

            local_last_time = now

            # Store newest result.
            with result_lock:
                latest_result = result
                latest_result_frame = frame_for_inference

            inference_count += 1

        except Exception as e:
            print(f"YOLO inference error: {e}")
            time.sleep(0.05)


# Start YOLO in a background thread.
inference_thread = threading.Thread(
    target=inference_worker,
    daemon=True
)
inference_thread.start()

print("")
print("Starting REALTIME ADAS...")
print("Camera target: 25 FPS")
print(f"YOLO input: {INFERENCE_SIZE} x {INFERENCE_SIZE}")
print("Press 'q' to quit, 's' to save frame")
print("")


try:
    while cap.isOpened():

        # --------------------------------------------------------
        # 1. Capture newest camera frame
        # --------------------------------------------------------
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to capture frame")
            continue

        frame_idx += 1

        # Give the inference worker the newest frame.
        # Old frames are deliberately discarded for low latency.
        with frame_lock:
            latest_frame = frame.copy()

        # --------------------------------------------------------
        # 2. Get newest YOLO result
        # --------------------------------------------------------
        with result_lock:
            results = latest_result

        # Start with the raw camera image.
        overlay = frame.copy()

        radar_view = np.zeros(
            (RADAR_H, RADAR_W, 3),
            dtype=np.uint8
        )

        signal = "Path Clear"
        signal_color = (255, 255, 255)

        # --------------------------------------------------------
        # 3. PERIMETER ZONE
        # --------------------------------------------------------
        perimeter_overlay = overlay.copy()

        cv2.fillPoly(
            perimeter_overlay,
            [ZONE_POINTS],
            (100, 200, 255)
        )

        cv2.polylines(
            overlay,
            [ZONE_POINTS],
            True,
            (255, 255, 0),
            3
        )

        overlay = cv2.addWeighted(
            perimeter_overlay,
            0.25,
            overlay,
            0.75,
            0
        )

        # --------------------------------------------------------
        # 4. YOLO SEGMENTATION / DETECTION PROCESSING
        # --------------------------------------------------------
        if results is not None and results.masks is not None:

            classes = results.boxes.cls.cpu().numpy()
            masks = results.masks.xy

            for i, mask_data in enumerate(masks):

                cls = int(classes[i])
                pts = np.array(mask_data, dtype=np.int32)

                if len(pts) < 3:
                    continue

                color = class_colors.get(
                    cls,
                    (200, 200, 200)
                )

                # Lanes
                if cls in [2, 5]:
                    cv2.polylines(
                        overlay,
                        [pts],
                        True,
                        color,
                        4
                    )

                # Vehicles / pedestrians
                else:
                    cv2.fillPoly(
                        overlay,
                        [pts],
                        color
                    )

                # ------------------------------------------------
                # RADAR
                # ------------------------------------------------
                radar_pts = (
                    pts *
                    [SCALE_X_RADAR, SCALE_Y_RADAR]
                ).astype(np.int32)

                radar_color = class_colors.get(
                    cls,
                    (200, 200, 200)
                )

                if cls in [2, 5]:
                    cv2.polylines(
                        radar_view,
                        [radar_pts],
                        True,
                        radar_color,
                        3
                    )
                else:
                    cv2.fillPoly(
                        radar_view,
                        [radar_pts],
                        radar_color
                    )

                # ------------------------------------------------
                # STEERING LOGIC
                # ------------------------------------------------
                if cls == 2 or cls == 5:

                    for p in pts:

                        inside = cv2.pointPolygonTest(
                            ZONE_POINTS,
                            (float(p[0]), float(p[1])),
                            False
                        )

                        if inside >= 0:

                            if p[0] < poly_center_x:
                                signal = "Turn Right"
                                signal_color = (0, 165, 255)
                            else:
                                signal = "Turn Left"
                                signal_color = (0, 255, 255)

                            break

                # ------------------------------------------------
                # CLASS LABEL
                # ------------------------------------------------
                M = cv2.moments(pts)

                if M["m00"] != 0:

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    label = class_names.get(
                        cls,
                        "Unknown"
                    )

                    cv2.putText(
                        overlay,
                        label,
                        (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # --------------------------------------------------------
        # 5. BLEND
        # --------------------------------------------------------
        output_frame = cv2.addWeighted(
            overlay,
            0.5,
            frame,
            0.5,
            0
        )

        # --------------------------------------------------------
        # 6. RADAR VIEW
        # --------------------------------------------------------
        radar_x, radar_y = 20, 20
        radar_w_display, radar_h_display = 130, 210

        radar_resized = cv2.resize(
            radar_view,
            (radar_w_display, radar_h_display)
        )

        cv2.rectangle(
            output_frame,
            (radar_x - 4, radar_y - 4),
            (
                radar_x + radar_w_display + 4,
                radar_y + radar_h_display + 4
            ),
            (255, 255, 255),
            3
        )

        car_tri = np.array([
            [radar_w_display // 2, radar_h_display - 15],
            [radar_w_display // 2 - 12, radar_h_display - 35],
            [radar_w_display // 2 + 12, radar_h_display - 35]
        ], np.int32)

        cv2.drawContours(
            radar_resized,
            [car_tri],
            0,
            (255, 0, 0),
            -1
        )

        output_frame[
            radar_y:radar_y + radar_h_display,
            radar_x:radar_x + radar_w_display
        ] = radar_resized

        # --------------------------------------------------------
        # 7. STATUS BOX
        # --------------------------------------------------------
        status_w, status_h = 280, 50

        status_x = w - status_w - 20
        status_y = 20

        try:

            sub_face = output_frame[
                status_y:status_y + status_h,
                status_x:status_x + status_w
            ].copy()

            black_rect = np.zeros(
                sub_face.shape,
                dtype=np.uint8
            )

            res = cv2.addWeighted(
                sub_face,
                0.65,
                black_rect,
                0.35,
                0
            )

            output_frame[
                status_y:status_y + status_h,
                status_x:status_x + status_w
            ] = res

        except Exception:
            pass

        cv2.rectangle(
            output_frame,
            (status_x - 3, status_y - 3),
            (status_x + status_w + 3,
             status_y + status_h + 3),
            (0, 255, 255),
            3
        )

        text = f"STATUS: {signal}"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85
        thickness = 2

        text_size = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )[0]

        text_x = status_x + (
            status_w - text_size[0]
        ) // 2

        text_y = status_y + (
            status_h + text_size[1]
        ) // 2

        cv2.putText(
            output_frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            signal_color,
            thickness
        )

        # --------------------------------------------------------
        # 8. DISPLAY FPS
        # --------------------------------------------------------
        current_time = time.time()

        dt = current_time - last_display_time

        if dt > 0:
            fps_current = 1.0 / dt
            fps_deque.append(fps_current)

        last_display_time = current_time

        fps_avg = np.mean(fps_deque) if fps_deque else 0.0

        inference_avg = (
            np.mean(inference_fps_deque)
            if inference_fps_deque
            else 0.0
        )

        cv2.putText(
            output_frame,
            f"Display FPS: {fps_avg:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_frame,
            f"YOLO FPS: {inference_avg:.1f}",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output_frame,
            f"Frame: {frame_idx} | {device_mode}",
            (10, h - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # --------------------------------------------------------
        # 9. DISPLAY
        # --------------------------------------------------------
        cv2.imshow(
            "ADAS REALTIME - Raspberry Pi",
            output_frame
        )

        # --------------------------------------------------------
        # 10. KEYBOARD
        # --------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("Exiting...")
            break

        elif key == ord("s"):

            filename = f"frame_{frame_idx}.jpg"

            cv2.imwrite(
                filename,
                output_frame
            )

            print(f"Saved: {filename}")

        # Prevent excessive CPU use if the GUI loop is too fast.
        # The camera itself is configured for 25 FPS.
        time.sleep(0.001)

        # --------------------------------------------------------
        # 11. PROGRESS
        # --------------------------------------------------------
        if frame_idx % 100 == 0:

            print(
                f"Frames: {frame_idx} | "
                f"Display FPS: {fps_avg:.1f} | "
                f"YOLO FPS: {inference_avg:.1f} | "
                f"Signal: {signal}"
            )

finally:

    stop_event.set()

    try:
        inference_thread.join(timeout=2)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()

    print("")
    print("SESSION COMPLETE!")
    print(f"Total frames processed: {frame_idx}")

    if fps_deque:
        print(f"Average display FPS: {np.mean(fps_deque):.1f}")

    if inference_fps_deque:
        print(f"Average YOLO FPS: {np.mean(inference_fps_deque):.1f}")

    print("Camera released gracefully")
except KeyboardInterrupt:
    print("\n  Interrupted by user")
except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ============================================================
    # 23. CLEANUP
    # ============================================================
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n SESSION COMPLETE!")
    print(f" Total frames processed: {frame_idx}")
    print(f" Average FPS: {np.mean(fps_deque):.1f}")
    print(f" Camera released gracefully")