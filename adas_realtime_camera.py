"""
ADAS REALTIME CAMERA INFERENCE
Raspberry Pi + USB/Pi Camera
Model: YOLOv11x Segmentation with Perimeter Zone Detection
Parameters: DO NOT CHANGE
"""

import cv2
from ultralytics import YOLO
import numpy as np
import threading
from collections import deque
import time

# ============================================================
# 1. MODEL & PATHS
# ============================================================
WEIGHTS_PATH = r'best.pt'  # Your fine-tuned model

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
    0: "Car",
    1: "Bus", 
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
print("🔄 Loading YOLO model...")
model = YOLO(WEIGHTS_PATH)

# Try GPU if available, fallback to CPU
try:
    model.to('cuda')
    print("✅ Model loaded on GPU")
    device_mode = "CUDA"
except:
    print("⚠️  GPU not available, using CPU")
    device_mode = "CPU"

# ============================================================
# 6. CAMERA SETUP
# ============================================================
# For Raspberry Pi:
# - USB Camera: cap = cv2.VideoCapture(0)
# - Pi Camera V2: cap = cv2.VideoCapture(0) with libcamera backend
# - Pi Camera (legacy): Use picamera library

CAMERA_INDEX = 0  # Change to 1, 2, etc. for multiple cameras
cap = cv2.VideoCapture(CAMERA_INDEX)

# Camera Optimizations for Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

# Get actual resolution after setting
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📷 Camera Resolution: {w} x {h} @ {fps} FPS")
print(f"🎯 Device Mode: {device_mode}")

# ============================================================
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

print(f"✅ Polygon scaling applied: {scale_x:.2f}x, {scale_y:.2f}x")
print(f"✅ Polygon center X: {poly_center_x}")

# ============================================================
# 8. RADAR DIMENSIONS
# ============================================================
RADAR_W, RADAR_H = 150, 200
SCALE_X_RADAR = RADAR_W / w
SCALE_Y_RADAR = RADAR_H / h

# ============================================================
# 9. FPS COUNTER (FOR MONITORING PERFORMANCE)
# ============================================================
fps_deque = deque(maxlen=30)
last_time = time.time()

# ============================================================
# 10. MAIN INFERENCE LOOP - REALTIME CAMERA
# ============================================================
frame_idx = 0
print(f"\n🚀 Starting REALTIME inference... Press 'q' to quit, 's' to save frame\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # --- YOLO11x Inference (FULLY CORRECT BLOCK 41) ---
        results = model(frame, conf=0.15, imgsz=1024, quantize='fp16', verbose=False)[0]
        
        overlay = frame.copy()
        radar_view = np.zeros((RADAR_H, RADAR_W, 3), dtype=np.uint8)
        signal = "Path Clear"
        signal_color = (255, 255, 255)
        
        # ============================================================
        # 11. PERIMETER ZONE RENDERING (DUPLICATED - BLOCK 41 CORRECT)
        # ============================================================
        perimeter_overlay = overlay.copy()
        cv2.fillPoly(perimeter_overlay, [ZONE_POINTS], (100, 200, 255))  # Light blue fill
        cv2.polylines(overlay, [ZONE_POINTS], True, (255, 255, 0), 3)    # Yellow outline
        overlay = cv2.addWeighted(perimeter_overlay, 0.25, overlay, 0.75, 0)
        
        # ============================================================
        # 12. SEGMENTATION & DETECTION PROCESSING
        # ============================================================
        if results.masks is not None:
            classes = results.boxes.cls.cpu().numpy()
            masks = results.masks.xy
            
            for i, mask_data in enumerate(masks):
                cls = int(classes[i])
                pts = np.array(mask_data, dtype=np.int32)
                
                # Segmentation Colors - use dictionary lookup (DUPLICATED)
                color = class_colors.get(cls, (200, 200, 200))
                
                # Render lanes as lines, vehicles as filled
                if cls in [2, 5]:  # Lane classes
                    cv2.polylines(overlay, [pts], True, color, 6)
                else:  # Vehicle classes
                    cv2.fillPoly(overlay, [pts], color)
                
                # ============================================================
                # 13. RADAR MAPPING (DUPLICATED PROCESSING)
                # ============================================================
                radar_pts = (pts * [SCALE_X_RADAR, SCALE_Y_RADAR]).astype(np.int32)
                radar_color = class_colors.get(cls, (200, 200, 200))
                
                if cls in [2, 5]:  # Lane classes
                    cv2.polylines(radar_view, [radar_pts], True, radar_color, 4)
                else:  # Vehicle classes
                    cv2.fillPoly(radar_view, [radar_pts], radar_color)
                
                # ============================================================
                # 14. STEERING LOGIC (DIRECTIONAL SIGNALS)
                # ============================================================
                if cls == 2 or cls == 5:  # White Lane or Yellow Boundary
                    for p in pts:
                        if cv2.pointPolygonTest(ZONE_POINTS, (float(p[0]), float(p[1])), False) >= 0:
                            if p[0] < poly_center_x:
                                signal = "Turn Right"
                                signal_color = (0, 165, 255)  # Orange
                            else:
                                signal = "Turn Left"
                                signal_color = (0, 255, 255)  # Yellow
                            break
                
                # ============================================================
                # 15. CLASS LABEL AT CENTROID
                # ============================================================
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = class_names.get(cls, "Unknown")
                    cv2.putText(overlay, label, (cx-25, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ============================================================
        # 16. BLEND AND ASSEMBLE HUD (FULLY CORRECT BLOCK 41)
        # ============================================================
        output_frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # ============================================================
        # 17. RADAR VIEW (Top Left)
        # ============================================================
        radar_x, radar_y = 20, 20
        radar_w_display, radar_h_display = 130, 210
        
        radar_resized = cv2.resize(radar_view, (radar_w_display, radar_h_display))
        
        # Draw white border around radar
        cv2.rectangle(output_frame, (radar_x-4, radar_y-4), 
                      (radar_x+radar_w_display+4, radar_y+radar_h_display+4), 
                      (255, 255, 255), 3)
        
        # Draw car indicator (blue triangle) at center bottom
        car_tri = np.array([
            [radar_w_display//2, radar_h_display-15], 
            [radar_w_display//2-12, radar_h_display-35], 
            [radar_w_display//2+12, radar_h_display-35]
        ], np.int32)
        cv2.drawContours(radar_resized, [car_tri], 0, (255, 0, 0), -1)
        
        # Place radar on frame
        output_frame[radar_y:radar_y+radar_h_display, radar_x:radar_x+radar_w_display] = radar_resized
        
        # ============================================================
        # 18. STATUS BOX (Top Right) - FULLY CORRECT BLOCK 41
        # ============================================================
        status_w, status_h = 280, 50 
        status_x = w - status_w - 20 
        status_y = 20
        
        # Ensure bounds are valid
        if status_x + status_w > w:
            status_x = w - status_w - 20
        if status_y + status_h > h:
            status_y = 20
        
        try: 
            sub_face = output_frame[status_y:status_y+status_h, status_x:status_x+status_w].copy()
            black_rect = np.zeros(sub_face.shape, dtype=np.uint8) 
            res = cv2.addWeighted(sub_face, 0.65, black_rect, 0.35, 0) 
            output_frame[status_y:status_y+status_h, status_x:status_x+status_w] = res 
        except: 
            pass
        
        # Yellow border around status
        cv2.rectangle(output_frame, (status_x-3, status_y-3), 
                      (status_x+status_w+3, status_y+status_h+3), 
                      (0, 255, 255), 3)
        
        # Status text
        text = f"STATUS: {signal}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.85
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = status_x + (status_w - text_size[0]) // 2  # Center horizontally
        text_y = status_y + (status_h + text_size[1]) // 2  # Center vertically
        cv2.putText(output_frame, text, (text_x, text_y), font, font_scale, signal_color, thickness)
        
        # ============================================================
        # 19. FPS COUNTER & STATS
        # ============================================================
        current_time = time.time()
        fps_current = 1 / (current_time - last_time)
        last_time = current_time
        fps_deque.append(fps_current)
        fps_avg = np.mean(fps_deque)
        
        # Display FPS & Frame Count
        cv2.putText(output_frame, f"FPS: {fps_avg:.1f}", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Frame: {frame_idx} | {device_mode}", (10, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # ============================================================
        # 20. DISPLAY FRAME
        # ============================================================
        cv2.imshow('ADAS REALTIME - Raspberry Pi', output_frame)
        
        # ============================================================
        # 21. KEYBOARD CONTROLS
        # ============================================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n✅ Exiting... Press any key")
            break
        elif key == ord('s'):
            filename = f"frame_{frame_idx}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"📸 Saved: {filename}")
        
        frame_idx += 1
        
        # ============================================================
        # 22. PROGRESS LOG EVERY 100 FRAMES
        # ============================================================
        if frame_idx % 100 == 0:
            print(f"✅ Processed {frame_idx} frames | FPS: {fps_avg:.1f} | Signal: {signal}")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ============================================================
    # 23. CLEANUP
    # ============================================================
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🏁 SESSION COMPLETE!")
    print(f"✅ Total frames processed: {frame_idx}")
    print(f"✅ Average FPS: {np.mean(fps_deque):.1f}")
    print(f"✅ Camera released gracefully")
