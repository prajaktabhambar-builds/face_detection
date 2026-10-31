import os
import time
from datetime import datetime
import cv2
import numpy as np

# ==========================================
# 1. SETUP DIRECTORIES & CLASSIFIERS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Load Haar Cascades with fallback to OpenCV data
face_xml = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(face_xml):
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

if face_cascade.empty():
    raise IOError(f"Failed to load Haar Cascade from: {face_xml}")


# ==========================================
# 2. HELPER & FILTER FUNCTIONS
# ==========================================
def draw_corner_rect(img, x, y, w, h, color=(0, 255, 128), thickness=2, corner_len=18):
    """Draws sleek sci-fi corner brackets around detected faces."""
    # Faint outer border
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    # Top-Left
    cv2.line(img, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(img, (x, y), (x, y + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)


def apply_blur(img, x, y, w, h):
    """Applies privacy Gaussian blur to the face ROI."""
    roi = img[y:y + h, x:x + w]
    if roi.size > 0:
        ksize = max(51, (w // 3) | 1)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
        img[y:y + h, x:x + w] = blurred


def apply_pixelate(img, x, y, w, h, blocks=12):
    """Applies retro 8-bit mosaic / pixelation censor to the face ROI."""
    roi = img[y:y + h, x:x + w]
    if roi.size > 0:
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y:y + h, x:x + w] = pixelated


def draw_hud_badge(img, text, x, y, bg_color=(35, 35, 35), text_color=(255, 255, 255), scale=0.45):
    """Draws a styled pill badge for stats and indicators."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    pad_x, pad_y = 8, 4
    cv2.rectangle(img, (x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), bg_color, -1)
    cv2.rectangle(img, (x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (80, 80, 80), 1)
    cv2.putText(img, text, (x + pad_x, y), font, scale, text_color, 1, cv2.LINE_AA)
    return tw + pad_x * 2 + 6


# ==========================================
# 3. CAMERA INITIALIZATION
# ==========================================
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not access the webcam (index 0). Please check camera connections or permissions.")
    exit(1)

# Sensor warm-up
for _ in range(5):
    cam.read()


# ==========================================
# 4. STATE VARIABLES
# ==========================================
MODES = {
    1: "Box & Reticle",
    2: "Privacy Blur",
    3: "8-Bit Mosaic",
    4: "Cyberpunk Neon"
}
current_mode = 1
detect_eyes = False
detect_smile = False
show_help = False

# Video recording
is_recording = False
video_writer = None
rec_filename = None

# Notification toast
toast_message = "Press 'H' for shortcut help"
toast_until = time.time() + 3.5

# Flash effect on snapshot
flash_timer = 0

# FPS calculation
fps_list = []
prev_time = time.time()

print("Face Detection Enhanced Suite Running!")
print("Controls: [1-4] Modes | [E] Eyes | [S] Smile | [C/Space] Snapshot | [R] Record | [H] Help | [Q/ESC] Quit")


# ==========================================
# 5. MAIN REAL-TIME LOOP
# ==========================================
while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Warning: Failed to grab frame. Exiting...")
        break

    # FPS update
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time
    fps = 1.0 / dt if dt > 0 else 30.0
    fps_list.append(fps)
    if len(fps_list) > 15:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)

    # Process grayscale for Haar cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional Cyberpunk Neon mode base
    if current_mode == 4:
        edges = cv2.Canny(gray, 60, 150)
        edges_color = cv2.applyColorMap(edges, cv2.COLORMAP_TURBO)
        display_frame = cv2.addWeighted(frame, 0.4, edges_color, 0.6, 0)
    else:
        display_frame = frame.copy()

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Process each detected face
    for idx, (x, y, w, h) in enumerate(faces, start=1):
        # 1. Apply selected mode filter
        if current_mode == 1 or current_mode == 4:
            draw_corner_rect(display_frame, x, y, w, h, color=(0, 255, 128), thickness=2)
            cv2.putText(display_frame, f"Face #{idx}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
        elif current_mode == 2:
            apply_blur(display_frame, x, y, w, h)
            draw_corner_rect(display_frame, x, y, w, h, color=(255, 180, 0), thickness=2)
            cv2.putText(display_frame, f"Blurred #{idx}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)
        elif current_mode == 3:
            apply_pixelate(display_frame, x, y, w, h, blocks=12)
            draw_corner_rect(display_frame, x, y, w, h, color=(0, 200, 255), thickness=2)
            cv2.putText(display_frame, f"Mosaic #{idx}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        face_gray = gray[y:y + h, x:x + w]
        face_color = display_frame[y:y + h, x:x + w]

        # 2. Eye Detection
        if detect_eyes:
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.15, minNeighbors=6, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                # Restrict to upper 60% of face to avoid false eye detections on lips/chin
                if ey + eh // 2 < int(h * 0.65):
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    radius = int(round((ew + eh) * 0.28))
                    cv2.circle(display_frame, center, radius, (255, 255, 0), 2, cv2.LINE_AA)

        # 3. Smile Detection
        if detect_smile:
            # Look for smile in lower 60% of the face
            lower_face_gray = face_gray[int(h * 0.4):h, :]
            smiles = smile_cascade.detectMultiScale(
                lower_face_gray,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25)
            )
            if len(smiles) > 0:
                cv2.putText(display_frame, "Smiling :-)", (x + w - 85, y + h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # ==========================================
    # 6. HEADS-UP DISPLAY (HUD)
    # ==========================================
    # Semi-transparent top bar
    h_frame, w_frame = display_frame.shape[:2]
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0, display_frame)

    # HUD Elements
    cur_x = 10
    face_count = len(faces)
    face_badge_color = (0, 140, 60) if face_count > 0 else (60, 60, 60)
    cur_x += draw_hud_badge(display_frame, f"Faces: {face_count}", cur_x, 24, bg_color=face_badge_color)
    cur_x += draw_hud_badge(display_frame, f"FPS: {avg_fps:.0f}", cur_x, 24)
    cur_x += draw_hud_badge(display_frame, f"Mode: {MODES[current_mode]}", cur_x, 24, bg_color=(80, 50, 0))

    if detect_eyes:
        cur_x += draw_hud_badge(display_frame, "Eyes: ON", cur_x, 24, bg_color=(100, 100, 0))
    if detect_smile:
        cur_x += draw_hud_badge(display_frame, "Smile: ON", cur_x, 24, bg_color=(0, 120, 120))

    # Recording Indicator
    if is_recording:
        # Pulsing red circle
        pulse = int((time.time() * 2) % 2)
        rec_color = (0, 0, 255) if pulse == 0 else (0, 0, 160)
        cv2.circle(display_frame, (w_frame - 80, 18), 7, rec_color, -1)
        cv2.putText(display_frame, "REC", (w_frame - 65, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Help Overlay Toggle
    if show_help:
        help_bg = display_frame.copy()
        box_x, box_y, box_w, box_h = 20, 50, 360, 230
        cv2.rectangle(help_bg, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)
        cv2.addWeighted(help_bg, 0.85, display_frame, 0.15, 0, display_frame)
        cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 128), 1)

        help_lines = [
            "--- CONTROLS & SHORTCUTS ---",
            "[1] Reticle Box Mode",
            "[2] Privacy Blur Mode",
            "[3] 8-Bit Pixelate Mosaic",
            "[4] Cyberpunk Neon Filter",
            "[E] Toggle Eye Detection",
            "[S] Toggle Smile Detection",
            "[C / Space] Save Snapshot",
            "[R] Start / Stop Recording",
            "[H] Close Help Menu",
            "[Q / Esc] Quit Application"
        ]
        for i, line in enumerate(help_lines):
            color = (0, 255, 128) if i == 0 else (220, 220, 220)
            weight = 2 if i == 0 else 1
            cv2.putText(display_frame, line, (box_x + 15, box_y + 24 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, weight, cv2.LINE_AA)

    # Toast Notifications
    if time.time() < toast_until and toast_message:
        draw_hud_badge(display_frame, toast_message, 15, h_frame - 15,
                       bg_color=(20, 80, 20), text_color=(255, 255, 255), scale=0.5)

    # Camera Flash Effect on Snapshot
    if flash_timer > 0:
        white = np.full_like(display_frame, 255)
        alpha = flash_timer / 4.0
        cv2.addWeighted(white, alpha * 0.6, display_frame, 1.0 - alpha * 0.6, 0, display_frame)
        flash_timer -= 1

    # Write video frame if recording
    if is_recording and video_writer is not None:
        video_writer.write(display_frame)

    cv2.imshow("FaceDetect", display_frame)

    # Check if window was closed via 'X' button
    if cv2.getWindowProperty("FaceDetect", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF

    # Exit
    if key == ord('q') or key == ord('Q') or key == 27:
        break
    # Modes 1-4
    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
        current_mode = int(chr(key))
        toast_message = f"Mode: {MODES[current_mode]}"
        toast_until = time.time() + 2.0
    # Eye detection toggle
    elif key == ord('e') or key == ord('E'):
        detect_eyes = not detect_eyes
        toast_message = f"Eye Detection: {'ON' if detect_eyes else 'OFF'}"
        toast_until = time.time() + 2.0
    # Smile detection toggle
    elif key == ord('s') or key == ord('S'):
        detect_smile = not detect_smile
        toast_message = f"Smile Detection: {'ON' if detect_smile else 'OFF'}"
        toast_until = time.time() + 2.0
    # Help toggle
    elif key == ord('h') or key == ord('H'):
        show_help = not show_help
    # Snapshot (C or Space)
    elif key == ord('c') or key == ord('C') or key == 32:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)
        cv2.imwrite(filepath, display_frame)
        flash_timer = 4
        toast_message = f"Saved: captures/{filename}"
        toast_until = time.time() + 3.0
        print(f"Snapshot saved to: {filepath}")
    # Video Recording toggle (R)
    elif key == ord('r') or key == ord('R'):
        if not is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_filename = f"rec_{timestamp}.mp4"
            rec_path = os.path.join(RECORDINGS_DIR, rec_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(rec_path, fourcc, 20.0, (w_frame, h_frame))
            is_recording = True
            toast_message = f"Recording started: {rec_filename}"
            toast_until = time.time() + 2.5
            print(f"Recording started: {rec_path}")
        else:
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            toast_message = f"Recording saved to recordings/"
            toast_until = time.time() + 2.5
            print("Recording stopped and saved.")

# Cleanup
if video_writer is not None:
    video_writer.release()
cam.release()
cv2.destroyAllWindows()
print("Face Detection terminated cleanly.")
