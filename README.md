# Real-Time Face Detection & Vision Suite in Python using OpenCV

A real-time Computer Vision application built with Python and OpenCV featuring multi-feature facial detection, interactive visual filters, privacy anonymization, snapshot capture, and video recording.

---

## Features

### 1. Multi-Feature Facial Detection
- **Face Detection**: Fast Haar Cascade frontal face detector with custom corner reticles and individual face counter IDs (`Face #1`, `Face #2`, etc.).
- **Eye Detection (`E`)**: Real-time detection of eyes with circular reticles constrained to facial regions.
- **Smile Detection (`S`)**: Detects smiles and dynamically displays a smiling indicator (`Smiling :-)`) below the face.

### 2. Interactive Visual Modes
Switch between 4 visual modes in real-time by pressing keys `1` to `4`:
- **`[1] Box & Reticle`**: Sleek neon corner brackets and target reticle with face index tags.
- **`[2] Privacy Blur`**: Gaussian blur applied dynamically over all detected faces (anonymizes identities for streaming or sensitive demos).
- **`[3] 8-Bit Mosaic`**: Retro pixelated censor mosaic on face regions.
- **`[4] Cyberpunk Neon`**: Real-time edge detection and turbo colormap overlay for a futuristic visual style.

### 3. Media Capture & Recording
- **Instant Snapshot (`C` or `Space`)**: Captures and saves photos to `captures/capture_YYYYMMDD_HHMMSS.jpg` with an authentic shutter flash effect and on-screen toast confirmation.
- **Live Video Recording (`R`)**: Starts and stops recording MP4 video directly to `recordings/rec_YYYYMMDD_HHMMSS.mp4` with a pulsing red `REC` indicator.

### 4. Modern Heads-Up Display (HUD)
- **Real-Time FPS Counter**: Smoothed framerate tracker.
- **Live Face Counter Badge**: Visual pill displaying how many faces are in the frame.
- **Feature Status Badges**: Shows active mode, eye tracking status, and smile detection status.
- **Interactive Help Menu (`H`)**: Toggle on-screen keyboard shortcuts overlay anytime.

---

## Keyboard Controls & Shortcuts

| Key | Action | Description |
|:---:|:---|:---|
| <kbd>1</kbd> | **Reticle Box Mode** | Default neon brackets & face indexing |
| <kbd>2</kbd> | **Privacy Blur** | Blurs detected faces for privacy |
| <kbd>3</kbd> | **8-Bit Mosaic** | Pixelated censor mosaic on faces |
| <kbd>4</kbd> | **Cyberpunk Neon** | Futuristic edge & colormap overlay |
| <kbd>E</kbd> | **Toggle Eyes** | Enable/disable eye detection |
| <kbd>S</kbd> | **Toggle Smile** | Enable/disable smile detection |
| <kbd>C</kbd> / <kbd>Space</kbd> | **Take Snapshot** | Saves frame to `captures/` |
| <kbd>R</kbd> | **Record Video** | Toggles MP4 video recording in `recordings/` |
| <kbd>H</kbd> | **Toggle Help** | Shows/hides on-screen shortcut guide |
| <kbd>Q</kbd> / <kbd>Esc</kbd> | **Quit** | Clean exit and releases camera resource |

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prajakta-bhambar/Face-Detection.git
   cd Face-Detection-1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python FaceDetection.py
   ```

---

## Directory Structure
```
Face-Detection-1/
├── captures/                           # Auto-created; stores snapshot photos
├── recordings/                         # Auto-created; stores recorded MP4 videos
├── FaceDetection.py                    # Main real-time computer vision application
├── haarcascade_frontalface_default.xml # Pre-trained frontal face classifier
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```
