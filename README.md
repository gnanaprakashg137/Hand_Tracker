# 🖐️ Hand Tracker

Real-time hand tracking using **MediaPipe** and **OpenCV** — detects hand color, finger count, movement direction, and screen position live from your webcam.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎨 **Hand Colour** | Detects skin tone / colour using HSV sampling from palm region |
| ✋ **Finger Count** | Counts raised fingers (0–5) using landmark tip vs knuckle comparison |
| 🏃 **Movement Direction** | Tracks wrist trajectory → Up / Down / Left / Right / Still |
| 📍 **Hand Position** | Divides screen into 9 zones (Top-Left, Mid-Center, Bottom-Right, etc.) |
| 🖐️ **Two Hand Support** | Detects and analyses both hands simultaneously |
| ⚡ **FPS Display** | Live frame rate shown on screen |
| 🔵 **Finger Dots** | Green dots = raised fingers, Blue dots = folded fingers |

---

## 📸 Demo

```
Camera முன்னாடி கை காட்டினால் இவை எல்லாம் live-ஆ காட்டும்:

  Hand     : Right
  Colour   : Skin
  Fingers  : 3
  Move     : Right ->
  Position : Mid-Center
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

> **Note:** Works with both MediaPipe **legacy** (`0.9.x`) and **new Task API** (`0.10+`). The script auto-detects your version.

---

## 🚀 Usage

```bash
python hand_tracker.py
```

- Show your hand in front of the camera
- All detections appear as an overlay on the live feed
- Press **`q`** to quit

---

## 📁 Project Structure

```
hand-tracker/
│
├── hand_tracker.py   # Main script
└── README.md         # This file
```

---

## ⚙️ How It Works

### 🖐️ Hand Detection
Uses **MediaPipe Hands** which detects **21 landmarks** on each hand in real time.

```
Landmark layout:
  0  = Wrist
  4  = Thumb tip
  8  = Index finger tip
  12 = Middle finger tip
  16 = Ring finger tip
  20 = Pinky tip
```

### 🎨 Colour Detection
Samples a small patch around the **palm center** in HSV colour space.
- Low saturation → White / Gray / Dark skin
- Higher saturation → Detects Red, Orange, Yellow, Green, Blue, etc.

### ✋ Finger Counting
Compares each **fingertip landmark** against its **knuckle (PIP) landmark**:
- `tip.y < pip.y` → finger is raised ✅
- Thumb uses horizontal (`x`) comparison instead of vertical

### 🏃 Movement Direction
Stores the last **14 wrist positions** in a queue.
Compares current position to oldest → calculates angle → maps to direction.

### 📍 Screen Position
Divides the frame into a **3×3 grid**:

```
Top-Left    | Top-Center    | Top-Right
Mid-Left    | Mid-Center    | Mid-Right
Bottom-Left | Bottom-Center | Bottom-Right
```

---

## 🔧 Configuration

You can tweak these values inside `hand_tracker.py`:

| Variable | Default | Description |
|---|---|---|
| `max_num_hands` | `2` | Maximum hands to detect |
| `min_detection_confidence` | `0.7` | Detection sensitivity |
| `min_tracking_confidence` | `0.6` | Tracking stability |
| `HISTORY` | `14` | Frames used for movement calculation |
| Speed threshold | `9px` | Minimum movement to register direction |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture & drawing |
| `mediapipe` | Hand landmark detection |
| `numpy` | Math / array operations |

---

## 🐛 Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
```bash
pip uninstall mediapipe -y
pip install mediapipe opencv-python numpy
```

**Camera not opening**
- Check if another app is using your webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Low FPS**
- Reduce resolution in code: change `1280` → `640`, `720` → `480`

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it in your own projects.

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google — hand landmark model
- [OpenCV](https://opencv.org/) — real-time computer vision library

---

*Built with ❤️ using Python, MediaPipe & OpenCV*
