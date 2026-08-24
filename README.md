# AirCanvas - Virtual Air Drawing Board

A computer vision project that lets you draw in the air using your index finger in front of a webcam. Built with Python, OpenCV, and MediaPipe.

---

## How It Works

- Raise **1 finger** (index) to draw on the screen
- Raise **2 fingers** (index + middle) to switch colors by hovering over the buttons
- Press **C** to clear the canvas
- Press **Q** to quit

---

## Features

- Real-time hand tracking using MediaPipe
- Color options: Blue, Green, Red, Eraser
- Drawings are overlaid on the live camera feed
- Hand skeleton drawn on screen for visual feedback

---

## Requirements

- Python 3.9 or higher
- A working webcam

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/AirCanvas-Computer-Vision.git
cd AirCanvas-Computer-Vision
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MediaPipe hand landmark model (done automatically on first run).

---

## Run

```bash
python air_canvas.py
```

Or open `AirCanvas.ipynb` in Jupyter Notebook or VS Code to run it step by step.

---

## Project Structure

```
AirCanvas-Computer-Vision/
├── air_canvas.py          # Main Python script
├── AirCanvas.ipynb        # Jupyter Notebook version
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Webcam capture and drawing |
| MediaPipe | Hand landmark detection |
| NumPy | Canvas array operations |

---

## Controls

| Key / Gesture | Action |
|---|---|
| Index finger up | Draw |
| Index + Middle finger up | Select color |
| C | Clear canvas |
| Q | Quit |

---

## Notes

- On first run, the script downloads the `hand_landmarker.task` model file (~8MB) automatically.
- Make sure your hand is well-lit and clearly visible to the camera for best results.
- This project uses the MediaPipe Tasks API (compatible with mediapipe 0.10.x and above).
