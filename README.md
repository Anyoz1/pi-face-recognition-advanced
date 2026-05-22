# Edge AI Face Recognition System (ArcFace + YuNet)

This repository contains an optimized, lightweight biometric access control system (PACS / СКУД) designed to run locally on Edge devices like the **Orange Pi 5B (Rockchip RK3588)**. 

The system leverages **YuNet** for ultra-fast face detection and **ArcFace (ResNet50)** for high-accuracy feature extraction and cosine similarity matching, operating entirely offline without cloud dependencies.

---

## ⚡ Performance & Telemetry (Orange Pi 5B)

* **Face Detection (YuNet):** ~10–15 ms (on downscaled 160x120 stream)
* **Feature Extraction (ArcFace):** ~150–160 ms (on CPU, single-thread OpenCV DNN backend)
* **Average CPU Load:** ~25–30%
* **Average RAM Footprint:** ~850 MB
* **Operational Temperature:** ~53°C–56°C (Passive cooling friendly)

---

## 🛠️ Tech Stack & Dependencies

* **Runtime / Package Manager:** `uv` (Fast Python package installer and resolver)
* **Core Libraries:** `opencv-python` (DNN module), `numpy`, `psutil`
* **Models (ONNX format):**
  * Detection: `face_detection_yunet_2023mar.onnx`
  * Recognition: `w600k_r50.onnx` (ArcFace Heavy industrial model)

---

## 📁 Repository Structure

```text
.
├── dataset/              # Local biometric database (Ignored by Git for privacy)
│   └── .gitkeep
├── models/               # Pre-trained ONNX models (Ignored by Git)
│   ├── arcface.onnx
│   └── face_detection_yunet.onnx
├── face_id.py            # Main verification & access control pipeline
├── register_face.py      # Automated face enrollment utility
├── pyproject.toml        # Project metadata and dependencies managed by uv
└── uv.lock               # Locked dependency tree

    🔒 Privacy Note: The dataset/ directory contains sensitive biometric data (user photos). It is strictly blocked via .gitignore to prevent leaking personal records to public repositories.

🚀 Quick Start Guide
1. Installation & Environment Setup

Clone the repository to your single-board computer and sync the environment using uv:
Bash

cd ~/pi-face-recognition-advanced
uv sync

2. Download Pre-trained Models

If you are setting up the project on a new device, download the required weights into the models/ directory:
Bash

mkdir -p models
wget -O models/face_detection_yunet.onnx "[https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)"
wget -O models/arcface.onnx "[https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx](https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx)"

3. Enroll New Users (Face Capture)

Run the interactive script to capture a high-quality face profile directly from the USB/CSI camera. The script waits for stable face placement before saving the template:
Bash

uv run python register_face.py

Input the user's name when prompted (e.g., sultan, tilek). The file will be securely saved under dataset/{name}.jpg.
4. Launch Biometric Recognition

Start the live access control monitoring cycle:
Bash

uv run python face_id.py

The engine continuously loops, running fast detection scans. Once a face is detected, it triggers the heavy ArcFace encoder, matches vectors using Cosine Similarity (threshold > 0.38), and displays comprehensive hardware telemetry upon access authorization.
🔮 Future Architecture Roadmap

    Distributed Network: Transition to a split architecture. The Orange Pi 5B will act as an Edge compute node (camera stream processing and hardware trigger), while a dedicated centralized server (FastAPI hosted on a host machine) will manage remote database logs, batch synchronization, and administrative controls.
