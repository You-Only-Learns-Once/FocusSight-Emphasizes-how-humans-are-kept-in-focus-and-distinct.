

## 🚀 Overview

This project leverages **Ultralytics YOLOv8** for **real-time segmentation**, combining both **semantic** and **instance-level understanding** of a scene.

Whether it’s detecting people, objects, or vehicles — the model applies **colored masks and labels** to visually highlight each detected object.  
A unique aspect of this project is how **humans are distinctly highlighted**, making them stand out from their environment — perfect for computer vision applications like human presence detection, safety monitoring, or visual analytics.

---

## 🧠 Key Features

- 🧩 **Semantic + Instance Segmentation** — Detect and mask multiple objects with clear class separation.  
- 🧍‍♂️ **Distinct Human Highlighting** — Visually separates humans from the surrounding background for enhanced focus.  
- ⚡ **Real-Time Webcam Inference** — Instantly segment live camera feeds.  
- 🖼️ **Image and Video Support** — Run on any image, video, or webcam stream.  
- 💾 **Optional Output Saving** — Save segmented results automatically to a custom directory.  
- 🧠 **GPU Acceleration** — Automatically detects and uses CUDA if available.  
- 🧰 **Simple Command-Line Interface** — Fully customizable with intuitive arguments.

---

## 🧾 Requirements

Make sure you have the following installed:

```bash
pip install ultralytics opencv-python torch
````

---

## ⚙️ Usage

### ▶️ Run with Webcam

```bash
python happy_new_year.py --source 0
```

### 🖼️ Run on an Image

```bash
python happy_new_year.py --source "path/to/image.jpg"
```

### 🎥 Run on a Video

```bash
python happy_new_year.py --source "path/to/video.mp4"
```

### 💾 Save the Output

```bash
python happy_new_year.py --source 0 --save --save_dir "seg_results"
```

### ⚙️ Adjust Confidence Threshold

```bash
python happy_new_year.py --source 0 --conf 0.6
```

---

## 📂 Project Structure

```
📁 Happy-New-Year/
 ├── happy_new_year.py      # Main segmentation script
 ├── seg_results/           # Output folder for saved results (auto-created)
 ├── README.md              # Project documentation
```

---

## 🧠 How It Works

1. Loads the **YOLOv8 segmentation model** (`yolov8n-seg.pt` by default).
2. Runs **real-time inference** on webcam or input files.
3. Uses **instance masks** and **bounding boxes** to identify and highlight each detected object.
4. Applies **unique coloring and labels** — especially enhancing visibility for humans to stand out clearly.
5. Optionally saves the annotated frames or videos for later analysis.

---

## 🌟 Example Use Cases

* Human detection and safety monitoring
* Visual analytics and scene understanding
* Smart surveillance
* Augmented reality overlays
* Academic or personal research in computer vision

---

## 📸 Output Preview

> Each detected object is displayed with a **color-filled mask**, bounding box, and label.
> Humans are visually emphasized, standing out distinctly from their surroundings.

---



Would you like me to include an **example output image section** (with a placeholder like `/examples/output.jpg`) so you can later add screenshots easily?
```
