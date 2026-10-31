# ✂️ Image & Word Cropper Suite

A high-performance image cropping and document segmentation web suite featuring both an interactive **Manual Multi-Region Cropper** and an AI-powered **Kraken Automatic Word Cropper**.

---

## 🌟 Features

### 1. 🐙 Kraken Auto Word Cropper (`autocrop.php` & `process.php`)
- **Neural Line Segmentation**: Powered by Kraken BLLA neural models to accurately identify text baselines and text boundaries.
- **Adaptive Word Detection**: Word-level boundary extraction using Otsu thresholding, connected component analysis, and adaptive gap spacing.
- **Hardware-Accelerated**: Automatically leverages NVIDIA CUDA GPUs (e.g., RTX 40-series) with multi-core CPU fallback.
- **Adaptive Resolution Scaling**: Automatically optimizes large/oversized scans (>1800 px) for neural line detection while preserving 100% full quality on final crops.
- **Interactive Review & Batch Export**: Preview all cropped words in a dark-mode grid, filter selections, adjust padding, and export as a single `.ZIP` archive.

### 2. 📐 Interactive Manual Cropper (`index.php`)
- **Multi-Region Selection**: Draw, resize, move, and duplicate multiple bounding boxes on an HTML5 canvas.
- **Multi-Format Support**: Supports JPEG, PNG, WEBP, and multi-page **PDF** documents (via PDF.js).
- **Aspect Ratio Locking**: Preset ratios (Freeform, 1:1, 4:3, 16:9, etc.).
- **Client-Side Processing**: Crops directly in the browser and bundles crops into `.ZIP` using JSZip.

---

## 🏗️ Tech Stack

- **Backend**: PHP 8.2+ (with `gd`, `fileinfo`, `zip` extensions)
- **Machine Learning / OCR**: Python 3.10+ (PyTorch, Kraken, scikit-image, Pillow, NumPy)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+), JSZip, PDF.js
- **Design**: Sleek monochrome / modern dark mode design system

---

## 📋 Prerequisites

### 1. PHP 8.2+
Ensure PHP is installed and accessible in your system `PATH`. The following PHP extensions must be enabled in `php.ini`:
```ini
extension=gd
extension=fileinfo
extension=zip
```

### 2. Python Environment (for Kraken Auto Cropper)
The Auto Cropper requires a Python virtual environment containing Kraken and PyTorch.

If you don't have one set up yet, create and install it:
```bash
# Create virtual environment
python -m venv .venv-kraken

# Activate virtual environment
# Windows PowerShell:
.\.venv-kraken\Scripts\Activate.ps1
# Windows CMD:
.\.venv-kraken\Scripts\activate.bat

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or standard PyTorch
pip install kraken scikit-image pillow numpy
```

> [!NOTE]
> `process.php` automatically looks for Python in common paths:
> - `C:/xampp/htdocs/crms-laravel-12/ml/.venv-kraken/Scripts/python.exe`
> - `./.venv-kraken/Scripts/python.exe`
> - `./.venv/Scripts/python.exe`
> - System `python`

---

## 🚀 How to Run

### Method 1: Built-in PHP Development Server (Fastest)

1. Open PowerShell or Command Prompt in the project folder:
   ```powershell
   cd c:\Users\limmi\OneDrive\Desktop\cropper
   ```
2. Start the local PHP server:
   ```powershell
   php -S localhost:8000
   ```
3. Open your browser:
   - **Kraken Word Cropper**: [http://localhost:8000/autocrop.php](http://localhost:8000/autocrop.php)
   - **Manual Canvas Cropper**: [http://localhost:8000](http://localhost:8000)

---

### Method 2: XAMPP Apache Server

1. Create a directory junction so files remain synced with your desktop:
   ```powershell
   cmd /c mklink /J "C:\xampp\htdocs\cropper" "c:\Users\limmi\OneDrive\Desktop\cropper"
   ```
2. Start **Apache** in the **XAMPP Control Panel**.
3. Access in your browser:
   - [http://localhost/cropper/autocrop.php](http://localhost/cropper/autocrop.php)
   - [http://localhost/cropper/](http://localhost/cropper/)

---

## 📂 Project Structure

```text
cropper/
├── index.php             # Interactive client-side canvas multi-region cropper
├── autocrop.php          # Kraken Auto Word Cropper upload interface
├── process.php           # Backend processing: runs Kraken, crops via GD, renders grid
├── kraken_segment.py     # Python engine for Kraken BLLA + Otsu word segmentation
├── uploads/              # Temporary storage for uploaded images
├── output/               # Storage for generated word crop slices
└── README.md             # Project documentation
```

---

## ⚡ Performance & Optimization Notes

- **CUDA Acceleration**: If an NVIDIA GPU with CUDA is detected, `kraken_segment.py` automatically uses `device="cuda"`, speeding up inference by **3.5× – 4×**.
- **CPU Fallback**: If CUDA is absent, it utilizes multi-threading (`min(8, os.cpu_count())`) to maximize CPU throughput.
- **Laptop Power**: When running on a laptop with a dedicated GPU (e.g. RTX 4050), plugging into the AC charger unlocks full GPU power and maximum scan speed.
- **Adaptive Downscaling**: High-resolution scans (>1800 px) are automatically scaled for line detection to prevent memory bloat, while word crops are extracted from the original full-resolution image to maintain crispness.

---

## 📄 License

MIT License. Free for personal and commercial use.
