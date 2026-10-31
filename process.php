<?php
// ── Config ──────────────────────────────────────────────────────────────────
$uploadDir  = __DIR__ . '/uploads/';
$previewDir = __DIR__ . '/uploads/previews/';
$outputDir  = __DIR__ . '/output/';

if (!is_dir($uploadDir))  mkdir($uploadDir, 0777, true);
if (!is_dir($previewDir)) mkdir($previewDir, 0777, true);
if (!is_dir($outputDir))  mkdir($outputDir, 0777, true);

// ── Handle AJAX Save to output/ ──────────────────────────────────────────────
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'save_to_output') {
    header('Content-Type: application/json');
    $filenames = $_POST['filenames'] ?? [];
    if (is_string($filenames)) {
        $filenames = json_decode($filenames, true) ?: [$filenames];
    }
    $isAppend = !empty($_POST['append']) && $_POST['append'] === '1';

    if (!$isAppend) {
        // Clear output folder so ONLY the currently saved pictures exist in output/
        foreach (glob($outputDir . '*') as $f) {
            if (is_file($f)) unlink($f);
        }
    }

    $saved = [];
    if (is_array($filenames)) {
        foreach ($filenames as $fname) {
            $cleanName = basename($fname);
            $src = $previewDir . $cleanName;
            $dst = $outputDir . $cleanName;
            if (is_file($src)) {
                copy($src, $dst);
                $saved[] = $cleanName;
            }
        }
    }

    echo json_encode([
        'success' => true,
        'count'   => count($saved),
        'files'   => $saved
    ]);
    exit;
}

// ── Kraken venv Python executable ───────────────────────────────────────────
$pythonCandidates = [
    'C:/xampp/htdocs/crms-laravel-12/ml/.venv-kraken/Scripts/python.exe',
    'C:/laragon/www/crms-laravel-12-main/ml/.venv-kraken/Scripts/python.exe',
    __DIR__ . '/.venv-kraken/Scripts/python.exe',
    __DIR__ . '/.venv/Scripts/python.exe',
];
$krakenPy = 'python';
foreach ($pythonCandidates as $candidate) {
    if (file_exists($candidate)) {
        $krakenPy = $candidate;
        break;
    }
}
define('KRAKEN_PYTHON', $krakenPy);
define('KRAKEN_SCRIPT', __DIR__ . '/kraken_segment.py');

// Clear output directory and old previews on new upload scan
foreach (glob($outputDir . '*') as $f) {
    if (is_file($f)) unlink($f);
}
foreach (glob($previewDir . '*') as $f) {
    if (is_file($f)) unlink($f);
}

// ── Redirect if not POST ─────────────────────────────────────────────────────
if ($_SERVER['REQUEST_METHOD'] !== 'POST' || !isset($_FILES['image'])) {
    header("Location: index.php");
    exit;
}

// ── Read options ─────────────────────────────────────────────────────────────
$file = $_FILES['image'];
$padding = max(0, min(30, (int) ($_POST['padding'] ?? 4)));
$outputFormat = in_array($_POST['output_format'] ?? 'png', ['png', 'jpeg', 'webp'])
    ? $_POST['output_format'] : 'png';

// ── Validate & save upload ───────────────────────────────────────────────────
if ($file['error'] !== UPLOAD_ERR_OK) {
    renderError("Upload failed (error code {$file['error']}).");
}

$ext = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
$allowedExt = ['jpg', 'jpeg', 'png', 'webp'];
if (!in_array($ext, $allowedExt)) {
    renderError("Unsupported file type. Please upload PNG, JPG, or WEBP.");
}

$uploadFilename = 'original_' . time() . '.' . $ext;
$uploadPath = $uploadDir . $uploadFilename;
$originalWebPath = 'uploads/' . $uploadFilename;

if (!move_uploaded_file($file['tmp_name'], $uploadPath)) {
    renderError("Could not save uploaded file.");
}

// ── Run Kraken Word Segmentation Engine ──────────────────────────────────────
if (!file_exists(KRAKEN_PYTHON)) {
    renderError('Kraken Python executable not found at ' . KRAKEN_PYTHON);
}
if (!file_exists(KRAKEN_SCRIPT)) {
    renderError('kraken_segment.py not found at ' . KRAKEN_SCRIPT);
}

$pyExe = escapeshellarg(KRAKEN_PYTHON);
$pyScript = escapeshellarg(KRAKEN_SCRIPT);
$pathEsc = escapeshellarg($uploadPath);
$krakenCmd = "$pyExe $pyScript $pathEsc 2>&1";
exec($krakenCmd, $krakenOut, $krakenCode);
$krakenJson = trim(implode('', $krakenOut));

if ($krakenCode !== 0) {
    renderError('Kraken segmentation failed (exit ' . $krakenCode . '): ' . htmlspecialchars($krakenJson));
}

// Extract JSON array payload even if warnings or notices were printed
$jsonStart = strpos($krakenJson, '[');
$jsonEnd = strrpos($krakenJson, ']');
if ($jsonStart !== false && $jsonEnd !== false && $jsonEnd > $jsonStart) {
    $krakenJsonClean = substr($krakenJson, $jsonStart, $jsonEnd - $jsonStart + 1);
    $krakenBoxes = json_decode($krakenJsonClean, true);
} else {
    $krakenBoxes = json_decode($krakenJson, true);
}

if (!is_array($krakenBoxes) || count($krakenBoxes) === 0) {
    renderError('Kraken detected 0 words. Please try uploading a clearer image.');
}

// ── Load image into GD for Cropping ──────────────────────────────────────────
$mime = mime_content_type($uploadPath);
$sourceImage = match ($mime) {
    'image/jpeg' => imagecreatefromjpeg($uploadPath),
    'image/png' => imagecreatefrompng($uploadPath),
    'image/webp' => imagecreatefromwebp($uploadPath),
    default => null
};

if (!$sourceImage) {
    renderError("Could not process image file into GD.");
}

$imgWidth = imagesx($sourceImage);
$imgHeight = imagesy($sourceImage);

// ── Crop helper (saves temporary preview crops) ──────────────────────────────
function saveCrop(
    \GdImage $src,
    int $x,
    int $y,
    int $w,
    int $h,
    int $padding,
    int $imgW,
    int $imgH,
    string $previewDir,
    string $outputFormat,
    int $idx
): ?array {
    $cx = max(0, $x - $padding);
    $cy = max(0, $y - $padding);
    $cw = min($imgW - $cx, $w + $padding * 2);
    $ch = min($imgH - $cy, $h + $padding * 2);
    if ($cw <= 0 || $ch <= 0)
        return null;

    $cropRect = ['x' => $cx, 'y' => $cy, 'width' => $cw, 'height' => $ch];
    $croppedImg = imagecrop($src, $cropRect);
    if ($croppedImg === false)
        return null;

    $filename = "word_{$idx}.{$outputFormat}";
    $path = $previewDir . $filename;

    switch ($outputFormat) {
        case 'jpeg':
            imagejpeg($croppedImg, $path, 95);
            break;
        case 'webp':
            imagewebp($croppedImg, $path, 95);
            break;
        default:
            imagepng($croppedImg, $path);
            break;
    }
    imagedestroy($croppedImg);

    return [
        'file' => 'uploads/previews/' . $filename,
        'filename' => $filename,
        'w' => $cw,
        'h' => $ch,
        'orig_x' => $cx,
        'orig_y' => $cy
    ];
}

// ── Perform single-word crops ───────────────────────────────────────────────
$croppedWords = [];
$wordCount = 0;
$skipped = 0;

foreach ($krakenBoxes as $box) {
    $crop = saveCrop(
        $sourceImage,
        (int) $box['x'],
        (int) $box['y'],
        (int) $box['w'],
        (int) $box['h'],
        $padding,
        $imgWidth,
        $imgHeight,
        $previewDir,
        $outputFormat,
        $wordCount + 1
    );

    if ($crop === null) {
        $skipped++;
        continue;
    }
    $wordCount++;
    $croppedWords[] = [
        'id' => $wordCount,
        'label' => 'Word #' . $wordCount,
        'file' => $crop['file'],
        'filename' => $crop['filename'],
        'w' => $crop['w'],
        'h' => $crop['h'],
        'box' => [
            'x' => (int) $box['x'],
            'y' => (int) $box['y'],
            'w' => (int) $box['w'],
            'h' => (int) $box['h']
        ]
    ];
}

imagedestroy($sourceImage);

// ── Error Renderer ──────────────────────────────────────────────────────────
function renderError(string $msg): never
{
    ?>
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Error — Kraken Cropper</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Inter', sans-serif;
                background: #050507;
                color: #ffffff;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .card {
                background: #0d0d10;
                border: 1px solid #27272a;
                border-radius: 20px;
                padding: 40px;
                max-width: 480px;
                text-align: center;
                box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            }
            .icon { font-size: 3rem; margin-bottom: 16px; filter: grayscale(1); }
            h2 { font-size: 1.4rem; font-weight: 800; margin-bottom: 10px; color: #ffffff; }
            p { color: #8b8b96; font-size: .88rem; line-height: 1.6; margin-bottom: 24px; word-break: break-word; }
            a {
                display: inline-block;
                padding: 12px 28px;
                background: #ffffff;
                color: #000000;
                border-radius: 12px;
                font-weight: 700;
                font-size: .88rem;
                text-decoration: none;
                transition: transform .15s;
            }
            a:hover { transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">⚠️</div>
            <h2>Scan Error</h2>
            <p><?= htmlspecialchars($msg) ?></p>
            <a href="index.php">← Try Again</a>
        </div>
    </body>
    </html>
    <?php
    exit;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kraken Word Cropper — Side-by-Side Results</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <!-- JSZip for dynamic client-side batch zip creation -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg:          #050507;
            --panel:       #0d0d10;
            --card:        #121216;
            --surface:     #18181d;
            --border:      #26262b;
            --border-hover:#404048;
            --text-primary:#ffffff;
            --text-muted:  #8b8b96;
            --text-dim:    #52525b;
            --btn-bg:      #ffffff;
            --btn-text:    #000000;
            --glow:        rgba(255, 255, 255, 0.18);
            --highlight:   rgba(255, 255, 255, 0.95);
        }

        html, body {
            height: 100%;
            background: var(--bg);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }

        /* ── Top App Bar ── */
        .navbar {
            height: 60px;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: -.3px;
        }

        .brand-pill {
            padding: 3px 10px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 700;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: .5px;
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn-nav {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 14px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-size: .82rem;
            font-weight: 600;
            transition: all .2s;
        }
        .btn-nav:hover {
            color: #ffffff;
            border-color: #ffffff;
        }

        /* ── Main Split Layout (Side-by-Side) ── */
        .workspace {
            display: grid;
            grid-template-columns: 1fr 1.15fr;
            height: calc(100vh - 60px);
            overflow: hidden;
        }

        @media (max-width: 992px) {
            .workspace {
                grid-template-columns: 1fr;
                height: auto;
                overflow: visible;
            }
        }

        /* ── Left Side: Original Image Panel ── */
        .image-pane {
            border-right: 1px solid var(--border);
            background: #020203;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        .pane-header {
            padding: 14px 20px;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }

        .pane-title {
            font-size: .82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .8px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .pane-tools {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tool-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-muted);
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: .85rem;
            cursor: pointer;
            transition: all .15s;
        }
        .tool-btn:hover {
            color: #ffffff;
            border-color: #ffffff;
        }

        .image-viewport {
            flex: 1;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            background-image: 
                radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
            background-size: 20px 20px;
            user-select: none;
        }

        .image-canvas-wrap {
            position: relative;
            display: inline-block;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9), 0 0 0 1px var(--border);
            border-radius: 4px;
            transition: transform .2s ease;
            transform-origin: center center;
        }

        .image-canvas-wrap img {
            display: block;
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }

        /* ── Overlay Bounding Boxes ── */
        .bbox-overlay {
            position: absolute;
            inset: 0;
            pointer-events: auto;
        }

        .bbox {
            position: absolute;
            border: 1.5px solid rgba(255, 255, 255, 0.6);
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            cursor: pointer;
            transition: all .15s ease;
        }

        .bbox .bbox-tag {
            position: absolute;
            top: -16px;
            left: -1px;
            background: #ffffff;
            color: #000000;
            font-size: 9px;
            font-weight: 800;
            padding: 1px 4px;
            border-radius: 2px;
            line-height: 1;
            pointer-events: none;
            opacity: 0.8;
            transition: opacity .15s;
        }

        .bbox:hover,
        .bbox.active-hover {
            border-color: #ffffff;
            background: rgba(255, 255, 255, 0.28);
            box-shadow: 0 0 14px rgba(255, 255, 255, 0.9);
            z-index: 10;
        }

        .bbox:hover .bbox-tag,
        .bbox.active-hover .bbox-tag {
            opacity: 1;
        }

        .bbox.selected {
            border: 2px solid #ffffff;
            background: rgba(255, 255, 255, 0.22);
        }

        /* ── Right Side: Selected Words Panel ── */
        .words-pane {
            display: flex;
            flex-direction: column;
            background: var(--bg);
            overflow: hidden;
        }

        /* ── Action Bar / Selection Controls ── */
        .action-bar {
            padding: 16px 24px;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            flex-shrink: 0;
        }

        .selection-stat {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sel-badge {
            background: #ffffff;
            color: #000000;
            font-weight: 800;
            font-size: .82rem;
            padding: 4px 10px;
            border-radius: 8px;
        }

        .sel-text {
            font-size: .85rem;
            color: var(--text-muted);
        }

        .action-buttons {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn-action {
            padding: 8px 14px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: .82rem;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: all .15s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-action:hover {
            border-color: #ffffff;
        }

        .btn-download-selected {
            padding: 8px 18px;
            background: #ffffff;
            color: #000000;
            border: 1px solid #ffffff;
            border-radius: 8px;
            font-size: .82rem;
            font-weight: 800;
            font-family: inherit;
            cursor: pointer;
            transition: all .2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 16px rgba(255, 255, 255, 0.25);
        }

        .btn-download-selected:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 22px rgba(255, 255, 255, 0.4);
        }

        .btn-download-selected:disabled {
            opacity: .4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* ── Search & Filter Toolbar ── */
        .search-bar {
            padding: 10px 24px;
            background: var(--card);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-shrink: 0;
        }

        .search-input-wrap {
            position: relative;
            flex: 1;
            max-width: 320px;
        }

        .search-input-wrap input {
            width: 100%;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 7px 12px 7px 32px;
            color: #ffffff;
            font-size: .82rem;
            font-family: inherit;
            outline: none;
            transition: border-color .15s;
        }
        .search-input-wrap input:focus {
            border-color: #ffffff;
        }

        .search-icon {
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: .8rem;
            color: var(--text-dim);
            pointer-events: none;
        }

        .meta-counts {
            font-size: .78rem;
            color: var(--text-muted);
        }

        /* ── Words Grid / Scroll Area ── */
        .words-grid-wrap {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .words-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 14px;
        }

        /* ── Word Card (Black & White Theme) ── */
        .word-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            transition: all .15s ease;
            position: relative;
            cursor: pointer;
        }

        .word-card:hover,
        .word-card.active-hover {
            border-color: #ffffff;
            box-shadow: 0 4px 20px rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .word-card.is-selected {
            border-color: #ffffff;
            background: rgba(255, 255, 255, 0.04);
        }

        .card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Custom B&W Checkbox */
        .chk-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: .82rem;
            font-weight: 700;
            color: #ffffff;
            cursor: pointer;
        }

        .custom-chk {
            width: 18px;
            height: 18px;
            border: 1.5px solid var(--border-hover);
            border-radius: 5px;
            background: var(--surface);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all .15s;
        }

        .custom-chk::after {
            content: '✓';
            font-size: 11px;
            font-weight: 900;
            color: #000000;
            display: none;
        }

        .chk-wrap input[type="checkbox"] {
            display: none;
        }

        .chk-wrap input[type="checkbox"]:checked + .custom-chk {
            background: #ffffff;
            border-color: #ffffff;
        }

        .chk-wrap input[type="checkbox"]:checked + .custom-chk::after {
            display: block;
        }

        .dim-badge {
            font-size: .72rem;
            color: var(--text-dim);
            font-weight: 500;
        }

        .crop-preview-wrap {
            width: 100%;
            height: 80px;
            background: #000000;
            border: 1px solid var(--border);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 6px;
        }

        .crop-preview-wrap img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            display: block;
        }

        .card-bottom {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: 4px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }

        .btn-single-dl {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-muted);
            padding: 4px 8px;
            font-size: .75rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all .15s;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .btn-single-dl:hover {
            color: #ffffff;
            border-color: #ffffff;
            background: var(--surface);
        }

        .btn-zoom {
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-size: .78rem;
            cursor: pointer;
            padding: 4px;
        }
        .btn-zoom:hover { color: #ffffff; }

        /* ── Modal Lightbox ── */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(8px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }

        .modal.open { display: flex; }

        .modal-content {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            max-width: 90vw;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9);
        }

        .modal-content img {
            max-width: 100%;
            max-height: 65vh;
            object-fit: contain;
            border-radius: 8px;
            background: #000000;
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 1.4rem;
            cursor: pointer;
        }
        .modal-close:hover { color: #ffffff; }
    </style>
</head>
<body>

    <!-- Top Navigation Bar -->
    <header class="navbar">
        <div class="nav-brand">
            <span>Word Cropper</span>
            <span class="brand-pill">Kraken Only</span>
        </div>
        <div class="nav-actions">
            <a href="index.php" class="btn-nav">← New Image</a>
        </div>
    </header>

    <!-- Side-by-Side Workspace -->
    <main class="workspace">
        
        <!-- Left Side: Original Image with Interactive Bounding Boxes -->
        <section class="image-pane">
            <div class="pane-header">
                <div class="pane-title">
                    <span>Original Picture</span>
                    <span style="color:var(--text-dim); font-weight:400; font-size:.75rem;">(<?= $imgWidth ?> × <?= $imgHeight ?> px)</span>
                </div>
                <div class="pane-tools">
                    <button type="button" class="tool-btn" id="zoomOutBtn" title="Zoom Out">−</button>
                    <button type="button" class="tool-btn" id="zoomResetBtn" title="Reset Zoom">1:1</button>
                    <button type="button" class="tool-btn" id="zoomInBtn" title="Zoom In">+</button>
                </div>
            </div>

            <div class="image-viewport" id="imageViewport">
                <div class="image-canvas-wrap" id="canvasWrap">
                    <img id="originalImg" src="<?= htmlspecialchars($originalWebPath) ?>" alt="Original Image">
                    <div class="bbox-overlay" id="bboxOverlay">
                        <?php foreach ($croppedWords as $w): 
                            $bx = ($w['box']['x'] / $imgWidth) * 100;
                            $by = ($w['box']['y'] / $imgHeight) * 100;
                            $bw = ($w['box']['w'] / $imgWidth) * 100;
                            $bh = ($w['box']['h'] / $imgHeight) * 100;
                        ?>
                            <div class="bbox selected" 
                                 id="bbox-<?= $w['id'] ?>" 
                                 data-id="<?= $w['id'] ?>"
                                 style="left: <?= $bx ?>%; top: <?= $by ?>%; width: <?= $bw ?>%; height: <?= $bh ?>%;"
                                 title="Word #<?= $w['id'] ?> (Click to toggle)">
                                <span class="bbox-tag"><?= $w['id'] ?></span>
                            </div>
                        <?php endforeach; ?>
                    </div>
                </div>
            </div>
        </section>

        <!-- Right Side: Detected Words Selection & Download Panel -->
        <section class="words-pane">
            
            <!-- Selection Action Bar -->
            <div class="action-bar">
                <div class="selection-stat">
                    <span class="sel-badge" id="selectedBadge"><?= $wordCount ?></span>
                    <span class="sel-text">of <?= $wordCount ?> words selected</span>
                </div>
                <div class="action-buttons">
                    <button type="button" class="btn-action" id="selectAllBtn">Select All</button>
                    <button type="button" class="btn-action" id="deselectAllBtn">Deselect All</button>
                    <button type="button" class="btn-download-selected" id="downloadSelectedBtn">
                        <span>💾 Save Selected (<span id="btnSelCount"><?= $wordCount ?></span>)</span>
                    </button>
                </div>
            </div>

            <!-- Search / Filter Bar -->
            <div class="search-bar">
                <div class="search-input-wrap">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="searchInput" placeholder="Filter words by # (e.g. 3)...">
                </div>
                <div class="meta-counts">
                    Format: <strong><?= strtoupper($outputFormat) ?></strong> • Padding: <strong><?= $padding ?>px</strong>
                </div>
            </div>

            <!-- Words Grid Area -->
            <div class="words-grid-wrap">
                <div class="words-grid" id="wordsGrid">
                    <?php foreach ($croppedWords as $w): ?>
                        <div class="word-card is-selected" 
                             id="card-<?= $w['id'] ?>" 
                             data-id="<?= $w['id'] ?>"
                             data-file="<?= htmlspecialchars($w['file']) ?>"
                             data-filename="<?= htmlspecialchars($w['filename']) ?>">
                            
                            <div class="card-top">
                                <label class="chk-wrap" onclick="event.stopPropagation();">
                                    <input type="checkbox" class="word-chk" data-id="<?= $w['id'] ?>" checked>
                                    <span class="custom-chk"></span>
                                    <span>#<?= $w['id'] ?></span>
                                </label>
                                <span class="dim-badge"><?= $w['w'] ?>×<?= $w['h'] ?></span>
                            </div>

                            <div class="crop-preview-wrap" onclick="openModal('<?= htmlspecialchars($w['file']) ?>', 'Word #<?= $w['id'] ?> (<?= $w['w'] ?>×<?= $w['h'] ?> px)')">
                                <img src="<?= htmlspecialchars($w['file']) ?>" alt="Word #<?= $w['id'] ?>" loading="lazy">
                            </div>

                            <div class="card-bottom">
                                <button type="button" class="btn-single-dl" onclick="saveSingleWord(event, '<?= htmlspecialchars($w['filename']) ?>');">
                                    💾 Save
                                </button>
                                <button type="button" class="btn-zoom" title="Inspect" onclick="openModal('<?= htmlspecialchars($w['file']) ?>', 'Word #<?= $w['id'] ?> (<?= $w['w'] ?>×<?= $w['h'] ?> px)')">
                                    🔍 Zoom
                                </button>
                            </div>

                        </div>
                    <?php endforeach; ?>
                </div>
            </div>

        </section>

    </main>

    <!-- Preview Modal -->
    <div class="modal" id="previewModal" onclick="closeModal()">
        <div class="modal-content" onclick="event.stopPropagation();">
            <div class="modal-header">
                <h3 id="modalTitle" style="font-size:1rem; font-weight:700;">Word Preview</h3>
                <button type="button" class="modal-close" onclick="closeModal()">×</button>
            </div>
            <img id="modalImg" src="" alt="Word Zoom">
        </div>
    </div>

    <!-- Client-Side Interactive Logic -->
    <script>
        const wordsData = <?= json_encode($croppedWords) ?>;
        let selectedIds = new Set(wordsData.map(w => w.id));

        const selectedBadge = document.getElementById('selectedBadge');
        const btnSelCount = document.getElementById('btnSelCount');
        const downloadSelectedBtn = document.getElementById('downloadSelectedBtn');
        const selectAllBtn = document.getElementById('selectAllBtn');
        const deselectAllBtn = document.getElementById('deselectAllBtn');
        const searchInput = document.getElementById('searchInput');

        // Zoom controls for original image
        let currentZoom = 1;
        const canvasWrap = document.getElementById('canvasWrap');
        document.getElementById('zoomInBtn').addEventListener('click', () => {
            currentZoom = Math.min(currentZoom + 0.25, 3.0);
            canvasWrap.style.transform = `scale(${currentZoom})`;
        });
        document.getElementById('zoomOutBtn').addEventListener('click', () => {
            currentZoom = Math.max(currentZoom - 0.25, 0.5);
            canvasWrap.style.transform = `scale(${currentZoom})`;
        });
        document.getElementById('zoomResetBtn').addEventListener('click', () => {
            currentZoom = 1.0;
            canvasWrap.style.transform = `scale(1.0)`;
        });

        // Update selection UI state
        function updateSelectionState() {
            const count = selectedIds.size;
            selectedBadge.textContent = count;
            btnSelCount.textContent = count;
            downloadSelectedBtn.disabled = (count === 0);

            // Update cards and bounding boxes classes
            wordsData.forEach(w => {
                const card = document.getElementById('card-' + w.id);
                const bbox = document.getElementById('bbox-' + w.id);
                const chk = card ? card.querySelector('.word-chk') : null;

                const isSelected = selectedIds.has(w.id);
                if (card) card.classList.toggle('is-selected', isSelected);
                if (bbox) bbox.classList.toggle('selected', isSelected);
                if (chk) chk.checked = isSelected;
            });
        }

        // Toggle single word
        function toggleWord(id) {
            if (selectedIds.has(id)) {
                selectedIds.delete(id);
            } else {
                selectedIds.add(id);
            }
            updateSelectionState();
        }

        // Checkbox events
        document.querySelectorAll('.word-chk').forEach(chk => {
            chk.addEventListener('change', (e) => {
                const id = parseInt(e.target.dataset.id);
                if (e.target.checked) selectedIds.add(id);
                else selectedIds.delete(id);
                updateSelectionState();
            });
        });

        // Clicking card toggles selection
        document.querySelectorAll('.word-card').forEach(card => {
            card.addEventListener('click', () => {
                const id = parseInt(card.dataset.id);
                toggleWord(id);
            });
        });

        // Clicking bounding box on image toggles selection
        document.querySelectorAll('.bbox').forEach(bbox => {
            bbox.addEventListener('click', (e) => {
                e.stopPropagation();
                const id = parseInt(bbox.dataset.id);
                toggleWord(id);

                // Scroll right card into view
                const card = document.getElementById('card-' + id);
                if (card) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    card.classList.add('active-hover');
                    setTimeout(() => card.classList.remove('active-hover'), 1200);
                }
            });
        });

        // Synchronized Hover Effects (Image BBox <-> Card)
        document.querySelectorAll('.word-card').forEach(card => {
            const id = card.dataset.id;
            const bbox = document.getElementById('bbox-' + id);
            card.addEventListener('mouseenter', () => {
                if (bbox) bbox.classList.add('active-hover');
            });
            card.addEventListener('mouseleave', () => {
                if (bbox) bbox.classList.remove('active-hover');
            });
        });

        document.querySelectorAll('.bbox').forEach(bbox => {
            const id = bbox.dataset.id;
            const card = document.getElementById('card-' + id);
            bbox.addEventListener('mouseenter', () => {
                if (card) card.classList.add('active-hover');
            });
            bbox.addEventListener('mouseleave', () => {
                if (card) card.classList.remove('active-hover');
            });
        });

        // Select All / Deselect All
        selectAllBtn.addEventListener('click', () => {
            wordsData.forEach(w => selectedIds.add(w.id));
            updateSelectionState();
        });

        deselectAllBtn.addEventListener('click', () => {
            selectedIds.clear();
            updateSelectionState();
        });

        // Search Filter
        searchInput.addEventListener('input', (e) => {
            const q = e.target.value.trim().toLowerCase();
            document.querySelectorAll('.word-card').forEach(card => {
                const id = card.dataset.id;
                const match = !q || id.includes(q) || ('word #' + id).includes(q);
                card.style.display = match ? 'flex' : 'none';
            });
        });

        // Save Single Word into output/ folder only
        async function saveSingleWord(e, filename) {
            e.preventDefault();
            e.stopPropagation();
            const btn = e.currentTarget;
            const origHtml = btn.innerHTML;
            btn.innerHTML = '💾...';

            try {
                const formData = new FormData();
                formData.append('action', 'save_to_output');
                formData.append('filenames', JSON.stringify([filename]));
                formData.append('append', '1');

                const res = await fetch('process.php', { method: 'POST', body: formData });
                const data = await res.json();

                btn.innerHTML = '✓ Saved in output/';
                setTimeout(() => { btn.innerHTML = origHtml; }, 2000);
            } catch (err) {
                alert('Error saving picture: ' + err.message);
                btn.innerHTML = origHtml;
            }
        }

        // Batch Save Selected into output/ folder only
        downloadSelectedBtn.addEventListener('click', async () => {
            if (selectedIds.size === 0) return;

            downloadSelectedBtn.disabled = true;
            const origHtml = downloadSelectedBtn.innerHTML;
            downloadSelectedBtn.innerHTML = '<span>💾 Saving to output/...</span>';

            try {
                const selectedItems = wordsData.filter(w => selectedIds.has(w.id));
                const filenames = selectedItems.map(w => w.filename);

                // Put ONLY the saved selected pictures into output/ folder on server
                const formData = new FormData();
                formData.append('action', 'save_to_output');
                formData.append('filenames', JSON.stringify(filenames));
                formData.append('append', '0');

                const res = await fetch('process.php', { method: 'POST', body: formData });
                const data = await res.json();

                downloadSelectedBtn.innerHTML = `<span>✓ Saved (${data.count} in output/)</span>`;
                setTimeout(() => {
                    downloadSelectedBtn.disabled = false;
                    downloadSelectedBtn.innerHTML = origHtml;
                }, 2000);
            } catch (err) {
                alert('Error saving: ' + err.message);
                downloadSelectedBtn.disabled = false;
                downloadSelectedBtn.innerHTML = origHtml;
            }
        });

        // Modal Lightbox
        function openModal(src, title) {
            document.getElementById('modalImg').src = src;
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('previewModal').classList.add('open');
        }

        function closeModal() {
            document.getElementById('previewModal').classList.remove('open');
        }
    </script>
</body>
</html>