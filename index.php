<?php
// ── HTR Dataset Preparation (paths) ─────────────────────────────────────────
$outputDir   = __DIR__ . '/output/';          // word crops saved from the cropper UI
$htrDir      = __DIR__ . '/htr_dataset/';     // batch-processed, HTR-ready dataset
$htrImageDir = $htrDir . 'images/';
$htrExcelRel = 'htr_dataset/dataset_labels.xlsx';
$htrExcelAbs = __DIR__ . '/' . $htrExcelRel;

if (!is_dir($outputDir))   mkdir($outputDir, 0777, true);
if (!is_dir($htrImageDir)) mkdir($htrImageDir, 0777, true);

/**
 * Normalizes a single saved word-crop so it's ready to feed into an HTR
 * model: grayscale, auto-trimmed to its ink bounding box, contrast-stretched,
 * then resized to a fixed height (preserving aspect ratio) with a small
 * clean margin — the standard input shape used by CRNN/Transformer HTR
 * architectures (e.g. IAM-style word strips).
 */
function prepareImageForHTR(\GdImage $src, int $targetHeight = 64, int $margin = 6): ?\GdImage
{
    imagefilter($src, IMG_FILTER_GRAYSCALE);

    $w = imagesx($src);
    $h = imagesy($src);
    if ($w < 2 || $h < 2) {
        return null;
    }

    // Auto-trim: find the bounding box of "ink" pixels (darker than threshold)
    $threshold = 200; // 0-255; pixels darker than this count as ink
    $minX = $w; $minY = $h; $maxX = 0; $maxY = 0;
    $found = false;

    for ($y = 0; $y < $h; $y++) {
        for ($x = 0; $x < $w; $x++) {
            $gray = imagecolorat($src, $x, $y) & 0xFF; // R=G=B after grayscale filter
            if ($gray < $threshold) {
                $found = true;
                if ($x < $minX) $minX = $x;
                if ($x > $maxX) $maxX = $x;
                if ($y < $minY) $minY = $y;
                if ($y > $maxY) $maxY = $y;
            }
        }
    }

    if (!$found) {
        $minX = 0; $minY = 0; $maxX = $w - 1; $maxY = $h - 1;
    }

    $cropW = $maxX - $minX + 1;
    $cropH = $maxY - $minY + 1;

    $trimmed = imagecreatetruecolor($cropW, $cropH);
    $white = imagecolorallocate($trimmed, 255, 255, 255);
    imagefill($trimmed, 0, 0, $white);
    imagecopy($trimmed, $src, 0, 0, $minX, $minY, $cropW, $cropH);

    // Stretch contrast so ink reads darker and background reads brighter,
    // which keeps crops from many different scans/lighting consistent.
    imagefilter($trimmed, IMG_FILTER_CONTRAST, -20);

    // Resize to a fixed height, preserving aspect ratio (variable width is
    // expected/required by sequence-based HTR models).
    $scale = $targetHeight / $cropH;
    $newW = max(1, (int) round($cropW * $scale));
    $newH = $targetHeight;

    $resized = imagecreatetruecolor($newW, $newH);
    $whiteR = imagecolorallocate($resized, 255, 255, 255);
    imagefill($resized, 0, 0, $whiteR);
    imagecopyresampled($resized, $trimmed, 0, 0, 0, 0, $newW, $newH, $cropW, $cropH);
    imagedestroy($trimmed);

    // Small white margin on every side (standard practice for CTC/CRNN HTR
    // inputs, avoids clipping ascenders/descenders at the image edge).
    $finalW = $newW + $margin * 2;
    $finalH = $newH + $margin * 2;
    $final = imagecreatetruecolor($finalW, $finalH);
    $whiteF = imagecolorallocate($final, 255, 255, 255);
    imagefill($final, 0, 0, $whiteF);
    imagecopy($final, $resized, $margin, $margin, 0, 0, $newW, $newH);
    imagedestroy($resized);

    return $final;
}

/**
 * Batch-processes every saved word image in output/ into HTR-ready training
 * data (see prepareImageForHTR) saved under htr_dataset/images/, then writes
 * an Excel (.xlsx) manifest mapping each processed image to a blank
 * "transcription" column ready for human labeling.
 *
 * @return array{success:bool, processed:int, skipped:int, manifest:array}
 */
function batchProcessForHTR(string $outputDir, string $htrImageDir, string $htrExcelAbs, int $targetHeight = 64): array
{
    $allowedExt = ['png', 'jpg', 'jpeg', 'webp'];
    $files = glob($outputDir . '*') ?: [];
    sort($files);

    $manifest = [];
    $processed = 0;
    $skipped = 0;
    $index = 0;

    foreach ($files as $path) {
        if (!is_file($path)) {
            continue;
        }
        $ext = strtolower(pathinfo($path, PATHINFO_EXTENSION));
        if (!in_array($ext, $allowedExt, true)) {
            $skipped++;
            continue;
        }

        $mime = mime_content_type($path);
        $src = match ($mime) {
            'image/jpeg' => imagecreatefromjpeg($path),
            'image/png'  => imagecreatefrompng($path),
            'image/webp' => imagecreatefromwebp($path),
            default      => null,
        };
        if (!$src) {
            $skipped++;
            continue;
        }

        $prepared = prepareImageForHTR($src, $targetHeight);
        imagedestroy($src);
        if (!$prepared) {
            $skipped++;
            continue;
        }

        $index++;
        $newFilename = sprintf('htr_%04d.png', $index);
        imagepng($prepared, $htrImageDir . $newFilename, 6);
        $finalW = imagesx($prepared);
        $finalH = imagesy($prepared);
        imagedestroy($prepared);

        $manifest[] = [
            'id'                 => $index,
            'original_filename'  => basename($path),
            'processed_filename' => $newFilename,
            'relative_path'      => 'htr_dataset/images/' . $newFilename,
            'width'              => $finalW,
            'height'             => $finalH,
            'transcription'      => '', // fill in ground-truth text before training
        ];
        $processed++;
    }

    writeXlsx(
        $htrExcelAbs,
        'HTR Dataset',
        ['ID', 'Image Filename', 'Relative Path', 'Width', 'Height', 'Transcription'],
        array_map(
            fn($r) => [$r['id'], $r['processed_filename'], $r['relative_path'], $r['width'], $r['height'], $r['transcription']],
            $manifest
        )
    );

    return [
        'success'   => true,
        'processed' => $processed,
        'skipped'   => $skipped,
        'manifest'  => $manifest,
    ];
}

/**
 * Writes a minimal single-sheet Excel (.xlsx) file using a hand-rolled ZIP
 * container (see createMinimalZip) so it works even when the ext-zip
 * extension is disabled — no Composer/PhpSpreadsheet dependency required.
 * Cell values are written as inline strings so no shared-strings part is
 * needed, keeping the workbook small and dependency-free.
 */
function writeXlsx(string $path, string $sheet, array $headers, array $rows): bool
{
    $sheetXml = xlsxBuildSheetXml(array_merge([$headers], $rows));
    $sheetNameEsc = htmlspecialchars($sheet, ENT_XML1 | ENT_QUOTES, 'UTF-8');

    $contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' .
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' .
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' .
        '<Default Extension="xml" ContentType="application/xml"/>' .
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' .
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' .
        '</Types>';

    $rootRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' .
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' .
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>' .
        '</Relationships>';

    $workbook = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' .
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ' .
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' .
        '<sheets><sheet name="' . $sheetNameEsc . '" sheetId="1" r:id="rId1"/></sheets>' .
        '</workbook>';

    $workbookRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' .
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' .
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>' .
        '</Relationships>';

    return createMinimalZip($path, [
        '[Content_Types].xml'        => $contentTypes,
        '_rels/.rels'                => $rootRels,
        'xl/workbook.xml'            => $workbook,
        'xl/_rels/workbook.xml.rels' => $workbookRels,
        'xl/worksheets/sheet1.xml'   => $sheetXml,
    ]);
}

/**
 * Creates a ZIP archive from an in-memory map of entries without requiring
 * the ext-zip extension — only PHP's always-available zlib functions
 * (gzdeflate) and crc32() are used. This keeps Excel export working even on
 * PHP installs where ext-zip is disabled (a common default on XAMPP/Laragon).
 *
 * @param array<string,string> $files Map of in-archive path => raw file contents
 */
function createMinimalZip(string $zipPath, array $files): bool
{
    $dir = dirname($zipPath);
    if (!is_dir($dir)) {
        mkdir($dir, 0777, true);
    }

    $localParts = [];
    $centralParts = [];
    $offset = 0;
    $dosTime = 0;
    $dosDate = (1 << 5) | 1; // Jan 1, 1980 — timestamp is irrelevant for this manifest

    foreach ($files as $name => $contents) {
        $nameEnc = str_replace('\\', '/', $name);
        $crc = crc32($contents);
        $uncompressedSize = strlen($contents);
        $compressed = gzdeflate($contents, 6);
        $compressedSize = strlen($compressed);

        $localHeader = pack(
            'VvvvvvVVVvv',
            0x04034b50,
            20,
            0,
            8,
            $dosTime,
            $dosDate,
            $crc,
            $compressedSize,
            $uncompressedSize,
            strlen($nameEnc),
            0
        ) . $nameEnc . $compressed;

        $localParts[] = $localHeader;

        $centralParts[] = pack(
            'VvvvvvvVVVvvvvvVV',
            0x02014b50,
            20,
            20,
            0,
            8,
            $dosTime,
            $dosDate,
            $crc,
            $compressedSize,
            $uncompressedSize,
            strlen($nameEnc),
            0,
            0,
            0,
            0,
            0x81A40000,
            $offset
        ) . $nameEnc;

        $offset += strlen($localHeader);
    }

    $localData = implode('', $localParts);
    $centralData = implode('', $centralParts);

    $endRecord = pack(
        'VvvvvVVv',
        0x06054b50,
        0,
        0,
        count($files),
        count($files),
        strlen($centralData),
        strlen($localData),
        0
    );

    if (file_exists($zipPath)) {
        @unlink($zipPath);
    }

    return file_put_contents($zipPath, $localData . $centralData . $endRecord) !== false;
}

/** Builds <sheetData> XML for a 2D array of rows using inline strings. */
function xlsxBuildSheetXml(array $rows): string
{
    $xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' .
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">' .
        '<sheetData>';

    foreach ($rows as $rowIdx => $row) {
        $r = $rowIdx + 1;
        $xml .= '<row r="' . $r . '">';
        foreach (array_values($row) as $colIdx => $value) {
            $cellRef = xlsxColumnLetter($colIdx) . $r;
            if (is_numeric($value) && $value !== '') {
                $xml .= '<c r="' . $cellRef . '"><v>' . htmlspecialchars((string) $value, ENT_XML1) . '</v></c>';
            } else {
                $escaped = htmlspecialchars((string) $value, ENT_XML1 | ENT_QUOTES, 'UTF-8');
                $xml .= '<c r="' . $cellRef . '" t="inlineStr"><is><t xml:space="preserve">' . $escaped . '</t></is></c>';
            }
        }
        $xml .= '</row>';
    }

    $xml .= '</sheetData></worksheet>';
    return $xml;
}

/** Converts a 0-based column index to an Excel column letter (0 -> A, 26 -> AA, ...). */
function xlsxColumnLetter(int $index): string
{
    $letter = '';
    $index++;
    while ($index > 0) {
        $mod = ($index - 1) % 26;
        $letter = chr(65 + $mod) . $letter;
        $index = (int) (($index - $mod) / 26);
    }
    return $letter;
}

// ── AJAX: run the HTR batch-processing job and return a JSON summary ───────
if ($_SERVER['REQUEST_METHOD'] === 'POST' && ($_POST['action'] ?? '') === 'batch_process_htr') {
    header('Content-Type: application/json');
    $result = batchProcessForHTR($outputDir, $htrImageDir, $htrExcelAbs);
    echo json_encode([
        'success'    => $result['success'],
        'processed'  => $result['processed'],
        'skipped'    => $result['skipped'],
        'excel_file' => $htrExcelRel,
        'images_dir' => 'htr_dataset/images/',
    ]);
    exit;
}
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kraken Word Cropper</title>
    <meta name="description"
        content="Automatic single-word scanner and cropper powered by Kraken BLLA neural segmentation.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap"
        rel="stylesheet">
    <style>
        *,
        *::before,
        *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg: #050507;
            --kc-card: #0d0d10;
            --kc-surface: #141418;
            --kc-border: #26262b;
            --kc-muted: #8b8b96;
            --kc-text: #ffffff;
            --kc-btn-bg: #ffffff;
            --kc-btn-text: #000000;
            --glow: rgba(255, 255, 255, 0.18);
        }

        html,
        body {
            min-height: 100vh;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--kc-text);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 48px 20px 80px;
            background-image:
                radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.05) 0%, transparent 60%),
                linear-gradient(to bottom, #070709, #030304);
            background-attachment: fixed;
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .engine-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: var(--kc-surface);
            border: 1px solid var(--kc-border);
            border-radius: 999px;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .8px;
            text-transform: uppercase;
            color: var(--kc-text);
            margin-bottom: 16px;
        }

        .engine-badge .pulse {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #ffffff;
            box-shadow: 0 0 8px #ffffff;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {

            0%,
            100% {
                opacity: 1;
                transform: scale(1);
            }

            50% {
                opacity: .4;
                transform: scale(.85);
            }
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -1px;
            color: #ffffff;
            line-height: 1.15;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--kc-muted);
            font-size: .92rem;
            max-width: 480px;
            line-height: 1.5;
            margin: 0 auto;
        }

        .kc {
            width: 100%;
            max-width: 580px;
        }

        .kc-form {
            background: var(--kc-card);
            border: 1px solid var(--kc-border);
            border-radius: 24px;
            padding: 36px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.03);
        }

        .kc-drop {
            border: 2px dashed var(--kc-border);
            border-radius: 18px;
            padding: 38px 24px;
            text-align: center;
            cursor: pointer;
            background: var(--kc-surface);
            position: relative;
            transition: all .25s ease;
        }

        .kc-drop:hover,
        .kc-drop.dragover {
            border-color: #ffffff;
            background: rgba(255, 255, 255, .04);
            box-shadow: 0 0 24px var(--glow);
        }

        .kc-drop input[type=file] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
            width: 100%;
            height: 100%;
        }

        .kc-drop-icon {
            font-size: 2.4rem;
            margin-bottom: 12px;
            filter: grayscale(1);
        }

        .kc-drop-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 4px;
        }

        .kc-drop-sub {
            font-size: .8rem;
            color: var(--kc-muted);
        }

        .kc-preview {
            display: none;
            position: relative;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--kc-border);
            background: #000;
            max-height: 220px;
        }

        .kc-preview img {
            width: 100%;
            max-height: 220px;
            object-fit: contain;
            display: block;
        }

        .kc-meta {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, .85);
            backdrop-filter: blur(8px);
            padding: 8px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: .75rem;
            border-top: 1px solid var(--kc-border);
        }

        .kc-meta span {
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 280px;
        }

        .kc-remove {
            background: transparent;
            border: none;
            color: #f87171;
            cursor: pointer;
            font-size: .8rem;
            font-weight: 600;
            padding: 2px 6px;
        }

        .kc-remove:hover {
            text-decoration: underline;
        }

        .kc-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .kc-field {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .kc-field label {
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .6px;
            color: var(--kc-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .kc-badge {
            color: #ffffff;
            background: var(--kc-surface);
            border: 1px solid var(--kc-border);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: .75rem;
        }

        .kc-slider {
            background: var(--kc-surface);
            border: 1px solid var(--kc-border);
            border-radius: 12px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
        }

        .kc-slider input {
            width: 100%;
            height: 4px;
            background: var(--kc-border);
            border-radius: 2px;
            accent-color: #ffffff;
            cursor: pointer;
        }

        .kc select {
            width: 100%;
            background: var(--kc-surface);
            border: 1px solid var(--kc-border);
            border-radius: 12px;
            color: #ffffff;
            font: inherit;
            font-size: .85rem;
            padding: 11px 14px;
            outline: none;
            cursor: pointer;
            transition: border-color .2s;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%23ffffff' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14 2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
        }

        .kc select:focus {
            border-color: #ffffff;
        }

        .kc-submit {
            width: 100%;
            padding: 16px;
            background: var(--kc-btn-bg);
            color: var(--kc-btn-text);
            border: none;
            border-radius: 14px;
            font: inherit;
            font-size: .95rem;
            font-weight: 800;
            letter-spacing: .5px;
            text-transform: uppercase;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all .2s ease;
            box-shadow: 0 4px 24px rgba(255, 255, 255, 0.2);
        }

        .kc-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(255, 255, 255, 0.35);
        }

        .kc-submit:active {
            transform: translateY(0);
        }

        .kc-submit:disabled {
            opacity: .4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .kc-spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid #000000;
            border-top-color: transparent;
            border-radius: 50%;
            animation: kc-spin .7s linear infinite;
        }

        @keyframes kc-spin {
            to {
                transform: rotate(360deg);
            }
        }
    </style>
</head>

<body>

    <div class="header">
        <div class="engine-badge">
            <div class="pulse"></div>
            Kraken BLLA Engine
        </div>
        <h1>Word Cropper</h1>
        <p class="subtitle">Upload an image to segment and extract individual cropped words using Kraken.</p>
    </div>

    <div class="kc">
        <form class="kc-form" id="kcForm" action="process.php" method="POST" enctype="multipart/form-data">
            <div class="kc-drop" id="kcDrop">
                <input type="file" name="image" id="kcFile" accept="image/png,image/jpeg,image/webp" required>
                <div class="kc-drop-icon">📄</div>
                <div class="kc-drop-title">Drop your image here or click to browse</div>
                <div class="kc-drop-sub">Supports PNG, JPG, WEBP • Max 20MB</div>
            </div>

            <div class="kc-preview" id="kcPreview">
                <img id="kcImg" src="" alt="Upload preview">
                <div class="kc-meta">
                    <span id="kcName"></span>
                    <button type="button" class="kc-remove" id="kcRemove">Remove</button>
                </div>
            </div>

            <div class="kc-grid">
                <div class="kc-field">
                    <label for="kcPad">Crop Padding <span class="kc-badge" id="kcPadVal">4 px</span></label>
                    <div class="kc-slider">
                        <input type="range" name="padding" id="kcPad" min="0" max="24" value="4">
                    </div>
                </div>
                <div class="kc-field">
                    <label for="kcFmt">Output Format</label>
                    <select name="output_format" id="kcFmt">
                        <option value="png" selected>PNG (Lossless)</option>
                        <option value="webp">WEBP (Optimized)</option>
                        <option value="jpeg">JPEG (Standard)</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="kc-submit" id="kcSubmit">
                <span class="kc-spinner" id="kcSpin"></span>
                <span id="kcBtnText">Scan &amp; Crop Words</span>
            </button>
        </form>
    </div>

    <div class="kc" style="margin-top:24px;">
        <div class="kc-form" id="htrForm">
            <div>
                <div class="kc-drop-title" style="margin-bottom:4px;">Prepare HTR Training Dataset</div>
                <div class="kc-drop-sub">Batch-processes every image saved in <code>output/</code> into
                    model-ready crops (grayscale, trimmed, height-normalized) and exports an Excel manifest
                    (<code>htr_dataset/dataset_labels.xlsx</code>) for transcription labeling.</div>
            </div>
            <button type="button" class="kc-submit" id="htrSubmit">
                <span class="kc-spinner" id="htrSpin"></span>
                <span id="htrBtnText">⚙️ Batch Process for HTR</span>
            </button>
            <div id="htrResult" style="display:none;font-size:.85rem;color:var(--kc-muted);line-height:1.6;"></div>
        </div>
    </div>

    <script>
        (function () {
            const $ = (id) => document.getElementById(id);
            const drop = $('kcDrop');
            const file = $('kcFile');
            const preview = $('kcPreview');

            $('kcPad').addEventListener('input', (e) => {
                $('kcPadVal').textContent = e.target.value + ' px';
            });

            ['dragenter', 'dragover'].forEach(name => {
                drop.addEventListener(name, (e) => {
                    e.preventDefault();
                    drop.classList.add('dragover');
                });
            });

            ['dragleave', 'drop'].forEach(name => {
                drop.addEventListener(name, (e) => {
                    e.preventDefault();
                    drop.classList.remove('dragover');
                });
            });

            file.addEventListener('change', () => {
                const f = file.files[0];
                if (!f) return;
                const reader = new FileReader();
                reader.onload = (e) => {
                    $('kcImg').src = e.target.result;
                    $('kcName').textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
                    drop.style.display = 'none';
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(f);
            });

            $('kcRemove').addEventListener('click', () => {
                file.value = '';
                $('kcImg').src = '';
                preview.style.display = 'none';
                drop.style.display = 'block';
            });

            $('kcForm').addEventListener('submit', () => {
                $('kcSubmit').disabled = true;
                $('kcSpin').style.display = 'inline-block';
                $('kcBtnText').textContent = 'Scanning with Kraken...';
            });

            // Batch-process everything saved in output/ into an HTR-ready dataset
            $('htrSubmit').addEventListener('click', async () => {
                const btn = $('htrSubmit');
                const spin = $('htrSpin');
                const text = $('htrBtnText');
                const result = $('htrResult');

                btn.disabled = true;
                spin.style.display = 'inline-block';
                text.textContent = 'Processing...';
                result.style.display = 'none';

                try {
                    const formData = new FormData();
                    formData.append('action', 'batch_process_htr');

                    const res = await fetch('index.php', { method: 'POST', body: formData });
                    const data = await res.json();

                    if (data.success) {
                        result.innerHTML = data.processed > 0
                            ? `✓ Processed ${data.processed} image(s) into <code>${data.images_dir}</code> ` +
                              `(${data.skipped} skipped).<br>` +
                              `<a href="${data.excel_file}" download style="color:#fff;font-weight:700;">⬇ Download dataset_labels.xlsx</a>`
                            : 'No images found in output/ yet. Save some cropped words first.';
                    } else {
                        result.textContent = 'Batch processing failed.';
                    }
                    result.style.display = 'block';
                } catch (err) {
                    result.textContent = 'Error: ' + err.message;
                    result.style.display = 'block';
                } finally {
                    btn.disabled = false;
                    spin.style.display = 'none';
                    text.textContent = '⚙️ Batch Process for HTR';
                }
            });
        })();
    </script>
</body>

</html>