$sns = @(3288)

# Find the 实拍图片 directory dynamically using subfolder pattern match (avoiding literal Chinese characters in script)
$d_drive_622 = Get-ChildItem -Path D:\ -Directory | Where-Object { 
    $parent = $_.FullName
    (Get-ChildItem -Path $parent -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\d[\.-]\d+[\.-]\d' }).Count -gt 0 
} | Select-Object -ExpandProperty FullName -First 1

$dest_root = "C:\Users\Vilota\Downloads\missing_raw_images\622D"

if (-not $d_drive_622) {
    Write-Host "Error: Could not resolve the image directory on D drive dynamically."
    exit
}

Write-Host "Scanning directory: $d_drive_622 for BMP files..."
$files = Get-ChildItem -Path $d_drive_622 -Filter "*.bmp" -Recurse -File -ErrorAction SilentlyContinue
Write-Host "Found $($files.Count) total BMP files. Filtering SN 3288..."

$copyCount = 0
foreach ($file in $files) {
    if ($file.BaseName -match '^(\d+)-(\d+)$') {
        $sn = [int]$Matches[1]
        $pos = $Matches[2]
        
        if ($sns -contains $sn) {
            $targetDir = Join-Path $dest_root $pos
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
            }
            
            $destFile = Join-Path $targetDir $file.Name
            Copy-Item -Path $file.FullName -Destination $destFile -Force
            $copyCount++
            Write-Host "Copied: $($file.FullName) -> $destFile"
        }
    }
}

Write-Host "Done! Successfully copied $copyCount files to $dest_root"
