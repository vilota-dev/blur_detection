$sns = @(2799, 2782, 2759, 2957, 2959)

# Resolve E:\实拍图片 path dynamically using child pattern match (avoiding literal Chinese characters in script)
$e_drive_621 = Get-ChildItem -Path E:\ -Directory | Where-Object { 
    $parent = $_.FullName
    (Get-ChildItem -Path $parent -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^5\.\d+' }).Count -gt 0 
} | Select-Object -ExpandProperty FullName -First 1

$dest_root = "C:\Users\Vilota\Downloads\missing_raw_images\621D"

if (-not $e_drive_621) {
    Write-Host "Error: Could not resolve the image directory on E drive dynamically."
    exit
}

Write-Host "Scanning directory: $e_drive_621 for BMP files..."
$files = Get-ChildItem -Path $e_drive_621 -Filter "*.bmp" -Recurse -File -ErrorAction SilentlyContinue
Write-Host "Found $($files.Count) total BMP files. Filtering missing SNs..."

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
