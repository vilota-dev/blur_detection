$sns = @(2209, 2401, 2415, 2426, 2479, 2662, 2250, 2253, 2259, 2299, 2307, 2310, 2319, 2323, 2600, 2599, 2598, 2593, 2544, 2377, 2553)
$d_drive_620d = "D:\620D"
$dest_root = "C:\Users\Vilota\Downloads\missing_raw_images\620D"

Write-Host "Scanning D:\620D for BMP files..."
$files = Get-ChildItem -Path $d_drive_620d -Filter "*.bmp" -Recurse -File -ErrorAction SilentlyContinue
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
        }
    }
}

Write-Host "Done! Successfully copied $copyCount files to $dest_root"
