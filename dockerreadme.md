docker run --gpus all -d `
  -p 8501:8501 `
  -v "${PWD}/blur_detection/assets:/workspace/blur_detection/blur_detection/assets" `
  -v "D:\600D:/data/input" `
  -v "D:\pipeline_outputs:/data/output" `
  --name blur_processor `
  --restart unless-stopped `
  blur-detection-app



Device name	Vilota-Compute
Processor	AMD Ryzen 7 9700X 8-Core Processor (3.80 GHz)
Installed RAM	32.0 GB (31.1 GB usable)
Graphics card	NVIDIA GeForce RTX 5060 Ti (16 GB)
AMD Radeon(TM) Graphics (486 MB)
Storage	427 GB of 1.82 TB used
Device ID	6EC95719-4EEE-465F-BC44-E3D1B20365FC
Product ID	00330-80000-00000-AA323
System type	64-bit operating system, x64-based processor
Pen and touch	No pen or touch input is available for this display
