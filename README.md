# Vis3D

A web application that converts images to interactive 3D point clouds using MiDaS depth estimation.

## Features

- Upload and process images to create 3D visualizations
- Interactive point cloud visualization with Three.js
- Adjustable point density, size, and visual style
- Depth threshold filtering to remove distant points
- Z-scale adjustment for depth intensity control
- Download generated 3D models in PLY format
- Visualization controls (axes, background color, camera reset)

## Live Application
[Check out the tool here](https://vis3d.fly.dev/)

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. Open your browser and navigate to http://localhost:8000

## Getting the Best 3D Visualization

For optimal results, consider these settings based on your image and hardware capabilities:

### For High-End Hardware (Best Quality):
- **Point Density**: High (Slow)
- **Detail Level**: High Detail
- **Point Size**: 0.003-0.008 (adjust based on preference)
- **Depth Scale**: 1.0-1.5 (adjust for best depth perception)
- **Depth Threshold**: Start at 0 and increase to filter out distant noise

### For Balanced Performance:
- **Point Density**: Medium
- **Detail Level**: Balanced
- **Point Size**: 0.01
- **Depth Scale**: 1.0
- **Depth Threshold**: 0.1-0.3 to reduce point count

### For Lower-End Hardware (Best Performance):
- **Point Density**: Low (Fast)
- **Detail Level**: Low Detail
- **Point Size**: 0.015-0.02 (larger points to fill gaps)
- **Depth Scale**: 0.8-1.0
- **Point Style**: Square (better performance)

### Tips for Best Results:
- Images with clear subjects and distinct depth layers work best
- Apply depth filtering to remove noise (increase Depth Threshold)
- Use "Reapply Filters" to update visualization after changing parameters
- Toggle Size Attenuation off for orthographic-style visualization
- Round point style generally looks better but may be slower
- Use a dark background for best contrast

## Technologies Used
- Python (FastAPI backend)
- MiDaS depth estimation model
- Three.js for 3D visualization
- HTML/CSS/JavaScript for frontend
