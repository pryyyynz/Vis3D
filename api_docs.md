# Vis3D API Documentation

This document provides information on how to use the Vis3D API from external applications.

## Base URL

When running locally, the base URL is:
```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication.

## Endpoints

### Test API Connection

```
GET /api/test
```

Test if the API is up and running.

#### Response

```json
{
  "message": "API is working!",
  "status": "ok"
}
```

### Convert Image to 3D

```
POST /convert_to_3d/
```

Convert an uploaded image to 3D point cloud data.

#### Parameters

| Name           | Type   | Required | Description                                                |
|----------------|--------|----------|------------------------------------------------------------|
| file           | File   | Yes      | Image file to process                                      |
| sample_rate    | Int    | No       | Sampling rate for points (4=High, 8=Medium, 16=Low)        |
| detail_level   | String | No       | Detail level ("high", "balanced", "low")                   |
| depth_threshold| Float  | No       | Filter out points with depth values below this (0.0-1.0)   |

#### Response

The response is a JSON object with the following structure:

```json
{
  "threejs_data": {
    "vertices": [x1, y1, z1, x2, y2, z2, ...],
    "colors": [r1, g1, b1, r2, g2, b2, ...],
    "metadata": {
      "version": 1.0,
      "type": "points",
      "points": 5700,
      "width": 449,
      "height": 800
    }
  },
  "depth_image_base64": "data:image/png;base64,/9j/4AAQS...",
  "original_image_base64": "data:image/png;base64,/9j/4AAQS..."
}
```

#### Example Usage (Python)

```python
import requests

# API endpoint
url = "http://localhost:8000/convert_to_3d/"

# Open image file
files = {'file': open('image.jpg', 'rb')}

# Additional parameters
data = {
    'sample_rate': 8,
    'detail_level': 'balanced',
    'depth_threshold': 0.0
}

# Make the request
response = requests.post(url, files=files, data=data)

# Parse the response
result = response.json()

# Access the 3D data
threejs_data = result['threejs_data']
depth_image = result['depth_image_base64']
original_image = result['original_image_base64']
```

#### Example Usage (JavaScript)

```javascript
async function convert3D(imageFile) {
    const url = 'http://localhost:8000/convert_to_3d/';
    
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('sample_rate', '8');
    formData.append('detail_level', 'balanced');
    formData.append('depth_threshold', '0.0');
    
    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`Processing failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
}

// Usage:
// const fileInput = document.getElementById('fileInput');
// const file = fileInput.files[0];
// const result = await convert3D(file);
```

### API for External Applications (Simplified)

```
POST /api/v1/convert
```

A simplified endpoint designed for external applications with cleaner response structure.

#### Parameters

| Name           | Type   | Required | Description                                                |
|----------------|--------|----------|------------------------------------------------------------|
| file           | File   | Yes      | Image file to process                                      |
| sample_rate    | Int    | No       | Sampling rate for points (4=High, 8=Medium, 16=Low)        |
| detail_level   | String | No       | Detail level ("high", "balanced", "low")                   |
| depth_threshold| Float  | No       | Filter out points with depth values below this (0.0-1.0)   |
| include_images | Bool   | No       | Whether to include base64 encoded images in the response   |

#### Response

The response is a JSON object with a cleaner structure:

```json
{
  "success": true,
  "data": {
    "threejs_data": {
      "vertices": [x1, y1, z1, x2, y2, z2, ...],
      "colors": [r1, g1, b1, r2, g2, b2, ...],
      "metadata": {
        "version": 1.0,
        "type": "points",
        "points": 5700,
        "width": 449,
        "height": 800
      }
    },
    "metadata": {
      "vertices_count": 5700,
      "width": 449,
      "height": 800
    }
  }
}
```

If `include_images=true`, the response will also include:

```json
"images": {
  "depth_map": "data:image/png;base64,/9j/4AAQS...",
  "original": "data:image/png;base64,/9j/4AAQS..."
}
```

#### Example Usage (Python)

```python
import requests

# API endpoint
url = "http://localhost:8000/api/v1/convert"

# Open image file
files = {'file': open('image.jpg', 'rb')}

# Additional parameters
data = {
    'sample_rate': 8,
    'detail_level': 'balanced',
    'depth_threshold': 0.0,
    'include_images': False
}

# Make the request
response = requests.post(url, files=files, data=data)

# Parse the response
result = response.json()

if result['success']:
    # Access the 3D data
    threejs_data = result['data']['threejs_data']
    vertices_count = result['data']['metadata']['vertices_count']
    print(f"Received 3D point cloud with {vertices_count} points")
else:
    print(f"Error: {result['error']['message']}")
```

#### Example Usage (JavaScript)

```javascript
async function convert3DApi(imageFile) {
    const url = 'http://localhost:8000/api/v1/convert';
    
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('sample_rate', '8');
    formData.append('detail_level', 'balanced');
    formData.append('depth_threshold', '0.0');
    formData.append('include_images', 'false');
    
    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
        return data.data;
    } else {
        throw new Error(`Processing failed: ${data.error.message}`);
    }
}

// Usage example:
// const fileInput = document.getElementById('fileInput');
// const file = fileInput.files[0];
// try {
//     const result = await convert3DApi(file);
//     console.log(`Received point cloud with ${result.metadata.vertices_count} points`);
// } catch (error) {
//     console.error("Error:", error);
// }
```

## API Documentation

Interactive API documentation is available at:
```
http://localhost:8000/docs
```

This provides a Swagger UI interface where you can test the API endpoints directly.
