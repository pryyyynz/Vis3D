#!/usr/bin/env python3
"""
Vis3D API Client Example

This script demonstrates how to use the Vis3D API from another Python application.
It shows how to make requests to the API and process the responses.
"""

import requests
import json
import base64
import os
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Default API endpoint (if running locally)
DEFAULT_API_URL = "http://localhost:8000"

def test_api_connection(api_url=DEFAULT_API_URL):
    """Test if the API is working by calling the /api/test endpoint."""
    try:
        response = requests.get(f"{api_url}/api/test")
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        return None

def convert_image_to_3d(image_path, api_url=DEFAULT_API_URL, sample_rate=8, 
                       detail_level="balanced", depth_threshold=0.0):
    """
    Convert an image file to a 3D point cloud using the Vis3D API.
    
    Args:
        image_path: Path to the image file to convert
        api_url: Base URL of the Vis3D API
        sample_rate: Sampling rate for the point density (4, 8, 12, or 16)
        detail_level: Level of detail ("high", "balanced", "low")
        depth_threshold: Depth threshold for filtering points (0.0-1.0)
        
    Returns:
        Dictionary containing the API response data, or None if the request failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # Prepare the file and data to send
    files = {
        'file': open(image_path, 'rb')
    }
    
    data = {
        'sample_rate': sample_rate,
        'detail_level': detail_level,
        'depth_threshold': depth_threshold
    }
    
    try:
        # Print info about the request
        print(f"Sending image to {api_url}/convert_to_3d/")
        print(f"Parameters: sample_rate={sample_rate}, detail_level={detail_level}, depth_threshold={depth_threshold}")
        
        # Make the API request
        start_time = time.time()
        response = requests.post(f"{api_url}/convert_to_3d/", files=files, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the response time
        elapsed_time = time.time() - start_time
        print(f"Request completed in {elapsed_time:.2f} seconds")
        
        # Parse the JSON response
        result = response.json()
        
        # Print info about the response
        if "threejs_data" in result and "vertices" in result["threejs_data"]:
            num_points = len(result["threejs_data"]["vertices"]) // 3
            print(f"Received 3D data with {num_points} points")
        
        return result
    
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    finally:
        # Close the file
        if 'file' in files:
            files['file'].close()

def save_3d_data(response_data, output_dir):
    """
    Save the 3D data from the API response to disk.
    
    Args:
        response_data: API response data
        output_dir: Directory to save the output files
    """
    if not response_data:
        print("No data to save.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename for outputs
    base_filename = f"vis3d_output_{int(time.time())}"
    
    # Save original image
    if "original_image_base64" in response_data:
        # Extract the base64 image data (remove data:image/png;base64, prefix)
        img_data = response_data["original_image_base64"].split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        
        # Save to file
        img_path = os.path.join(output_dir, f"{base_filename}_original.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"Original image saved to {img_path}")
    
    # Save depth map image
    if "depth_image_base64" in response_data:
        # Extract the base64 image data (remove data:image/png;base64, prefix)
        img_data = response_data["depth_image_base64"].split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        
        # Save to file
        img_path = os.path.join(output_dir, f"{base_filename}_depth.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"Depth map saved to {img_path}")
    
    # Save 3D point cloud data as PLY file
    if "threejs_data" in response_data and "vertices" in response_data["threejs_data"]:
        try:
            # Get vertices and colors
            vertices = response_data["threejs_data"]["vertices"]
            colors = response_data["threejs_data"]["colors"]
            
            # Create PLY file
            ply_path = os.path.join(output_dir, f"{base_filename}_pointcloud.ply")
            write_ply(ply_path, vertices, colors)
            print(f"3D point cloud saved to {ply_path}")
            
            # Also save as JSON for direct use with Three.js
            json_path = os.path.join(output_dir, f"{base_filename}_threejs.json")
            with open(json_path, "w") as f:
                json.dump(response_data["threejs_data"], f)
            print(f"Three.js data saved to {json_path}")
            
        except Exception as e:
            print(f"Error saving 3D data: {e}")

def write_ply(filepath, vertices, colors):
    """
    Write a point cloud to PLY file format.
    
    Args:
        filepath: Output PLY file path
        vertices: List of vertex coordinates [x1, y1, z1, x2, y2, z2, ...]
        colors: List of RGB color values [r1, g1, b1, r2, g2, b2, ...] in range 0-1
    """
    # Convert flat lists to numpy arrays of vertices and colors
    num_points = len(vertices) // 3
    vertices_array = np.array(vertices).reshape(num_points, 3)
    colors_array = np.array(colors).reshape(num_points, 3)
    
    # Convert colors from 0-1 float to 0-255 int
    colors_array = (colors_array * 255).astype(np.uint8)
    
    # Write PLY header
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(num_points):
            x, y, z = vertices_array[i]
            r, g, b = colors_array[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"PLY file written with {num_points} points")

def main():
    """Main function to demonstrate the API client."""
    parser = argparse.ArgumentParser(description="Vis3D API Client")
    parser.add_argument("--image", type=str, help="Path to image file to process")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, 
                        help=f"Base URL of the Vis3D API (default: {DEFAULT_API_URL})")
    parser.add_argument("--sample-rate", type=int, default=8, choices=[4, 8, 12, 16],
                        help="Sampling rate for point density (default: 8)")
    parser.add_argument("--detail-level", type=str, default="balanced", 
                        choices=["high", "balanced", "low"],
                        help="Level of detail (default: balanced)")
    parser.add_argument("--depth-threshold", type=float, default=0.0,
                        help="Depth threshold for filtering points (0.0-1.0) (default: 0.0)")
    parser.add_argument("--output-dir", type=str, default="./vis3d_output",
                        help="Directory to save output files (default: ./vis3d_output)")
    parser.add_argument("--test", action="store_true", 
                        help="Test the API connection only")
    
    args = parser.parse_args()
    
    # Test API connection
    if args.test or not args.image:
        print(f"Testing connection to {args.api_url}...")
        result = test_api_connection(args.api_url)
        if result:
            print(f"API test successful: {result}")
        else:
            print("API test failed.")
        
        if not args.image:
            return
    
    # Convert image to 3D
    print(f"Processing image: {args.image}")
    response_data = convert_image_to_3d(
        args.image, 
        args.api_url,
        args.sample_rate,
        args.detail_level,
        args.depth_threshold
    )
    
    if response_data:
        # Save the data to disk
        save_3d_data(response_data, args.output_dir)

if __name__ == "__main__":
    main()
