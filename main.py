from midas_model import MidasDepthEstimation
import os
import cv2
import traceback
# Import numpy with specific error handling and a functional test
try:
    import numpy as np
    # Perform a simple operation to confirm functionality
    _test_arr = np.array([1, 2, 3])
    if _test_arr.sum() != 6: # Basic check on the result
        # This custom exception helps differentiate from a direct numpy error
        raise RuntimeError("NumPy basic operation test failed logic check.")
    NUMPY_AVAILABLE = True
    print("NumPy imported and basic test passed successfully in main.py.")
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None # Ensure np is defined as None if import fails
    print("ERROR: Failed to import NumPy in main.py.")
    print(f"Specific ImportError: {e}")
    print("Traceback for NumPy import failure in main.py:")
    print(traceback.format_exc())
except Exception as e: # Catch other errors during import or test (e.g., RuntimeError from np.array or the test)
    NUMPY_AVAILABLE = False
    np = None # Ensure np is defined as None if test fails
    print(f"ERROR: NumPy imported but a test operation failed in main.py: {e}")
    print("Traceback for NumPy test failure in main.py:")
    print(traceback.format_exc())
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

# Create FastAPI app
app = FastAPI(title="VIS3D - 3D from Images using MiDaS")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Type", "X-Content-Type-Options"]
)

# Mount static files for the frontend
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize the MiDaS model
depth_model = None


@app.on_event("startup")
async def startup_event():
    global depth_model
    # Check if numpy is available first
    if not NUMPY_AVAILABLE:
        print(
            "ERROR: NumPy is not available. Please install numpy with 'pip install numpy'")
        return

    try:
        depth_model = MidasDepthEstimation()
        print(f"MiDaS model initialized on {depth_model.device}")
    except Exception as e:
        print(f"Error initializing MiDaS model: {e}")
        print(traceback.format_exc())


@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the main frontend HTML
    index_path = BASE_DIR / "static" / "index.html"
    html_content = index_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content, media_type="text/html")


@app.post("/convert_to_3d/")
async def convert_to_3d(
    file: UploadFile = File(...), 
    sample_rate: Optional[int] = 8,
    detail_level: Optional[str] = "balanced",
    depth_threshold: Optional[float] = 0.0
):
    """
    Convert an uploaded image to 3D using MiDaS.
    Returns both the depth map visualization and 3D point cloud data for Three.js.
    No files are saved to disk.

    Args:
        file: Image file to process
        sample_rate: Sampling rate for 3D points (higher values = fewer points but faster)
        detail_level: Level of detail for the 3D point cloud ("high", "balanced", "low")
        depth_threshold: Filter out points with depth values below this threshold (0.0-1.0)

    Returns:
        JSON with threejs_data, depth and original image (base64 encoded)
    """
    print(f"=== Convert to 3D endpoint called with file: {file.filename}, sample_rate: {sample_rate} ===")
    # Check if numpy is available
    if not NUMPY_AVAILABLE:
        print("ERROR: NumPy not available!")
        raise HTTPException(
            status_code=500,
            detail="NumPy is not available. Please install numpy with 'pip install numpy'"
        )

    # Validate image file
    if not file.content_type.startswith("image/"):
        print(f"ERROR: Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")

    # Initialize model if not already done
    global depth_model
    if depth_model is None:
        print("Initializing depth model - it wasn't loaded at startup")
        try:
            depth_model = MidasDepthEstimation()
            print(f"Model initialized on {depth_model.device}")
        except Exception as e:
            print(f"ERROR initializing model: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error initializing model: {str(e)}"
            )
    else:
        print(f"Using already initialized model on {depth_model.device}")

    try:
        # Read image file into memory
        print("Reading uploaded file...")
        contents = await file.read()
        try:
            print("Converting to numpy array...")
            nparr = np.frombuffer(contents, np.uint8)
        except Exception as e:
            print(f"ERROR converting to numpy: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error converting image to numpy array: {str(e)}"
            )

        try:
            print("Decoding image with OpenCV...")
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize large images to reduce processing time and response size
            max_dimension = 800  # Maximum dimension for processing
            h, w = img.shape[:2]
            
            if max(h, w) > max_dimension:
                print(f"Resizing image from {w}x{h} to fit within {max_dimension}px")
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Resized to {new_w}x{new_h}")
            
        except Exception as e:
            print(f"ERROR decoding with OpenCV: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding image with OpenCV: {str(e)}"
            )

        if img is None:
            print("ERROR: Decoded image is None")
            raise HTTPException(
                status_code=400, detail="Could not decode image"
            )
        
        print(f"Successfully loaded image with shape: {img.shape}")

        # Process the image and get 3D data
        try:
            # Adjust sample_rate based on detail_level if provided
            adjusted_sample_rate = sample_rate
            if detail_level == "high":
                # Increase density (decrease sample rate) for high detail
                adjusted_sample_rate = max(4, sample_rate - 2)
            elif detail_level == "low":
                # Decrease density (increase sample rate) for low detail
                adjusted_sample_rate = sample_rate + 2
            
            print(f"Generating 3D data from image with sample_rate={adjusted_sample_rate}, depth_threshold={depth_threshold}...")
            result = depth_model.generate_3d_data_from_memory(
                img, 
                adjusted_sample_rate,
                depth_threshold=depth_threshold
            )
            
            # Calculate response size
            import sys
            import json
            serialized = json.dumps(result)
            response_size_mb = sys.getsizeof(serialized) / (1024 * 1024)
            
            # Debug info
            vertices_count = len(result["threejs_data"]["vertices"]) / 3
            print(f"3D data generated with {vertices_count} points (response size: {response_size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"ERROR in generate_3d_data_from_memory: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error generating 3D data: {str(e)}\n{traceback.format_exc()}"
            )

        # Return the results
        print("Returning successful response")
        return JSONResponse(content=result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/api/test")
async def test_api():
    """Simple endpoint to test if the API is working."""
    return {"message": "API is working!", "status": "ok"}


@app.get("/api/docs")
async def get_documentation():
    """Redirect to the API documentation."""
    return HTMLResponse(content=f"""
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/docs" />
            <title>Redirecting to API docs</title>
        </head>
        <body>
            <p>Redirecting to <a href="/docs">API documentation</a>...</p>
        </body>
    </html>
    """)


class DetailLevel(str, Enum):
    HIGH = "high"
    BALANCED = "balanced"
    LOW = "low"


@app.post("/api/v1/convert")
async def api_convert_to_3d(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = 8,
    detail_level: Optional[DetailLevel] = DetailLevel.BALANCED,
    depth_threshold: Optional[float] = 0.0,
    include_images: Optional[bool] = False
):
    """
    API endpoint for external applications to convert images to 3D point clouds.
    Returns a clean response structure with just the necessary data.
    
    Args:
        file: Image file to process
        sample_rate: Sampling rate for 3D points (4=High, 8=Medium, 16=Low)
        detail_level: Level of detail for the 3D point cloud 
        depth_threshold: Filter out points with depth values below this threshold (0.0-1.0)
        include_images: Whether to include base64 encoded images in the response (can be large)
        
    Returns:
        JSON with threejs_data and optionally the depth and original images
    """
    try:
        # Use the existing implementation for processing
        print(f"=== API v1 convert endpoint called with file: {file.filename} ===")
        
        # Process the image using the existing logic but with a wrapper
        # Reuse the existing function for conversion 
        result = await convert_to_3d(
            file=file,
            sample_rate=sample_rate,
            detail_level=detail_level.value,
            depth_threshold=depth_threshold
        )
        
        # Extract content from JSONResponse
        if isinstance(result, JSONResponse):
            result = result.body.decode()
            import json
            result = json.loads(result)
        
        # Create a simplified response structure for API clients
        response = {
            "success": True,
            "data": {
                "threejs_data": result["threejs_data"],
                "metadata": {
                    "vertices_count": len(result["threejs_data"]["vertices"]) // 3,
                    "width": result["threejs_data"]["metadata"]["width"],
                    "height": result["threejs_data"]["metadata"]["height"]
                }
            }
        }
        
        # Optionally include image data
        if include_images:
            response["data"]["images"] = {
                "depth_map": result["depth_image_base64"],
                "original": result["original_image_base64"]
            }
            
        return response
        
    except HTTPException as he:
        return {
            "success": False,
            "error": {
                "code": he.status_code,
                "message": he.detail
            }
        }
    except Exception as e:
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return {
            "success": False,
            "error": {
                "code": 500,
                "message": f"Internal server error: {str(e)}"
            }
        }
