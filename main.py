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
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
from typing import Optional

# Create FastAPI app
app = FastAPI(title="VIS3D - 3D from Images using MiDaS")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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


@app.get("/")
async def root():
    if not NUMPY_AVAILABLE:
        return {"message": "WARNING: NumPy is not available. The service won't work until numpy is installed."}
    return {"message": "Welcome to VIS3D API - Convert images to 3D using MiDaS"}


@app.post("/convert_to_3d/")
async def convert_to_3d(file: UploadFile = File(...), sample_rate: Optional[int] = 4):
    """
    Convert an uploaded image to 3D using MiDaS.
    Returns both the depth map visualization and 3D point cloud data for Three.js.
    No files are saved to disk.

    Args:
        file: Image file to process
        sample_rate: Sampling rate for 3D points (higher values = fewer points but faster)

    Returns:
        JSON with threejs_data, depth and original image (base64 encoded)
    """
    # Check if numpy is available
    if not NUMPY_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="NumPy is not available. Please install numpy with 'pip install numpy'"
        )

    # Validate image file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Initialize model if not already done
    global depth_model
    if depth_model is None:
        try:
            depth_model = MidasDepthEstimation()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error initializing model: {str(e)}"
            )

    try:
        # Read image file into memory
        contents = await file.read()
        try:
            nparr = np.frombuffer(contents, np.uint8)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting image to numpy array: {str(e)}"
            )

        try:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding image with OpenCV: {str(e)}"
            )

        if img is None:
            raise HTTPException(
                status_code=400, detail="Could not decode image"
            )

        # Process the image and get 3D data
        try:
            result = depth_model.generate_3d_data_from_memory(img)
        except Exception as e:
            print(f"Error in generate_3d_data_from_memory: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Error generating 3D data: {str(e)}\n{traceback.format_exc()}"
            )

        # Return the results
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
