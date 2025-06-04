import os
import torch
import numpy as np
from PIL import Image
import cv2
import io
import base64
import json
import traceback  # Added import

# Try to import numpy with specific error handling and a functional test
try:
    import numpy as np
    _test_np_midas = np.array([1, 2, 3])
    if _test_np_midas.sum() != 6:  # Basic check
        raise RuntimeError(
            "NumPy basic operation test failed logic check in midas_model.py.")
    NUMPY_MIDAS_AVAILABLE = True
    print("NumPy imported and basic test passed successfully in midas_model.py.")
except ImportError as e_imp:
    NUMPY_MIDAS_AVAILABLE = False
    np = None  # Crucial: ensure np is None if import fails
    print(f"ERROR: Failed to import NumPy in midas_model.py: {e_imp}")
    print("Traceback for NumPy import failure in midas_model.py:")
    print(traceback.format_exc())
except Exception as e_test:  # Catch other errors during import or test
    NUMPY_MIDAS_AVAILABLE = False
    np = None  # Crucial: ensure np is None if test fails
    print(
        f"ERROR: NumPy imported in midas_model.py but a test operation failed: {e_test}")
    print("Traceback for NumPy test failure in midas_model.py:")
    print(traceback.format_exc())

# Try to import matplotlib but handle if it's not available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available, using OpenCV for visualization")
    MATPLOTLIB_AVAILABLE = False
from download_model import download_model, MODEL_PATH


class MidasDepthEstimation:
    def __init__(self):
        # Check if model exists, otherwise download it
        if not os.path.exists(MODEL_PATH):
            download_model()

        # Initialize model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "dpt_hybrid"  # MiDaS v3 - DPT-Hybrid
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS DPT-Hybrid model."""
        # We need to import here to avoid loading torch and models at startup
        from torchvision.transforms import Compose
        import torch.nn as nn
        import torch.nn.functional as F

        # Import MiDaS directly from the PyTorch model file
        try:
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        except Exception as e:
            print(f"Error loading model from torch hub: {e}")
            # Fallback: load model directly from downloaded weights
            from midas.model_loader import default_models
            from midas.dpt_depth import DPTDepthModel

            self.model = DPTDepthModel(
                path=MODEL_PATH,
                backbone="vitb_rn50_384",
                non_negative=True,
            )

            # Define transform manually
            def transform(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = cv2.resize(img, (384, 384))
                img = torch.from_numpy(img).permute(2, 0, 1).float()
                return img.unsqueeze(0)

            self.transform = transform

        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, image_path):
        """
        Generate depth map from input image.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (original RGB image as numpy array, depth map as numpy array)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Store original image for return
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform input for model
        input_batch = self.transform(img).to(self.device)

        # Predict and resize to original resolution
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy array
        depth_map = prediction.cpu().numpy()

        # Normalize depth map for visualization
        depth_map = cv2.normalize(
            depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        return original_image, depth_map

    def estimate_depth_from_image(self, image_data):
        """
        Generate depth map from image data in memory.

        Args:
            image_data: Image as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (original RGB image as numpy array, depth map as numpy array)
        """
        # Store original image for return (convert BGR to RGB)
        original_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        # Transform input for model
        input_batch = self.transform(image_data).to(self.device)

        # Predict and resize to original resolution
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_data.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Check if NumPy was successfully initialized in this module
        if not NUMPY_MIDAS_AVAILABLE:
            raise RuntimeError(
                "NumPy is not available or failed to initialize within midas_model.py. Cannot convert tensor to NumPy array.")

        # Convert to numpy array
        depth_map = prediction.cpu().numpy()

        # Normalize depth map for visualization
        depth_map = cv2.normalize(
            depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        return original_image, depth_map

    def save_depth_map(self, image_path, output_dir):
        """
        Generate and save depth map from input image.

        Args:
            image_path: Path to input image
            output_dir: Directory to save output

        Returns:
            Dictionary with paths to the original image, depth image, and 3D data file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate file names
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_image_path = os.path.join(output_dir, f"{base_name}_depth.png")
        depth_data_path = os.path.join(output_dir, f"{base_name}_depth.npy")

        # Generate depth map
        rgb_image, depth_map = self.estimate_depth(image_path)

        # Save depth visualization
        if MATPLOTLIB_AVAILABLE:
            plt.imsave(depth_image_path, depth_map, cmap="inferno")
        else:
            # Use OpenCV to create colormap instead
            if not NUMPY_MIDAS_AVAILABLE:  # Added check before np usage
                raise RuntimeError(
                    "NumPy is not available or failed to initialize within midas_model.py for save_depth_map.")
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite(depth_image_path, depth_colored)

        # Save depth data for 3D reconstruction
        if not NUMPY_MIDAS_AVAILABLE:  # Added check before np usage
            raise RuntimeError(
                "NumPy is not available or failed to initialize within midas_model.py for save_depth_map (np.save).")
        np.save(depth_data_path, depth_map)

        # Return paths
        return {
            "original_image": image_path,
            "depth_image": depth_image_path,
            "depth_data": depth_data_path
        }

    def generate_3d_data(self, image_path, output_dir):
        """
        Generate 3D point cloud data suitable for Three.js rendering.

        Args:
            image_path: Path to input image
            output_dir: Directory to save output

        Returns:
            Path to the generated 3D JSON data file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate file names
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_3d.json")

        # Generate depth map
        rgb_image, depth_map = self.estimate_depth(image_path)

        # Create height/width for point cloud
        height, width = depth_map.shape

        # Create simplified mesh data for Three.js
        # We'll create a JSON format that Three.js can easily consume
        import json

        # Create a normalized vertices array
        # Scale vertices to be in [-0.5, 0.5] range for easier display in Three.js
        vertices = []
        colors = []

        # Sample points from the depth map (downscale for better performance)
        sample_rate = 4  # Adjust based on performance needs
        for y in range(0, height, sample_rate):
            for x in range(0, width, sample_rate):
                # Create normalized coordinates
                norm_x = (x / width) - 0.5
                # Flip Y for 3D coordinate system
                norm_y = -((y / height) - 0.5)
                # Negative to make points go "into" the screen
                norm_z = -depth_map[y, x] * 0.5

                # Add vertex
                vertices.extend([norm_x, norm_y, norm_z])

                # Add color (normalized RGB values)
                r = rgb_image[y, x, 0] / 255.0
                g = rgb_image[y, x, 1] / 255.0
                b = rgb_image[y, x, 2] / 255.0
                colors.extend([r, g, b])

        # Create JSON data for Three.js
        threejs_data = {
            "vertices": vertices,
            "colors": colors,
            "metadata": {
                "version": 1.0,
                "type": "points",
                "points": len(vertices) // 3,
                "width": width,
                "height": height
            }
        }

        # Save JSON file
        with open(json_path, 'w') as f:
            json.dump(threejs_data, f)

        return json_path

    def generate_3d_data_from_memory(self, image_data, sample_rate=4, depth_threshold=0.0):
        """
        Generate 3D point cloud data suitable for Three.js rendering directly 
        from image data without saving files.

        Args:
            image_data: Image as numpy array (BGR format from OpenCV)
            sample_rate: Sampling rate for the point cloud (higher = fewer points)
            depth_threshold: Filter out points with depth values below this threshold (0.0-1.0)

        Returns:
            Dict containing the 3D point cloud data and depth map visualization
        """
        # Generate depth map from image data
        rgb_image, depth_map = self.estimate_depth_from_image(image_data)

        # Create height/width for point cloud
        height, width = depth_map.shape

        # Create normalized vertices array and colors for Three.js
        vertices = []
        colors = []

        # Sample points from the depth map (downscale for better performance)
        # Ensure sample_rate is at least 4 for reasonable performance
        sample_rate = max(4, sample_rate)
        
        # Convert threshold from 0-1 range to the actual depth map range
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        depth_range = max_depth - min_depth
        actual_threshold = min_depth + (depth_threshold * depth_range)
        
        # Optional: Apply adaptive sampling based on local depth variance
        # This creates denser sampling in detailed areas and sparser in flat areas
        adaptive_sampling = False
        if adaptive_sampling:
            from scipy.ndimage import gaussian_filter
            # Calculate gradient magnitude of depth map
            dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(dx**2 + dy**2)
            gradient_mag = gaussian_filter(gradient_mag, sigma=2)
            # Normalize
            gradient_mag = (gradient_mag - np.min(gradient_mag)) / (np.max(gradient_mag) - np.min(gradient_mag))
        
        for y in range(0, height, sample_rate):
            for x in range(0, width, sample_rate):
                # Skip points below the depth threshold (filter out distant points)
                if depth_map[y, x] < actual_threshold:
                    continue
                
                # Create normalized coordinates
                norm_x = (x / width) - 0.5
                # Flip Y for 3D coordinate system
                norm_y = -((y / height) - 0.5)
                # Negative to make points go "into" the screen
                norm_z = -depth_map[y, x] * 0.5

                # Add vertex
                vertices.extend([norm_x, norm_y, norm_z])

                # Add color (normalized RGB values)
                r = rgb_image[y, x, 0] / 255.0
                g = rgb_image[y, x, 1] / 255.0
                b = rgb_image[y, x, 2] / 255.0
                colors.extend([r, g, b])

        # Create Three.js data structure
        threejs_data = {
            "vertices": vertices,
            "colors": colors,
            "metadata": {
                "version": 1.0,
                "type": "points",
                "points": len(vertices) // 3,
                "width": width,
                "height": height
            }
        }

        # Create depth visualization image
        if MATPLOTLIB_AVAILABLE:
            # Use matplotlib for better colormaps
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(depth_map, cmap='inferno')
            plt.axis('off')

            # Save to a BytesIO buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)

            # Create base64 encoded string
            depth_viz_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        else:
            # Use OpenCV as fallback
            if not NUMPY_MIDAS_AVAILABLE:  # Added check before np usage
                raise RuntimeError(
                    "NumPy is not available or failed to initialize within midas_model.py for depth visualization.")
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            _, buffer = cv2.imencode('.png', depth_colored)
            depth_viz_base64 = base64.b64encode(buffer).decode('utf-8')

        # Also encode the original image
        _, buffer = cv2.imencode(
            '.png', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        original_viz_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return everything together
        return {
            "threejs_data": threejs_data,
            "depth_image_base64": f"data:image/png;base64,{depth_viz_base64}",
            "original_image_base64": f"data:image/png;base64,{original_viz_base64}"
        }
