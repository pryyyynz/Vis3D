import os
import sys
import requests
import torch
from tqdm import tqdm

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODELS_DIR, "dpt_hybrid-midas-501f0c75.pt")

# List of potential URLs to try in order
MODEL_URLS = [
    # Direct links to DPT-Hybrid model
    "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_hybrid-midas-501f0c75.pt",
    "https://github.com/isl-org/MiDaS/releases/download/v3.1/dpt_hybrid-midas-501f0c75.pt",
    "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid-midas-501f0c75.pt",
    "https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/dpt_hybrid-midas-501f0c75.pt",
    # Fall back to any version if the specific one isn't available
    "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_hybrid_nyu-2f14c969.pt",
    "https://github.com/isl-org/MiDaS/releases/download/v2.1/model-f6b98070.pt"
]


def download_file(url, dest_path):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for non-200 status codes

        # Get file size from headers
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)

        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        # Remove partial file if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_model_torch_hub():
    """Attempt to download the model via torch hub"""
    try:
        print("Trying to download model via torch hub...")
        model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
        # Save the model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to download via torch hub: {e}")
        return False


def download_model():
    """Download the MiDaS DPT-Hybrid model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH

    # Try downloading directly from URLs
    for url in MODEL_URLS:
        print(f"\nTrying to download from: {url}")
        if download_file(url, MODEL_PATH):
            print(f"\nModel successfully downloaded to {MODEL_PATH}")
            return MODEL_PATH

    # If direct download failed, try torch hub
    if download_model_torch_hub():
        return MODEL_PATH

    # If all methods fail, raise error
    raise RuntimeError(
        "Failed to download MiDaS model from all sources. "
        "Please manually download one of the MiDaS models from "
        "https://github.com/isl-org/MiDaS/releases and place it in the models directory."
    )


if __name__ == "__main__":
    try:
        download_model()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
