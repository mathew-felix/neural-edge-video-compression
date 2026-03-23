import os
import urllib.request
import sys

# Compute models directory relative to this script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RELEASE_URL_BASE = "https://github.com/mathew-felix/neural-edge-video-compression/releases/download/pretrained-models/"

MODELS_TO_DOWNLOAD = [
    "amt-l.pth",
    "amt-s.pth",
    "cvpr2025_image.pth.tar",
    "cvpr2025_video.pth.tar",
    "MDV6-yolov9-c.onnx",
    "MDV6-yolov9-c.pt"
]

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, int((downloaded / total_size) * 100)) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading... {percent}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    sys.stdout.flush()

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading models into {MODELS_DIR}...")
    
    for model_name in MODELS_TO_DOWNLOAD:
        model_path = os.path.join(MODELS_DIR, model_name)
        download_url = RELEASE_URL_BASE + model_name
        
        if os.path.exists(model_path):
            print(f"\n[SKIP] {model_name} already exists.")
            continue
            
        print(f"\n[FETCH] downloading {model_name} from {download_url}")
        try:
            urllib.request.urlretrieve(download_url, model_path, reporthook=download_progress_hook)
            print(f"\n[SUCCESS] {model_name} saved.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {model_name}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            sys.exit(1)
            
    print("\nAll models have been downloaded successfully!")

if __name__ == "__main__":
    main()
