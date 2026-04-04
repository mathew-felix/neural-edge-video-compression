import hashlib
import os
import sys
import urllib.request


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKSUMS_PATH = os.path.join(PROJECT_ROOT, "docs", "model_checksums.sha256")
RELEASE_URL_BASE = "https://github.com/mathew-felix/neural-edge-video-compression/releases/download/pretrained-models/"

MODELS_TO_DOWNLOAD = [
    "amt-l.pth",
    "amt-s.pth",
    "cvpr2025_image.pth.tar",
    "cvpr2025_video.pth.tar",
    "MDV6-yolov9-c.onnx",
    "MDV6-yolov9-c.pt",
]


def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, int((downloaded / total_size) * 100)) if total_size > 0 else 0
    sys.stdout.write(
        f"\rDownloading... {percent}% ({downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)"
    )
    sys.stdout.flush()


def load_expected_hashes():
    if not os.path.exists(CHECKSUMS_PATH):
        raise FileNotFoundError(f"Checksum manifest not found: {CHECKSUMS_PATH}")

    expected = {}
    with open(CHECKSUMS_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid checksum manifest line: {raw_line.rstrip()}")
            sha256_hex, rel_path = parts
            model_name = os.path.basename(rel_path.replace("\\", "/"))
            expected[model_name] = sha256_hex.lower()
    return expected


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def download_and_verify(model_name, model_path, download_url, expected_sha256):
    temp_path = model_path + ".part"
    safe_remove(temp_path)

    print(f"\n[FETCH] downloading {model_name} from {download_url}")
    urllib.request.urlretrieve(download_url, temp_path, reporthook=download_progress_hook)
    print()

    actual_sha256 = sha256_file(temp_path)
    if actual_sha256 != expected_sha256:
        safe_remove(temp_path)
        raise RuntimeError(
            f"Checksum mismatch for {model_name}: expected {expected_sha256}, got {actual_sha256}"
        )

    os.replace(temp_path, model_path)
    print(f"[SUCCESS] {model_name} saved and checksum verified.")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    expected_hashes = load_expected_hashes()
    print(f"Downloading models into {MODELS_DIR}...")
    print(f"Using checksum manifest: {CHECKSUMS_PATH}")

    for model_name in MODELS_TO_DOWNLOAD:
        model_path = os.path.join(MODELS_DIR, model_name)
        download_url = RELEASE_URL_BASE + model_name
        expected_sha256 = expected_hashes.get(model_name)

        if not expected_sha256:
            print(f"[ERROR] No checksum entry found for {model_name} in {CHECKSUMS_PATH}")
            sys.exit(1)

        if os.path.exists(model_path):
            actual_sha256 = sha256_file(model_path)
            if actual_sha256 == expected_sha256:
                print(f"\n[SKIP] {model_name} already exists and checksum verified.")
                continue

            print(
                f"\n[WARN] Existing {model_name} failed checksum verification; redownloading."
            )
            safe_remove(model_path)

        try:
            download_and_verify(model_name, model_path, download_url, expected_sha256)
        except Exception as e:
            print(f"[ERROR] Failed to fetch verified copy of {model_name}: {e}")
            safe_remove(model_path + ".part")
            safe_remove(model_path)
            sys.exit(1)

    print("\nAll models are present and checksum verified.")


if __name__ == "__main__":
    main()
