#!/usr/bin/env python3
import os
import argparse
import urllib.request


MODELS = {
    "xception_video.pt": "https://example.com/path/to/xception_video.pt",
    "rawnet2_audio.pt": "https://example.com/path/to/rawnet2_audio.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Download model weights to MODEL_DIR")
    parser.add_argument("--dir", dest="dir", default=os.environ.get("MODEL_DIR", "/models"))
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for fname, url in MODELS.items():
        dest = os.path.join(args.dir, fname)
        if os.path.exists(dest):
            print(f"Exists: {dest}")
            continue
        try:
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
            print("OK")
        except Exception as e:
            print(f"Failed {url}: {e}")


if __name__ == "__main__":
    main()

