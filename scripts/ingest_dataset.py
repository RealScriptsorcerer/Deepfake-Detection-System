#!/usr/bin/env python3
import argparse
import os
import json
import glob
import requests


def main():
    parser = argparse.ArgumentParser(description="Ingest a folder of media into the Deepfake API")
    parser.add_argument("folder", help="Path to folder containing media files")
    parser.add_argument("api", help="Base API URL, e.g. http://localhost:8000")
    parser.add_argument("--key", dest="key", default=os.environ.get("DETECT_API_KEY", ""), help="API key")
    args = parser.parse_args()

    patterns = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.wav", "**/*.mp3", "**/*.flac", "**/*.m4a", "**/*.ogg"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.folder, p), recursive=True))
    print(f"Found {len(files)} files")

    headers = {"x-api-key": args.key} if args.key else {}
    for path in files:
        fn = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                files_mp = {"file": (fn, f)}
                # Route based on extension
                lower = fn.lower()
                if lower.endswith((".png",".jpg",".jpeg")):
                    url = f"{args.api}/detect/image"
                elif lower.endswith((".mp4",".avi",".mov",".mkv")):
                    url = f"{args.api}/detect/video"
                else:
                    url = f"{args.api}/detect/audio"
                resp = requests.post(url, files=files_mp, headers=headers, timeout=600)
                if resp.ok:
                    data = resp.json()
                    print(json.dumps({"file": fn, "id": data.get("id"), "label": data.get("label"), "score": data.get("score")}))
                else:
                    print(json.dumps({"file": fn, "error": resp.text}))
        except Exception as e:
            print(json.dumps({"file": fn, "exception": str(e)}))


if __name__ == "__main__":
    main()

