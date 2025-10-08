from __future__ import annotations
from typing import Tuple
import tempfile
import os


def download_media_to_temp(url: str) -> Tuple[str, str]:
    """Download media from URL/YouTube to a temporary file.

    Returns (path, content_type_hint)
    content_type_hint is one of: 'video', 'audio'.
    """
    # Try yt_dlp if available
    try:
        import yt_dlp
        temp_dir = os.path.join(tempfile.gettempdir(), "deepfake_ingest")
        os.makedirs(temp_dir, exist_ok=True)
        out_tmplt = os.path.join(temp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            'outtmpl': out_tmplt,
            'format': 'bv*+ba/b',
            'noplaylist': True,
            'quiet': True,
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            # If separate formats were merged, prefer .mp4
            if not os.path.exists(filepath) and filepath.endswith((".webm", ".m4a")):
                alt = os.path.splitext(filepath)[0] + ".mp4"
                if os.path.exists(alt):
                    filepath = alt
            ct = 'video' if info.get('vcodec', 'none') != 'none' else 'audio'
            return filepath, ct
    except Exception:
        pass

    # Fallback: simple wget-like download
    import urllib.request
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    with urllib.request.urlopen(url, timeout=30) as resp, open(path, 'wb') as f:
        f.write(resp.read())
    # Heuristic content type by extension
    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi")):
        return path, 'video'
    if any(lower.endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        return path, 'audio'
    return path, 'video'

