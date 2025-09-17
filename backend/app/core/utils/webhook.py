import os
import json
import urllib.request


WEBHOOK_URL = os.environ.get("DF_WEBHOOK_URL", "").strip()


def notify_high_risk(payload: dict) -> None:
    if not WEBHOOK_URL:
        return
    try:
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

