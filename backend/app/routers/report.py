from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from ..core.storage.db import get_result
from html import escape


router = APIRouter()


@router.get("/report/{result_id}")
def report(result_id: int) -> HTMLResponse:
    res = get_result(result_id)
    if not res:
        raise HTTPException(status_code=404, detail="Not Found")
    payload = res.get("payload", {})
    label = escape(res.get("label", ""))
    score = float(res.get("score", 0.0))
    modality = escape(res.get("modality", ""))
    filename = escape(str(res.get("filename", "")))
    html = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>Deepfake Report #{res['id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            .tag {{ display:inline-block; padding:4px 8px; border-radius:6px; background:#eee; margin-right:6px; }}
            .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
            img {{ max-width:100%; height:auto; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
        </style>
    </head>
    <body>
        <h2>Deepfake Detection Report</h2>
        <div>
            <span class='tag'>ID: {res['id']}</span>
            <span class='tag'>Modality: {modality}</span>
            <span class='tag'>Filename: {filename}</span>
            <span class='tag'>Label: {label}</span>
            <span class='tag'>Score: {score:.3f}</span>
        </div>
        <h3>Summary</h3>
        <div class='grid'>
            <div>
                <h4>Timeline</h4>
                {f"<img src='data:image/png;base64,{payload.get('explanations',{}).get('timeline_png_base64','')}'/>" if payload.get('explanations') else ''}
            </div>
            <div>
                <h4>Top Overlays</h4>
                {''.join([f"<div><b>t={escape(str(o.get('timestamp')))}s</b><br/><img src='data:image/png;base64,{o.get('overlay_png_base64','')}'/></div>" for o in payload.get('explanations',{}).get('heatmap_overlays', [])[:4]]) if payload.get('explanations') else ''}
            </div>
        </div>
        <h3>Details</h3>
        <pre>{escape(str(payload))}</pre>
    </body>
    </html>
    """
    return HTMLResponse(html)

