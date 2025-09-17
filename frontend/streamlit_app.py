import os
import requests
import streamlit as st
import base64

API_URL = os.environ.get("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY") or st.secrets.get("API_KEY", "")

st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title("Deepfake Detection System (MVP)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Image", "Video", "Audio", "History & Batch", "URL Ingest"])


def render_result(resp_json):
    st.json({k: v for k, v in resp_json.items() if k not in ("explanations",)})
    exp = resp_json.get("explanations", {})
    if "timeline_png_base64" in exp and exp["timeline_png_base64"]:
        st.subheader("Video Suspicion Timeline")
        st.image(base64.b64decode(exp["timeline_png_base64"]))
    if "spectrogram_png_base64" in exp and exp["spectrogram_png_base64"]:
        st.subheader("Spectrogram")
        st.image(base64.b64decode(exp["spectrogram_png_base64"]))
    with st.expander("Raw explanations"):
        st.json(exp)


with tab1:
    st.header("Image Detection")
    img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if img_file is not None and st.button("Analyze Image"):
        files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.post(f"{API_URL}/detect/image", files=files, headers=headers, timeout=120)
        if resp.ok:
            render_result(resp.json())
        else:
            st.error(resp.text)

with tab2:
    st.header("Video Detection")
    vid_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if vid_file is not None and st.button("Analyze Video"):
        files = {"file": (vid_file.name, vid_file.getvalue(), vid_file.type)}
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.post(f"{API_URL}/detect/video", files=files, headers=headers, timeout=600)
        if resp.ok:
            data = resp.json()
            render_result(data)
            exp = data.get("explanations", {})
            if exp.get("heatmap_overlays"):
                st.subheader("Top Suspicious Overlays")
                cols = st.columns(2)
                for i, item in enumerate(exp["heatmap_overlays"]):
                    img = base64.b64decode(item["overlay_png_base64"]) if item.get("overlay_png_base64") else None
                    with cols[i % 2]:
                        st.caption(f"Frame {item['frame_index']} @ {item['timestamp']}s")
                        if img:
                            st.image(img)
        else:
            st.error(resp.text)

with tab3:
    st.header("Audio Detection")
    aud_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a", "ogg"]) 
    if aud_file is not None and st.button("Analyze Audio"):
        files = {"file": (aud_file.name, aud_file.getvalue(), aud_file.type)}
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.post(f"{API_URL}/detect/audio", files=files, headers=headers, timeout=300)
        if resp.ok:
            render_result(resp.json())
        else:
            st.error(resp.text)

with tab4:
    st.header("Batch Detection")
    files = st.file_uploader("Upload multiple files", type=["png","jpg","jpeg","mp4","avi","mov","mkv","wav","mp3","flac","m4a","ogg"], accept_multiple_files=True)
    if files and st.button("Analyze Batch"):
        mp = []
        for f in files:
            mp.append(("files", (f.name, f.getvalue(), f.type)))
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.post(f"{API_URL}/detect/batch", files=mp, headers=headers, timeout=1200)
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

    st.header("History")
    if st.button("Refresh History"):
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.get(f"{API_URL}/history/list", headers=headers)
        if resp.ok:
            items = resp.json().get("items", [])
            for item in items:
                with st.expander(f"[{item['id']}] {item['modality']} {item['label']} {item['score']:.3f} - {item.get('filename','')}"):
                    detail = requests.get(f"{API_URL}/history/get/{item['id']}", headers=headers)
                    if detail.ok:
                        data = detail.json()
                        st.json(data)
                        st.markdown(f"[Open Report]({API_URL}/report/{item['id']})")
                    else:
                        st.error(detail.text)

with tab5:
    st.header("URL/YouTube Ingestion")
    url = st.text_input("Paste media URL (YouTube, MP4, etc.)")
    if st.button("Analyze URL") and url:
        headers = {"x-api-key": API_KEY} if API_KEY else {}
        resp = requests.post(f"{API_URL}/ingest/url", json={"url": url}, headers=headers, timeout=1200)

# Footer: models status
st.divider()
try:
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    ms = requests.get(f"{API_URL}/models/status", headers=headers, timeout=10)
    if ms.ok:
        st.caption(f"Models: {ms.json()}")
except Exception:
    pass
        if resp.ok:
            render_result(resp.json())
        else:
            st.error(resp.text)

