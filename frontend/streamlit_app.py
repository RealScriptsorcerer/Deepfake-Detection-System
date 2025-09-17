import os
import requests
import streamlit as st
import base64

API_URL = os.environ.get("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title("Deepfake Detection System (MVP)")

tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Audio", "History & Batch"])


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
        resp = requests.post(f"{API_URL}/detect/image", files=files, timeout=120)
        if resp.ok:
            render_result(resp.json())
        else:
            st.error(resp.text)

with tab2:
    st.header("Video Detection")
    vid_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if vid_file is not None and st.button("Analyze Video"):
        files = {"file": (vid_file.name, vid_file.getvalue(), vid_file.type)}
        resp = requests.post(f"{API_URL}/detect/video", files=files, timeout=600)
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
        resp = requests.post(f"{API_URL}/detect/audio", files=files, timeout=300)
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
        resp = requests.post(f"{API_URL}/detect/batch", files=mp, timeout=1200)
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

    st.header("History")
    if st.button("Refresh History"):
        resp = requests.get(f"{API_URL}/history/list")
        if resp.ok:
            items = resp.json().get("items", [])
            for item in items:
                with st.expander(f"[{item['id']}] {item['modality']} {item['label']} {item['score']:.3f} - {item.get('filename','')}"):
                    detail = requests.get(f"{API_URL}/history/get/{item['id']}")
                    if detail.ok:
                        st.json(detail.json())
                    else:
                        st.error(detail.text)

