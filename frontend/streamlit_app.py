import os
import requests
import streamlit as st
import base64

API_URL = os.environ.get("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title("Deepfake Detection System (MVP)")

tab1, tab2, tab3 = st.tabs(["Image", "Video", "Audio"])


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
            render_result(resp.json())
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

