from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from estimator import estimate_ph

PREVIEW_SIZE = 240


def square_preview(image):
    height, width = image.shape[:2]
    scale = min(PREVIEW_SIZE / width, PREVIEW_SIZE / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((PREVIEW_SIZE, PREVIEW_SIZE, 3), 255, dtype=np.uint8)
    y = (PREVIEW_SIZE - resized.shape[0]) // 2
    x = (PREVIEW_SIZE - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


st.set_page_config(
    page_title="pH Strip Analyzer",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #30383d;
        --muted: #727b80;
        --accent: #68757b;
        --line: #dfe3e5;
        --paper: #f3f4f4;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(93, 105, 111, .07), transparent 30rem),
            var(--paper);
        color: var(--ink);
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    .project-intro {
        margin: 0 0 2rem;
        padding: 1.5rem 1.75rem;
        border: 1px solid #d9dddf;
        border-radius: 14px;
        background: #e8eaeb;
    }
    .project-label {
        margin: 0 0 .45rem;
        color: var(--muted);
        font-size: .76rem;
        font-weight: 650;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .project-intro h1 {
        margin: 0;
        color: #626a6f;
        font-size: clamp(2rem, 4vw, 2.6rem);
        font-weight: 720;
        letter-spacing: -.04em;
        line-height: 1.15;
    }
    .project-intro p {
        max-width: 720px;
        margin: .7rem 0 0;
        color: #737b80;
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-title {
        margin: 1.8rem 0 .25rem;
        color: var(--ink);
        font-size: 1.22rem;
        font-weight: 700;
        letter-spacing: -.02em;
    }
    .section-copy {
        margin: 0 0 1rem;
        color: var(--muted);
        font-size: .92rem;
    }
    .step-heading {
        display: flex;
        align-items: center;
        gap: .55rem;
        min-height: 2.5rem;
        margin: 1.1rem 0 .6rem;
        color: var(--ink);
        font-size: .94rem;
        font-weight: 700;
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.7rem;
        height: 1.7rem;
        flex: 0 0 1.7rem;
        border: 1px solid #d7dddf;
        border-radius: 50%;
        background: #e9eced;
        color: var(--accent);
        font-size: .86rem;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stCameraInput"] section {
        border: 1px dashed #c3cbce;
        border-radius: 16px;
        background: rgba(255,255,255,.76);
    }
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stCameraInput"] p,
    [data-testid="stCameraInput"] span,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] label,
    [data-testid="stAlert"] p {
        color: var(--ink) !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stCameraInput"] button {
        border: 0;
        border-radius: 9px;
        background: var(--accent);
        color: white !important;
    }
    [data-testid="stRadio"] > label {
        color: var(--ink);
        font-weight: 650;
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        gap: .55rem;
    }
    [data-testid="stRadio"] [role="radio"] {
        padding: .45rem .9rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: white;
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: white;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent);
        font-size: 2.35rem;
        font-weight: 750;
    }
    [data-testid="stImage"] img {
        border-radius: 14px;
    }
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="project-intro">
        <div class="project-label">Water quality · Color analysis</div>
        <h1>Automated pH Estimation using Computer Vision</h1>
        <p>Traditional pH testing depends on manual observation and interpretation.
        This computer-vision tool automates test-strip color analysis, reducing manual
        effort by detecting the strip and comparing its colors with reference data.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Different pad colors provide the signal: we isolate the strip, '
    'extract its dominant colors, then estimate pH from the closest reference match.</div>',
    unsafe_allow_html=True,
)
example_columns = st.columns(2, gap="medium")
examples = (
    ("higherph.jpg", "Higher-pH strip example"),
    ("lowerph.jpg", "Lower-pH strip example"),
)
for column, (filename, caption) in zip(example_columns, examples):
    with column:
        st.image(
            str(Path(__file__).resolve().parent / "data" / filename),
            caption=caption,
            width=280,
        )

st.markdown('<div class="section-title">Start an analysis</div>', unsafe_allow_html=True)

source = st.radio("Choose an image source", ["Upload image", "Use camera"], horizontal=True)
if source == "Upload image":
    image_file = st.file_uploader(
        "Choose a test-strip photo",
        type=["jpg", "jpeg", "png"],
        help="For best results, keep the strip in focus against a plain background.",
    )
else:
    image_file = st.camera_input("Take a photo of the test strip")

if image_file is None:
    st.info("Your analysis will appear here after you add an image.")
else:
    img = cv2.imdecode(np.frombuffer(image_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("This image could not be opened. Please try a different JPG or PNG file.")
    else:
        st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Detecting the strip and analyzing its colors..."):
                result = estimate_ph(img)
        except ValueError as error:
            st.error(str(error))
        else:
            st.metric("Estimated pH", f"{result['ph']:.1f}")

            steps = [
                ("Source photo", "Original image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                ("Detected strip", "Detected region of interest", result["roi"]),
                ("Dominant colors", "Colors extracted from the strip", result["dominant_colors"]),
                ("Reference match", "Closest matching color sequence", result["matched_sequence"]),
            ]
            step_columns = st.columns(4, gap="medium")
            for index, ((title, caption, image), column) in enumerate(
                zip(steps, step_columns),
                start=1,
            ):
                with column:
                    st.markdown(
                        f'<div class="step-heading"><span class="step-number">{index}</span>'
                        f'{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(
                        square_preview(image),
                        caption=caption,
                        width=PREVIEW_SIZE,
                    )
