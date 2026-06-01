import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
from src.preprocessing import preprocess
from src.porosity import calculate_porosity
from src.tortuosity import calculate_tortuosity, draw_path_on_image
from src.cpk import calculate_cpk
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.use('Agg')


st.set_page_config(
    page_title="SEM Electrode Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"], p, div, {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
    }

    .stApp {
        background-color: #080B14;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0, 163, 224, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0, 224, 163, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 30% at 60% 40%, rgba(100, 60, 255, 0.06) 0%, transparent 50%);
    }

    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        max-width: 1400px !important;
        padding: 4rem 5rem !important;
        position: relative;
        z-index: 1;
    }
    .main-header {
        font-family: 'Syne', sans-serif !important;
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #00A3E0 50%, #00E0A3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.25rem;
        text-align: center;
    }

    .sub-header {
        font-size: 1.12rem !important;
        color: #8085A4;
        text-align: center;
        letter-spacing: 0.01em;
        margin-bottom: 0.4rem;
        font-weight: 300;
    }

    .made-by {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.88rem !important;
        color: #8085A4;
        text-align: center;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 3.3rem;
    }

    .upload-box {
        border: 2px solid #00A3E0;
        border-radius: 16px;
        padding: 2rem 2rem 0.25rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 163, 224, 0.1);
    }

    .upload-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.28rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 1.25rem;
    }

    .section-header {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.04rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-left: 4px solid #00A3E0;
        padding-left: 0.85rem;
        margin: 2.5rem 0 1.5rem 0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
    }
    .metric-number {
        font-family: 'DM Mono', monospace !important;
        font-size: 2.8rem !important;
        font-weight: 500;
        color: #00A3E0;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.88rem !important;
        color: #5A6080;
        margin-top: 1rem;
        line-height: 1.5;
        text-align: center;
    }

    .image-caption {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.88rem !important;
        color: #5A6080;
        text-align: center !important;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        display: block;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1.2px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.75rem 2rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
        text-align: center !important;
    }
    div[data-testid="stMetric"] label {
        font-family: 'DM Mono', monospace !important;
        font-size: 1.12rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #5A6080 !important;
        display: block !important;
        text-align: center !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.76rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-align: center !important;
    }

    .timer-text {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #00A3E0;
        text-align: right;
        padding-top: 0.75rem;
        letter-spacing: 0.03em;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 3rem 0;
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 11, 20, 0.97) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li {
        font-size: 16px !important;
        color: #CCCCCC !important;
    }

    [data-testid="stSidebar"] h3 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.28rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 1rem !important;
        text-align: center !important;
    }

    [data-testid="stSlider"] {
        margin-top: 1rem !important;
        margin-bottom: 2rem !important;
    }

    [data-testid="stSlider"] label {
        font-size: 16px !important;
        color: #CCCCCC !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        text-align: center !important;
    }

    /* Center slider value number */
    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        text-align: center !important;
    }

    /* Center spec limits text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        text-align: center !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    .stSpinner > div {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        color: #5A6080 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px dashed rgba(0, 163, 224, 0.3) !important;
        border-radius: 12px !important;
        padding: 3rem 2rem !important;
        min-height: 160px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stFileUploaderDropzone"] p {
        font-size: 1.2rem !important;
        color: #8B8FA8 !important;
        text-align: center !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        font-size: 1.04rem !important;
        padding: 0.6rem 2rem !important;
        margin-top: 0.75rem !important;
    }
    [data-testid="stFileUploader"] {
        margin-top: -2rem !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: block !important;}
    section[data-testid="stSidebar"] {
        min-width: 250px !important;
        width: 250px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">SEM Cathode Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Porosity · Tortuosity · Process Capability — for Li-ion battery electrode microstructure</div>', unsafe_allow_html=True)
st.markdown('<div class="made-by">Made by Ali Elsebaie · 2026</div>',
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.markdown("**Try It**")
    st.markdown("<div style='font-size:0.85rem; color:#5A6080; margin-bottom:0.75rem;'>No SEM images? Download a real NMC cathode sample to test the analyzer.</div>", unsafe_allow_html=True)
    try:
        sample_image = open("data/sem7.png", "rb").read()
    except:
        sample_image = open("../data/sem7.png", "rb").read()
    st.download_button(
        label="⬇️ Download Sample Image",
        data=sample_image,
        file_name="sample_nmc_cathode_sem.png",
        mime="image/png",
        use_container_width=True
    )
    st.markdown("---")
    target_porosity = st.slider("Target Porosity (%)", 20, 45, 30)
    tolerance = st.slider("Spec Tolerance (±%)", 1, 10, 5)
    usl = target_porosity + tolerance
    lsl = target_porosity - tolerance
    st.markdown("---")
    st.markdown(f"**USL** &nbsp;`{usl}%` &nbsp;&nbsp; **LSL** &nbsp;`{lsl}%`")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    Analyzes SEM cross-section images of Li-ion battery electrodes using:
    - Otsu thresholding → porosity
    - A\\* pathfinding → tortuosity
    - Cpk statistics → process capability
    """)
    st.markdown("---")

# Uploader in centered box
st.markdown('<div class="upload-box"><div class="upload-title"> ⬇️ Upload Images ⬇️</div><div style="font-size:0.9rem; color:#5A6080; margin-bottom:0.75rem; font-family: DM Mono, monospace;">No images? Download a sample from the sidebar! </div>',
            unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload SEM Images",
    accept_multiple_files=True,
    type=['png', 'jpg', 'tif'],
    help="Upload grayscale SEM cross-section images",
    label_visibility="collapsed"
)
if uploaded_files:
    results = []
    porosity_values = []

    for file in uploaded_files:
        st.markdown(
            f'<div class="section-header">🖼️ {file.name}</div>', unsafe_allow_html=True)

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            st.error(f"Could not read {file.name}")
            continue

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        porosity = calculate_porosity(binary)

        status_col, timer_col = st.columns([4, 1])
        start_time = time.time()

        with status_col:
            with st.spinner("Running A* tortuosity analysis..."):
                tortuosity, path_coords = calculate_tortuosity(binary)

        elapsed = time.time() - start_time
        timer_col.markdown(
            f'<div class="timer-text">⏱ {elapsed:.1f}s</div>',
            unsafe_allow_html=True
        )

        in_spec = lsl <= porosity <= usl
        porosity_values.append(porosity)
        results.append({
            'File': file.name,
            'Porosity (%)': porosity,
            'Tortuosity': tortuosity if tortuosity else 'N/A',
            'In Spec': '✅' if in_spec else '❌'
        })

        st.markdown('<div style="text-align:center; font-size:0.75rem; color:#3A4060; margin-bottom:0.75rem; font-family: DM Mono, monospace; letter-spacing:0.05em;">CLICK IMAGE TO EXPAND · ESC TO CLOSE</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, use_container_width=True)
            st.markdown(
                '<div class="image-caption">Original SEM</div>', unsafe_allow_html=True)
        with col2:
            st.image(binary, use_container_width=True)
            st.markdown('<div class="image-caption">Segmented</div>',
                        unsafe_allow_html=True)
        with col3:
            path_image = draw_path_on_image(binary, path_coords)
            st.image(path_image, use_container_width=True)
            st.markdown(
                '<div class="image-caption">Shortest Li⁺ Path</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Porosity", f"{porosity}%")
        m2.metric("Tortuosity", tortuosity if tortuosity else "N/A")
        m3.metric("Spec", "✅ In Spec" if in_spec else "❌ Out of Spec")

        st.divider()

    if len(porosity_values) >= 2:
        st.markdown(
            '<div class="section-header">📈 Process Capability Report</div>', unsafe_allow_html=True)
        cpk_results = calculate_cpk(
            porosity_values, target=target_porosity, tolerance=tolerance)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Porosity", f"{cpk_results['mean']}%")
        c2.metric("Std Dev", f"{cpk_results['std']}%")
        c3.metric("Cpk", cpk_results['Cpk'])
        c4.metric("Verdict", cpk_results['verdict'])

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0C0F1A')
        ax.set_facecolor('#0C0F1A')
        ax.hist(porosity_values, bins=8, color='#00A3E0',
                edgecolor='#080B14', alpha=0.9, linewidth=2)
        ax.axvline(usl, color='#FF4B4B', linestyle='--',
                   linewidth=2, label=f'USL: {usl}%')
        ax.axvline(lsl, color='#FF4B4B', linestyle='--',
                   linewidth=2, label=f'LSL: {lsl}%')
        ax.axvline(cpk_results['mean'], color='#00E0A3',
                   linestyle='-', linewidth=2.5,
                   label=f"Mean: {cpk_results['mean']}%")
        ax.set_xlabel("Porosity (%)", color='#5A6080', fontsize=13)
        ax.set_ylabel("Count", color='#5A6080', fontsize=13)
        ax.set_title("Porosity Distribution", color='#FFFFFF',
                     fontsize=15, fontweight='bold')
        ax.tick_params(colors='#5A6080', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('#1A1D27')
        ax.legend(facecolor='#0C0F1A', edgecolor='#1A1D27',
                  labelcolor='#FFFFFF', fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">01</div>
            <div class="metric-label">Adjust target porosity and spec tolerance in the sidebar</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">02</div>
            <div class="metric-label">Upload grayscale SEM cross-section images of battery electrodes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">03</div>
            <div class="metric-label">Get porosity, tortuosity, and Cpk process capability report</div>
        </div>""", unsafe_allow_html=True)
