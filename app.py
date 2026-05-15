import os
from pathlib import Path
import io
import cv2
import numpy as np
import torch
import torchvision
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from model.faster_rcnn import get_model

# --- UI CONSTANTS ---
PRIMARY_COLOR = "#007BFF"
BG_COLOR = "#F8F9FA"

# --- HELPERS ---
@st.cache_resource
def load_model(weights_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model()
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location=device)
        m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m, device

def detect_image(model, device, img_bgr: np.ndarray, score_thresh: float, nms_iou: float):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.tensor(img_rgb / 255.0).permute(2, 0, 1).float().to(device)
    with torch.no_grad():
        p = model([t])[0]
    
    keep = torchvision.ops.nms(p["boxes"], p["scores"], nms_iou)
    boxes = p["boxes"][keep]
    scores = p["scores"][keep]
    
    # Filter by score threshold
    mask = scores > score_thresh
    final_boxes = boxes[mask]
    final_scores = scores[mask]
    
    count = len(final_boxes)
    annotated_img = img_rgb.copy()
    
    ship_details = []
    for i, (b, s) in enumerate(zip(final_boxes, final_scores)):
        x1, y1, x2, y2 = b.int().tolist()
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"Ship {i+1}: {float(s):.2f}", (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Mock details
        ship_details.append({
            "id": i + 1,
            "confidence": float(s),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1+x2)//2, (y1+y2)//2],
            "area": (x2-x1)*(y2-y1)
        })
        
    return cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR), count, ship_details

def check_collisions(ship_details, min_dist=100):
    collisions = []
    for i in range(len(ship_details)):
        for j in range(i + 1, len(ship_details)):
            c1 = ship_details[i]["center"]
            c2 = ship_details[j]["center"]
            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if dist < min_dist:
                collisions.append((ship_details[i]["id"], ship_details[j]["id"], dist))
    return collisions

def to_png_bytes(img_bgr: np.ndarray):
    _, buf = cv2.imencode(".png", img_bgr)
    return io.BytesIO(buf.tobytes())

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global App Styles */
        .stApp {
            background: linear-gradient(135deg, #f8f9fb 0%, #eef2f7 100%);
            font-family: 'Inter', sans-serif;
            animation: fadeIn 0.8s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e6ed;
            box-shadow: 4px 0 15px rgba(0,0,0,0.03);
        }
        [data-testid="stSidebar"] .stTitle {
            color: #1e293b;
            font-weight: 700;
            font-size: 1.8rem !important;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Modern Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            color: #0f172a !important;
            letter-spacing: -0.02em;
        }
        
        p, span, div {
            font-family: 'Inter', sans-serif;
            color: #475569;
        }

        /* Stat Card System */
        .stat-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #f1f5f9;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: left;
            position: relative;
            overflow: hidden;
        }
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border-color: #3b82f6;
        }
        .stat-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: #3b82f6;
        }
        .stat-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: #1e293b;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .stat-label {
            color: #64748b;
            text-transform: uppercase;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }

        /* Component Styling */
        .stButton>button {
            border-radius: 10px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            border: none !important;
            background: #3b82f6 !important;
            color: white !important;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2) !important;
        }
        .stButton>button:hover {
            background: #2563eb !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3) !important;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            white-space: pre;
            background-color: #ffffff;
            border-radius: 8px 8px 0px 0px;
            color: #64748b;
            font-weight: 500;
            border: 1px solid #e2e8f0;
            border-bottom: none;
            padding: 10px 20px;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #3b82f6;
            background-color: #f8fafc;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #3b82f6 !important;
            border-top: 2px solid #3b82f6 !important;
            font-weight: 600 !important;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            font-weight: 600 !important;
        }
        .streamlit-expanderContent {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
            padding: 1rem !important;
        }

        /* Custom Alert Classes */
        .custom-alert {
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border-left: 6px solid;
            display: flex;
            flex-direction: column;
            gap: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .alert-high { background-color: #fef2f2; border-color: #ef4444; color: #991b1b; }
        .alert-medium { background-color: #fffbeb; border-color: #f59e0b; color: #92400e; }
        .alert-low { background-color: #f0fdf4; border-color: #22c55e; color: #166534; }

        /* Badge Styling */
        .ship-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            background-color: #dbeafe;
            color: #1e40af;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 4px;
            border: 1px solid #bfdbfe;
        }

        /* Divider Polish */
        hr {
            margin: 2rem 0 !important;
            border: 0 !important;
            border-top: 1px solid #e2e8f0 !important;
        }

        /* Chart Container */
        .plotly-graph-div {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #f1f5f9;
        }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    st.set_page_config(
        page_title="MarineEye | Ship Detection Pro",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()

    root = Path(__file__).resolve().parent
    weights_path = (root / "outputs" / "ship_model.pth").resolve()
    test_dir = (root / "test_inputs").resolve()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🚢 MarineEye")
        st.subheader("Control Panel")
        
        score_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        nms_iou = st.slider("Overlap (NMS) IoU", 0.0, 1.0, 0.2, 0.05)
        
        st.divider()
        st.info("MarineEye uses Faster R-CNN for real-time maritime vessel identification and tracking.")
        
        if st.button("Reset Session", use_container_width=True):
            st.rerun()

    # --- MAIN CONTENT ---
    st.title("AI Maritime Surveillance Dashboard")
    
    tabs = st.tabs(["📊 Dashboard", "🔍 Live Detection", "📈 Traffic Analysis", "⚠️ Alerts"])
    
    model, device = load_model(weights_path)

    # --- DASHBOARD TAB ---
    with tabs[0]:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="stat-card"><div class="stat-label">Active Vessels</div><div class="stat-value">24</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-card"><div class="stat-label">Detections Today</div><div class="stat-value">1,402</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card" style="border-left-color: #ef4444;"><div class="stat-label">Potential Risks</div><div class="stat-value" style="color:#ef4444">3</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="stat-card" style="border-left-color: #10b981;"><div class="stat-label">System Health</div><div class="stat-value" style="color:#10b981">100%</div></div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### Recent System Performance")
        # Mock chart
        df = pd.DataFrame({
            'Time': pd.date_range(datetime.now()-timedelta(hours=1), periods=10, freq='6min'),
            'Inference (ms)': np.random.randint(40, 60, 10),
            'Detections': np.random.randint(1, 5, 10)
        })
        fig = px.line(df, x='Time', y='Inference (ms)', title='Inference Latency Trend')
        st.plotly_chart(fig, use_container_width=True)

    # --- LIVE DETECTION TAB ---
    with tabs[1]:
        st.subheader("Real-Time Identification")
        source = st.radio("Source Selector", ["Standard Test Set", "External Upload"], horizontal=True)
        
        if source == "Standard Test Set":
            exts = (".png", ".jpg", ".jpeg")
            imgs = [p for p in sorted(test_dir.iterdir()) if p.suffix.lower() in exts]
            
            if not imgs:
                st.warning("No test images found in 'test_inputs/'")
            else:
                selected_file = st.selectbox("Select Image for Analysis", [p.name for p in imgs])
                img_path = test_dir / selected_file
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    col_img, col_info = st.columns([2, 1])
                    
                    with st.spinner("Processing neural network..."):
                        out, cnt, details = detect_image(model, device, img, score_thresh, nms_iou)
                    
                    with col_img:
                        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Analyzed: {selected_file}")
                    
                    with col_info:
                        st.markdown(f"#### Results for {selected_file}")
                        st.metric("Vessels Found", cnt)
                        
                        if cnt > 0:
                            st.markdown("##### Vessel Details")
                            for d in details:
                                with st.expander(f"🚢 Vessel ID: {d['id']}"):
                                    st.write(f"Confidence: {d['confidence']:.2%}")
                                    st.write(f"Position (Center): {d['center']}")
                                    st.write(f"Size: {d['area']} px²")
                            
                            colls = check_collisions(details)
                            if colls:
                                st.error(f"⚠️ Proximity Alert: {len(colls)} clusters detected!")
                                for c in colls:
                                    st.write(f"- Vessel {c[0]} & {c[1]} (Dist: {c[2]:.1f}px)")
                        
                        st.download_button("Export Results", data=to_png_bytes(out), file_name=f"detected_{selected_file}", mime="image/png")

        else:
            up = st.file_uploader("Drop imagery here", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if up:
                for f in up:
                    data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None: continue
                    
                    with st.container():
                        st.markdown(f"---")
                        c_img, c_det = st.columns([3, 2])
                        out, cnt, details = detect_image(model, device, img, score_thresh, nms_iou)
                        with c_img:
                            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with c_det:
                            st.subheader(f.name)
                            st.metric("Detected", cnt)
                            if cnt > 0:
                                colls = check_collisions(details)
                                if colls: st.warning(f"Collision Risk Detected!")
                            st.download_button("Save", data=to_png_bytes(out), file_name=f"det_{f.name}", key=f.name)

    # --- TRAFFIC ANALYSIS TAB ---
    with tabs[2]:
        st.subheader("Maritime Traffic Intelligence")
        
        # Mock traffic data
        hours = list(range(24))
        traffic = [5, 3, 2, 1, 2, 8, 15, 25, 30, 28, 20, 22, 25, 24, 26, 28, 35, 40, 38, 30, 20, 15, 10, 8]
        traffic_df = pd.DataFrame({"Hour of Day": hours, "Vessel Count": traffic})
        
        fig_traffic = px.bar(traffic_df, x="Hour of Day", y="Vessel Count", 
                             title="Hourly Vessel Density (24h Window)",
                             color="Vessel Count", color_continuous_scale="Blues")
        
        fig_traffic.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter",
            font_color="#475569",
            margin=dict(t=50, b=50, l=50, r=50),
            title_font_size=20,
            title_font_color="#0f172a"
        )
        st.plotly_chart(fig_traffic, use_container_width=True)
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Vessel Size Distribution")
            sizes = np.random.normal(5000, 1500, 100)
            fig_size = px.histogram(sizes, labels={'value':'Area (px²)'}, title="Detected Vessel Sizes")
            fig_size.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Inter",
                font_color="#475569",
                margin=dict(t=50, b=50, l=50, r=50),
                title_font_size=18,
                title_font_color="#0f172a"
            )
            fig_size.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig_size, use_container_width=True)
            
        with c2:
            st.markdown("#### Confidence Score Heatmap")
            conf = np.random.uniform(0.5, 1.0, 50)
            fig_conf = px.box(conf, title="Detection Confidence Distribution")
            fig_conf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="Inter",
                font_color="#475569",
                margin=dict(t=50, b=50, l=50, r=50),
                title_font_size=18,
                title_font_color="#0f172a"
            )
            fig_conf.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig_conf, use_container_width=True)

    # --- ALERTS TAB ---
    with tabs[3]:
        st.subheader("Security & Proximity Alerts")
        alerts = [
            {"time": "10:45 AM", "type": "Proximity", "msg": "Vessel 4 and Vessel 7 are within 50m.", "severity": "high"},
            {"time": "09:12 AM", "type": "Speed", "msg": "Vessel 12 exceeding port speed limit.", "severity": "medium"},
            {"time": "08:05 AM", "type": "System", "msg": "AI Model recalibrated successfully.", "severity": "low"}
        ]
        
        for a in alerts:
            severity_class = f"alert-{a['severity']}"
            st.markdown(f"""
            <div class="custom-alert {severity_class}">
                <small style="color:#666">{a['time']} - {a['type']}</small><br/>
                <strong style="font-size: 1.1rem;">{a['msg']}</strong>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
