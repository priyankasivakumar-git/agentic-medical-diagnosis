import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from datetime import datetime

from agents.master_agent import run_pipeline

st.set_page_config(page_title="Chest AI · Multi-Agent", layout="wide", page_icon="🫁")

# ---------- Custom styling ----------
st.markdown("""
<style>
    .agent-card {
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        background: #fafafa;
        text-align: left;
    }
    .agent-card.done { border-color: #2ecc71; background: #f3fdf6; }
    .agent-card.pending { border-color: #dcdcdc; background: #fafafa; opacity: 0.6; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; margin-top: 6px;
    }
    .badge-green { background: #d4f5df; color: #1e8449; }
    .badge-grey { background: #eaeaea; color: #666; }
    .badge-red { background: #fde2e2; color: #c0392b; }
    .top-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "result" not in st.session_state:
    st.session_state.result = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {"image": "pending", "clinical": "pending", "prediction": "pending", "explainability": "pending"}

# ---------- Top bar ----------
col_title, col_badges = st.columns([3, 2])
with col_title:
    st.markdown("### 🫁 Chest Disease Diagnosis — Multi-Agent System")
    st.caption("MASTER AGENT · LLM-ORCHESTRATED · IMAGE + CLINICAL FUSION")
with col_badges:
    st.markdown(
        f"<div style='text-align:right; padding-top:10px;'>"
        f"<span class='badge badge-green'>● LIVE</span>&nbsp;"
        f"<span class='badge badge-grey'>4 AGENTS</span>&nbsp;"
        f"<span class='badge badge-grey'>{datetime.now().strftime('%d %b %Y')}</span>"
        f"</div>", unsafe_allow_html=True
    )

st.divider()

# ---------- Agent status row ----------
agent_meta = {
    "image": ("🖼️ Image Agent", "DenseNet121"),
    "clinical": ("🩺 Clinical Agent", "Dense Embedding"),
    "prediction": ("🧬 Prediction Agent", "AWMF Fusion"),
    "explainability": ("🔍 Explainability", "Grad-CAM + SHAP"),
}

cols = st.columns(4)
for i, (key, (label, sub)) in enumerate(agent_meta.items()):
    status = st.session_state.agent_status[key]
    css_class = "done" if status == "done" else "pending"
    badge_class = "badge-green" if status == "done" else "badge-grey"
    badge_text = "COMPLETE" if status == "done" else "IDLE"
    with cols[i]:
        st.markdown(f"""
        <div class="agent-card {css_class}">
            <b>{label}</b><br>
            <span style="color:#888; font-size:13px;">{sub}</span><br>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------- Layout: sidebar-style input + main content ----------
left, right = st.columns([1, 2.2])

with left:
    st.markdown("#### Patient Intake")
    uploaded_file = st.file_uploader("Chest X-Ray", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)

    age = st.number_input("Age", min_value=0, max_value=120, value=54)
    gender = st.selectbox("Gender", ["male", "female"])
    fever = st.checkbox("Fever", value=True)
    cough = st.checkbox("Cough", value=True)

    run_button = st.button("▶ Run Multi-Agent Diagnosis", type="primary", use_container_width=True)

with right:
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Diagnosis Report", "🩻 Grad-CAM", "📊 SHAP", "📈 Model Performance"])

    if run_button:
        if uploaded_file is None:
            st.error("Please upload a chest X-ray first.")
        else:
            os.makedirs("uploads", exist_ok=True)
            image_path = os.path.join("uploads", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Agents coordinating... (Image → Clinical → Prediction → Explainability)"):
                report_text = run_pipeline(
                    image_path=image_path, age=age, gender=gender, fever=fever, cough=cough,
                )
                st.session_state.result = report_text
                st.session_state.agent_status = {k: "done" for k in st.session_state.agent_status}
            st.rerun()

    with tab1:
        if st.session_state.result:
            st.markdown(st.session_state.result)
        else:
            st.info("Run a diagnosis to see the Master Agent's report here.")

    with tab2:
        if os.path.exists("outputs/gradcam_result.png"):
            st.image("outputs/gradcam_result.png", caption="Grad-CAM — Image Agent Explanation", use_container_width=True)
        else:
            st.info("No Grad-CAM generated yet.")

    with tab3:
        if os.path.exists("outputs/shap_result.png"):
            st.image("outputs/shap_result.png", caption="SHAP — Clinical Feature Importance", use_container_width=True)
        else:
            st.info("No SHAP chart generated yet.")

    with tab4:
        eval_path = "outputs/evaluation_results.json"
        if not os.path.exists(eval_path):
            st.info("Run evaluate_model.py first to generate performance results.")
        else:
            with open(eval_path) as f:
                eval_data = json.load(f)

            class_names = eval_data["class_names"]
            cm = np.array(eval_data["confusion_matrix"])
            report = eval_data["classification_report"]
            overall_acc = eval_data["accuracy"]

            st.markdown(f"### Overall Test Accuracy: **{overall_acc*100:.2f}%**")
            st.caption(f"Evaluated on {int(report['weighted avg']['support'])} test samples")

            # --- Confusion matrix heatmap ---
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    val = cm[i, j]
                    color = "white" if val > cm.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            st.divider()

            # --- Interactive TP/FP/FN/TN breakdown ---
            st.markdown("#### How metrics are calculated from the confusion matrix")
            selected_class = st.selectbox("Choose a class to explain:", class_names)
            idx = class_names.index(selected_class)

            TP = cm[idx, idx]
            FN = cm[idx, :].sum() - TP
            FP = cm[:, idx].sum() - TP
            TN = cm.sum() - TP - FN - FP

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("True Positive (TP)", int(TP))
            c2.metric("False Positive (FP)", int(FP))
            c3.metric("False Negative (FN)", int(FN))
            c4.metric("True Negative (TN)", int(TN))

            st.markdown(f"""
            For **{selected_class}**, reading the confusion matrix:
            - **TP** = {TP}: actual **{selected_class}** cases correctly predicted as **{selected_class}** (diagonal cell).
            - **FP** = {FP}: cases from *other* classes wrongly predicted as **{selected_class}** (sum of the {selected_class} column, excluding the diagonal).
            - **FN** = {FN}: actual **{selected_class}** cases wrongly predicted as something else (sum of the {selected_class} row, excluding the diagonal).
            - **TN** = {TN}: everything else — correctly *not* predicted as {selected_class} when it truly wasn't.
            """)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy_cls = (TP + TN) / (TP + TN + FP + FN)

            st.markdown("#### Formulas, computed live from the numbers above")
            st.latex(r"Precision = \frac{TP}{TP + FP} = \frac{%d}{%d + %d} = %.3f" % (TP, TP, FP, precision))
            st.latex(r"Recall = \frac{TP}{TP + FN} = \frac{%d}{%d + %d} = %.3f" % (TP, TP, FN, recall))
            st.latex(r"F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} = %.3f" % f1)
            st.latex(r"Accuracy = \frac{TP + TN}{TP+TN+FP+FN} = %.3f" % accuracy_cls)

            st.divider()

            # --- Full per-class table ---
            st.markdown("#### All Classes — Summary Table")
            table_rows = []
            for cls in class_names:
                m = report[cls]
                table_rows.append({
                    "Class": cls,
                    "Precision": round(m["precision"], 3),
                    "Recall": round(m["recall"], 3),
                    "F1-score": round(m["f1-score"], 3),
                    "Support": int(m["support"]),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)