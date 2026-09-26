import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from agents.image_agent import run_image_agent
from agents.clinical_agent import run_clinical_agent
from agents.prediction_agent import run_prediction_agent

os.makedirs("outputs/paper_assets", exist_ok=True)

LABELS_CSV = "dataset_encoded.csv"   # change if your file is named differently
SAMPLE_SIZE = 200                 # how many rows to randomly test

CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

df_full = pd.read_csv(LABELS_CSV)
df = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)), random_state=42).reset_index(drop=True)
print(f"Sampled {len(df)} rows from {LABELS_CSV} for latency/weight analysis")

rows = []
image_times, clinical_times, prediction_times = [], [], []

for i, row in df.iterrows():
    image_path = row["image_path"]
    if not os.path.exists(image_path):
        print(f"Skipping missing file: {image_path}")
        continue

    t0 = time.perf_counter()
    image_result = run_image_agent(image_path)
    t1 = time.perf_counter()

    clinical_result = run_clinical_agent(
        age=row["age"], gender=row["gender"], fever=bool(row["fever"]), cough=bool(row["cough"])
    )
    t2 = time.perf_counter()

    prediction = run_prediction_agent(
        image_feature_path=image_result["feature_path"],
        clinical_feature_path=clinical_result["feature_path"],
    )
    t3 = time.perf_counter()

    image_times.append(t1 - t0)
    clinical_times.append(t2 - t1)
    prediction_times.append(t3 - t2)

    rows.append({
        "image_path": os.path.basename(image_path),
        "true_label": CLASS_NAMES[int(row["label"])],
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence_score"],
        "image_weight": image_result["image_importance_weight"],
        "clinical_weight": clinical_result["clinical_importance_weight"],
        "image_agent_time_s": round(t1 - t0, 3),
        "clinical_agent_time_s": round(t2 - t1, 3),
        "prediction_agent_time_s": round(t3 - t2, 3),
    })

    print(f"[{i+1}/{len(df)}] {os.path.basename(image_path)} -> {prediction['predicted_class']} "
          f"(true: {CLASS_NAMES[int(row['label'])]}, img_w={image_result['image_importance_weight']:.3f}, "
          f"clin_w={clinical_result['clinical_importance_weight']:.3f})")

results_df = pd.DataFrame(rows)
results_df.to_csv("outputs/paper_assets/modality_weights_latency.csv", index=False)
print("\nSaved: outputs/paper_assets/modality_weights_latency.csv")

# ============================================================
# 1. Modality importance weight scatter plot (image vs clinical, per case)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
colors_map = {c: plt.cm.tab10(i) for i, c in enumerate(CLASS_NAMES)}
for cls in CLASS_NAMES:
    subset = results_df[results_df["predicted_class"] == cls]
    if len(subset) > 0:
        ax.scatter(subset["clinical_weight"], subset["image_weight"], label=cls, s=80,
                   color=colors_map[cls], edgecolor="black", linewidth=0.5)

ax.set_xlabel("Clinical Importance Weight (Adaptive Gate)")
ax.set_ylabel("Image Importance Weight (Adaptive Gate)")
ax.set_title(f"AWMF Adaptive Fusion Weights Across {len(results_df)} Test Cases")
ax.legend(title="Predicted Class", fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/paper_assets/modality_weights_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/modality_weights_scatter.png")

# ============================================================
# 2. Modality weight DISTRIBUTION histograms (supports the near-zero finding)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(results_df["image_weight"], bins=20, color="#4472C4", edgecolor="black")
axes[0].set_title("Distribution of Image Importance Weights")
axes[0].set_xlabel("Image Weight (Adaptive Gate)")
axes[0].set_ylabel("Number of Cases")
axes[0].axvline(results_df["image_weight"].mean(), color="red", linestyle="--",
                 label=f"Mean = {results_df['image_weight'].mean():.4f}")
axes[0].legend()

axes[1].hist(results_df["clinical_weight"], bins=20, color="#ED7D31", edgecolor="black")
axes[1].set_title("Distribution of Clinical Importance Weights")
axes[1].set_xlabel("Clinical Weight (Adaptive Gate)")
axes[1].set_ylabel("Number of Cases")
axes[1].axvline(results_df["clinical_weight"].mean(), color="red", linestyle="--",
                 label=f"Mean = {results_df['clinical_weight'].mean():.4f}")
axes[1].legend()

plt.suptitle(f"Modality Weight Distributions Across {len(results_df)} Test Cases", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/paper_assets/modality_weight_distributions.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/modality_weight_distributions.png")

print("\n=== Modality Weight Summary Statistics ===")
print(results_df[["image_weight", "clinical_weight"]].describe())

# ============================================================
# 3. Agent latency bar chart (average time per agent)
# ============================================================
avg_times = {
    "Image Agent\n(DenseNet121)": np.mean(image_times),
    "Clinical Agent\n(Dense Embedding)": np.mean(clinical_times),
    "Prediction Agent\n(AWMF Fusion)": np.mean(prediction_times),
}

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(avg_times.keys(), avg_times.values(), color=["#4472C4", "#ED7D31", "#70AD47"])
for bar, val in zip(bars, avg_times.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}s", ha="center", fontsize=10)

ax.set_ylabel("Average Execution Time (seconds)")
ax.set_title(f"Per-Agent Execution Latency (averaged over {len(results_df)} runs, CPU)")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/paper_assets/agent_latency_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/agent_latency_bar_chart.png")

print(f"\nSummary — Total pipeline avg time (Image+Clinical+Prediction only): "
      f"{np.mean(image_times) + np.mean(clinical_times) + np.mean(prediction_times):.3f}s per case")
print("(Note: Grad-CAM, SHAP, and Master Agent LLM orchestration add additional time on top of this.)")