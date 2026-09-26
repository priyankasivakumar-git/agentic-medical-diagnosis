import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/paper_assets", exist_ok=True)

with open("outputs/evaluation_results.json") as f:
    eval_data = json.load(f)

class_names = eval_data["class_names"]
cm = np.array(eval_data["confusion_matrix"])
report = eval_data["classification_report"]
overall_acc = eval_data["accuracy"]

# ============================================================
# 1. Confusion Matrix Heatmap (publication quality)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(class_names, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("Actual Label", fontsize=11)
ax.set_title(f"Confusion Matrix — ChestXAI-MultiPredict Agentic Pipeline\n(Overall Accuracy: {overall_acc*100:.2f}%, n={cm.sum()})", fontsize=12)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = cm[i, j]
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10, fontweight="bold" if i == j else "normal")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Number of Samples", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/paper_assets/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/confusion_matrix.png")

# ============================================================
# 2. Classification Report as a formatted table image
# ============================================================
rows = []
for cls in class_names:
    m = report[cls]
    rows.append([cls, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1-score']:.3f}", int(m['support'])])

rows.append(["Macro Avg", f"{report['macro avg']['precision']:.3f}", f"{report['macro avg']['recall']:.3f}", f"{report['macro avg']['f1-score']:.3f}", int(report['macro avg']['support'])])
rows.append(["Weighted Avg", f"{report['weighted avg']['precision']:.3f}", f"{report['weighted avg']['recall']:.3f}", f"{report['weighted avg']['f1-score']:.3f}", int(report['weighted avg']['support'])])

col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]

fig, ax = plt.subplots(figsize=(7, len(rows) * 0.5 + 1))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(len(rows) - 2, len(rows)):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor("#D9E2F3")
        table[i + 1, j].set_text_props(fontweight="bold")

plt.title(f"Classification Report — ChestXAI-MultiPredict Agentic Pipeline (n={cm.sum()})", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig("outputs/paper_assets/classification_report_table.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/classification_report_table.png")

# ============================================================
# 3. Per-class Precision/Recall/F1 grouped bar chart
# ============================================================
precisions = [report[c]["precision"] for c in class_names]
recalls = [report[c]["recall"] for c in class_names]
f1s = [report[c]["f1-score"] for c in class_names]

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, precisions, width, label="Precision", color="#4472C4")
ax.bar(x, recalls, width, label="Recall", color="#ED7D31")
ax.bar(x + width, f1s, width, label="F1-Score", color="#70AD47")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=20, ha="right")
ax.set_ylim(0.9, 1.02)
ax.set_ylabel("Score")
ax.set_title("Per-Class Precision, Recall, and F1-Score")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

for i, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
    ax.text(i - width, p + 0.003, f"{p:.3f}", ha="center", fontsize=7)
    ax.text(i, r + 0.003, f"{r:.3f}", ha="center", fontsize=7)
    ax.text(i + width, f + 0.003, f"{f:.3f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig("outputs/paper_assets/per_class_metrics_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/per_class_metrics_bar_chart.png")

print("\nAll Part 1 assets generated in outputs/paper_assets/")