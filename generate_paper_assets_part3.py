import os
import matplotlib.pyplot as plt

os.makedirs("outputs/paper_assets", exist_ok=True)

# These are the real numbers from our earlier testing (Step 11 vs Step 22, plus the
# age-encoding correction in Step 45). Edit these if you re-run and get different values.
rows = [
    ["Fused Model (original, before age fix)", "COVID", "0.5752", "TUBERCULOSIS", "0.4241"],
    ["Split-Agent Pipeline (before age fix)",  "COVID", "0.5752", "TUBERCULOSIS", "0.4241"],
    ["Split-Agent Pipeline (after age fix)",   "COVID", "0.5591", "TUBERCULOSIS", "0.4407"],
]

col_labels = ["Configuration", "Predicted Class", "Confidence", "2nd Class", "2nd Confidence"]

fig, ax = plt.subplots(figsize=(10, 2.2))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.0)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#4472C4")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Highlight the two rows that should match exactly (proof of correct decomposition)
for j in range(len(col_labels)):
    table[1, j].set_facecolor("#D9E2F3")
    table[2, j].set_facecolor("#D9E2F3")

plt.title(
    "Table X. Correctness Validation: Fused Model vs. Split-Agent Pipeline\n"
    "(Rows 1-2 match exactly, confirming agent decomposition preserves the original model's mathematical behavior;\n"
    "Row 3 reflects a subsequent clinical-encoding correction applied identically across the pipeline.)",
    fontsize=9, pad=20
)
plt.tight_layout()
plt.savefig("outputs/paper_assets/correctness_validation_table.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: outputs/paper_assets/correctness_validation_table.png")