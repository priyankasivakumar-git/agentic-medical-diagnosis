import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from agents.image_agent import run_image_agent
from agents.clinical_agent import run_clinical_agent
from agents.prediction_agent import run_prediction_agent
from agents.explainability_agent import run_explainability_agent

LABELS_CSV = "dataset_encoded.csv"   # change if needed
CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

os.makedirs("outputs/paper_assets/gradcam_examples", exist_ok=True)

df = pd.read_csv(LABELS_CSV)

# Shuffle so we don't always grab the same rows near the top of the file
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

found = {}   # predicted_class -> (image_path, gradcam_path)

print("Searching for one confidently-correct example per class...")

for i, row in df.iterrows():
    if len(found) == len(CLASS_NAMES):
        break

    image_path = row["image_path"]
    if not os.path.exists(image_path):
        continue

    true_class = CLASS_NAMES[int(row["label"])]
    if true_class in found:
        continue   # already have an example for this class

    try:
        image_result = run_image_agent(image_path)
        clinical_result = run_clinical_agent(
            age=row["age"], gender=row["gender"], fever=bool(row["fever"]), cough=bool(row["cough"])
        )
        prediction = run_prediction_agent(
            image_feature_path=image_result["feature_path"],
            clinical_feature_path=clinical_result["feature_path"],
        )

        # Only keep it if the model predicted correctly AND confidently
        if prediction["predicted_class"] == true_class and prediction["confidence_score"] > 0.9:
            explain_result = run_explainability_agent(
                image_path=image_path,
                image_feature_path=image_result["feature_path"],
                age=row["age"], gender=row["gender"], fever=bool(row["fever"]), cough=bool(row["cough"]),
                output_dir=f"outputs/paper_assets/gradcam_examples"
            )
            # Rename so each class gets its own saved file (otherwise each run overwrites gradcam_result.png)
            saved_path = f"outputs/paper_assets/gradcam_examples/gradcam_{true_class}.png"
            os.replace(explain_result["gradcam_image_path"], saved_path)

            found[true_class] = {
                "gradcam_path": saved_path,
                "confidence": prediction["confidence_score"],
                "image_file": os.path.basename(image_path),
            }
            print(f"  Found {true_class}: {os.path.basename(image_path)} (confidence={prediction['confidence_score']:.3f})")

    except Exception as e:
        continue

if len(found) < len(CLASS_NAMES):
    missing = set(CLASS_NAMES) - set(found.keys())
    print(f"\nWarning: could not find confident examples for: {missing}")
    print("(Try increasing the search range or lowering the 0.9 confidence threshold.)")

# ============================================================
# Assemble the grid figure
# ============================================================
classes_found = [c for c in CLASS_NAMES if c in found]
n = len(classes_found)

fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
if n == 1:
    axes = [axes]

for ax, cls in zip(axes, classes_found):
    info = found[cls]
    img = mpimg.imread(info["gradcam_path"])
    ax.imshow(img)
    ax.set_title(f"{cls}\n(confidence={info['confidence']:.2f})", fontsize=11)
    ax.axis("off")

plt.suptitle("Grad-CAM Visualizations Across Predicted Disease Classes", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/paper_assets/gradcam_variety_grid.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved: outputs/paper_assets/gradcam_variety_grid.png ({n} classes)")