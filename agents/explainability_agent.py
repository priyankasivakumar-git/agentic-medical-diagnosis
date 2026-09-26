import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras
import cv2
import matplotlib.cm as cm
import shap
import matplotlib.pyplot as plt

from preprocessing.image_preprocess import preprocess_xray
from preprocessing.clinical_preprocess import preprocess_clinical

_model = None
LAST_CONV_LAYER_NAME = "conv5_block16_2_conv"  # last conv layer in DenseNet121

def load_model():
    global _model
    if _model is None:
        _model = keras.models.load_model("models/best_model.h5")
    return _model


def make_gradcam_heatmap(img_batch, clinical_batch, model, last_conv_layer_name, pred_index=None):
    # Temporarily disable the final softmax so gradients don't saturate for
    # confident predictions (a known Grad-CAM limitation) — restored afterward.
    original_activation = model.layers[-1].activation
    model.layers[-1].activation = None

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    import tensorflow as tf
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([img_batch, clinical_batch])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    # Restore original activation so other code (predictions elsewhere) is unaffected
    model.layers[-1].activation = original_activation

    return heatmap.numpy()


CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
_combined_clinical_model = None

def _load_combined_clinical_model():
    """Builds a model that takes [image_features(1024,), raw_clinical(4,)]
    and runs it through the clinical branch + fusion + classifier,
    skipping DenseNet121 entirely (since image features are already computed)."""
    global _combined_clinical_model
    if _combined_clinical_model is not None:
        return _combined_clinical_model

    full_model = load_model()

    image_features_input = keras.Input(shape=(1024,), name="image_features_in")
    clinical_raw_input = keras.Input(shape=(4,), name="clinical_raw_in")

    d1 = full_model.get_layer("dense_1")(clinical_raw_input)
    d2 = full_model.get_layer("dense_2")(d1)
    m1 = full_model.get_layer("multiply_1")([d1, d2])

    x = full_model.get_layer("concatenate")([image_features_input, m1])
    x = full_model.get_layer("dense_3")(x)
    x = full_model.get_layer("dropout")(x, training=False)
    x = full_model.get_layer("dense_4")(x)

    _combined_clinical_model = keras.Model(
        inputs=[image_features_input, clinical_raw_input], outputs=x
    )
    return _combined_clinical_model


def run_shap_explanation(image_feature_path: str, age: float, gender: str, fever: bool, cough: bool, output_dir="outputs") -> dict:
    model = _load_combined_clinical_model()
    image_features = np.load(image_feature_path).astype("float32")

    instance = np.array([preprocess_clinical(age, gender, fever, cough)], dtype="float32")  # shape (1, 4)

    # Get predicted class on the real instance, to explain THAT class specifically
    image_batch_single = np.expand_dims(image_features, axis=0)
    probs = model.predict([image_batch_single, instance], verbose=0)[0]
    target_class_index = int(np.argmax(probs))

    def predict_fn(X):
        n = X.shape[0]
        image_batch = np.repeat(image_batch_single, n, axis=0)
        preds = model.predict([image_batch, X.astype("float32")], verbose=0)
        return preds[:, target_class_index]

    # Small synthetic background of plausible clinical value combinations
    rng = np.random.default_rng(42)
    background = np.array([
        [rng.uniform(0.2, 0.8), float(g), float(f), float(c)]
        for g in [0, 1] for f in [0, 1] for c in [0, 1]
        for _ in range(2)
    ], dtype="float32")

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(instance, nsamples=100, silent=True)
    shap_values = np.array(shap_values).flatten()  # 4 values, one per clinical feature

    feature_names = ["Age", "Gender", "Fever", "Cough"]

    os.makedirs(output_dir, exist_ok=True)
    shap_image_path = os.path.join(output_dir, "shap_result.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_values]
    bars = ax.bar(feature_names, shap_values, color=colors)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"SHAP value (impact on '{CLASS_NAMES[target_class_index]}' prediction)")
    ax.set_title("Clinical Feature Importance (SHAP)")

    # Value labels on each bar
    for bar, val in zip(bars, shap_values):
        offset = 0.00005 if val >= 0 else -0.00005
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:.4f}",
                 ha="center", va=va, fontsize=8)

    # Explicit color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label=f"Increases '{CLASS_NAMES[target_class_index]}' prediction"),
        Patch(facecolor="#3498db", label=f"Decreases '{CLASS_NAMES[target_class_index]}' prediction"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - y_range * 0.05, ymax + y_range * 0.15)

    plt.tight_layout()
    plt.savefig(shap_image_path, dpi=150, bbox_inches="tight")

    return {
        "shap_image_path": shap_image_path,
        "explained_class": CLASS_NAMES[target_class_index],
        "shap_values": {feature_names[i]: round(float(shap_values[i]), 4) for i in range(4)},
    }

def run_explainability_agent(image_path: str, image_feature_path: str, age: float, gender: str, fever: bool, cough: bool, output_dir="outputs") -> dict:
    model = load_model()

    img_array = preprocess_xray(image_path)
    img_batch = np.expand_dims(img_array, axis=0)

    clinical_array = preprocess_clinical(age, gender, fever, cough)
    clinical_batch = np.expand_dims(clinical_array, axis=0)

    # --- Grad-CAM (image explanation) ---
    heatmap = make_gradcam_heatmap(img_batch, clinical_batch, model, LAST_CONV_LAYER_NAME)
    heatmap = heatmap.astype("float32")

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    os.makedirs(output_dir, exist_ok=True)
    gradcam_path = os.path.join(output_dir, "gradcam_result.png")
    cv2.imwrite(gradcam_path, overlay)

    # --- SHAP (clinical explanation) ---
    shap_result = run_shap_explanation(image_feature_path, age, gender, fever, cough, output_dir)

    return {
        "gradcam_image_path": gradcam_path,
        "shap_image_path": shap_result["shap_image_path"],
        "shap_values": shap_result["shap_values"],
        "explanation_summary": (
            "Grad-CAM heatmap generated highlighting the image regions most influential "
            "to the prediction. SHAP values show each clinical feature's contribution "
            f"toward the '{shap_result['explained_class']}' prediction."
        ),
    }