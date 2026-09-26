import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras

from preprocessing.image_preprocess import preprocess_xray

_full_model = None
_image_feature_model = None

def _load():
    global _full_model, _image_feature_model
    if _full_model is None:
        _full_model = keras.models.load_model("models/best_model.h5")
        _image_feature_model = keras.models.Model(
            inputs=_full_model.get_layer("input_1").input,
            outputs=[
                _full_model.get_layer("multiply").output,   # weighted image features (1024,)
                _full_model.get_layer("dense").output,      # image importance gate (1,)
            ],
        )
    return _image_feature_model


def run_image_agent(image_path: str, output_dir="outputs") -> dict:
    model = _load()

    img_array = preprocess_xray(image_path)
    img_batch = np.expand_dims(img_array, axis=0)

    weighted_features, gate_weight = model.predict(img_batch, verbose=0)
    weighted_features = weighted_features[0]  # shape (1024,)
    gate_weight = float(gate_weight[0][0])

    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, "image_features.npy")
    np.save(feature_path, weighted_features)

    return {
        "feature_path": feature_path,
        "feature_vector_length": len(weighted_features),
        "image_importance_weight": round(gate_weight, 4),
        "feature_summary": {
            "mean": round(float(np.mean(weighted_features)), 4),
            "max": round(float(np.max(weighted_features)), 4),
            "nonzero_count": int(np.count_nonzero(weighted_features)),
        },
    }