import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras

from preprocessing.clinical_preprocess import preprocess_clinical

_full_model = None
_clinical_feature_model = None

def _load():
    global _full_model, _clinical_feature_model
    if _full_model is None:
        _full_model = keras.models.load_model("models/best_model.h5")
        _clinical_feature_model = keras.models.Model(
            inputs=_full_model.get_layer("input_2").input,
            outputs=[
                _full_model.get_layer("multiply_1").output,  # weighted clinical features (64,)
                _full_model.get_layer("dense_2").output,     # clinical importance gate (1,)
            ],
        )
    return _clinical_feature_model


def run_clinical_agent(age: float, gender: str, fever: bool, cough: bool, output_dir="outputs") -> dict:
    model = _load()

    clinical_array = preprocess_clinical(age, gender, fever, cough)
    clinical_batch = np.expand_dims(clinical_array, axis=0)

    weighted_features, gate_weight = model.predict(clinical_batch, verbose=0)
    weighted_features = weighted_features[0]  # shape (64,)
    gate_weight = float(gate_weight[0][0])

    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, "clinical_features.npy")
    np.save(feature_path, weighted_features)

    return {
        "feature_path": feature_path,
        "feature_vector_length": len(weighted_features),
        "clinical_importance_weight": round(gate_weight, 4),
        "input_values_used": {
            "age": age, "gender": gender, "fever": fever, "cough": cough
        },
    }