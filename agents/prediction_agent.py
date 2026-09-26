import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tf_keras as keras

_full_model = None
_fusion_classifier_model = None

CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def _load():
    global _full_model, _fusion_classifier_model
    if _full_model is None:
        _full_model = keras.models.load_model("models/best_model.h5")

        image_features_input = keras.Input(shape=(1024,), name="image_features_in")
        clinical_features_input = keras.Input(shape=(64,), name="clinical_features_in")

        x = _full_model.get_layer("concatenate")([image_features_input, clinical_features_input])
        x = _full_model.get_layer("dense_3")(x)
        x = _full_model.get_layer("dropout")(x, training=False)
        x = _full_model.get_layer("dense_4")(x)

        _fusion_classifier_model = keras.Model(
            inputs=[image_features_input, clinical_features_input],
            outputs=x,
        )
    return _fusion_classifier_model


def run_prediction_agent(image_feature_path: str, clinical_feature_path: str) -> dict:
    model = _load()

    image_features = np.load(image_feature_path)
    clinical_features = np.load(clinical_feature_path)

    image_batch = np.expand_dims(image_features, axis=0)
    clinical_batch = np.expand_dims(clinical_features, axis=0)

    probs = model.predict([image_batch, clinical_batch], verbose=0)[0]

    predicted_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence_score": round(confidence, 4),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
        },
    }