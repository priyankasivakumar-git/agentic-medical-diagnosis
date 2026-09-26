import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tf_keras as keras

model = keras.models.load_model("models/best_model.h5")

# DenseNet121's internal layers all start with these prefixes — skip them
skip_prefixes = ("conv1", "conv2", "conv3", "conv4", "conv5", "pool1", "pool2", "pool3", "pool4", "zero_pad")

with open("model_info.txt", "w") as f:
    f.write("=== CUSTOM / TOP-LEVEL LAYERS (non-DenseNet-backbone) ===\n\n")
    for layer in model.layers:
        name = layer.name
        if not any(name.startswith(p) for p in skip_prefixes):
            inbound = []
            try:
                for node in layer._inbound_nodes:
                    for inbound_layer in node.inbound_layers if isinstance(node.inbound_layers, list) else [node.inbound_layers]:
                        inbound.append(inbound_layer.name)
            except Exception:
                pass
            f.write(f"name={name:25s} type={layer.__class__.__name__:20s} out={layer.output_shape}  inputs_from={inbound}\n")

print("Done — open model_info.txt")