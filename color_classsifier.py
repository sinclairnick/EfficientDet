from tensorflow.keras import models, layers
from layers import SpatialPyramidPooling


def create_color_classifier(feature_inputs, num_colors, dropout_rate=0.5):
    spp = SpatialPyramidPooling()
    pyramids = [spp(feature) for feature in feature_inputs]
    final_layer = layers.Concatenate(axis=1)(pyramids)
    final_layer = layers.Dropout(rate=dropout_rate)(final_layer)

    colors = layers.Dense(num_colors, name="colors/out", activation="softmax")(final_layer)

    color_classifier = models.Model(inputs=feature_inputs, outputs=colors, name="color_classifier")

    return color_classifier