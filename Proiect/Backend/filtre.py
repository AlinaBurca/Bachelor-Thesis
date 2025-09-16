import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import os


full_model = tf.keras.models.load_model("InceptionV3_3_model.keras")

base_model = full_model.layers[0]

print("Base model:", base_model.name)

conv_layers = []
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layers.append(layer)
    if len(conv_layers) == 3:
        break

print("Primele 3 conv layers:", [layer.name for layer in conv_layers])

activation_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[layer.output for layer in conv_layers]
)

img_path = "MildDemented.jpg"
img = image.load_img(img_path, target_size=(299, 299))
x_img = img_to_array(img) / 255.0
x_img = np.expand_dims(x_img, axis=0)

feature_maps = activation_model.predict(x_img)
output_dir = "feature_maps_saved"
os.makedirs(output_dir, exist_ok=True)

for layer_name, feature_map in zip(
        [layer.name for layer in conv_layers],
        feature_maps):

    num_features = min(10, feature_map.shape[-1])
    size = feature_map.shape[1]

    cols = 5
    rows = num_features // cols + (num_features % cols != 0)

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_features):
        fmap = feature_map[0, :, :, i]
        fmap -= fmap.mean()
        fmap /= (fmap.std() + 1e-5)
        fmap *= 64
        fmap += 128
        fmap = np.clip(fmap, 0, 255).astype('uint8')

        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        plt.title(f'Filtru {i}')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{layer_name}_filters.png")
    plt.savefig(save_path)
    print(f"Salvat: {save_path}")

    plt.show()








