import visualkeras
import tensorflow as tf
from PIL import ImageFont
model = tf.keras.models.load_model("CNN_model_nou.keras")

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!

visualkeras.layered_view(
    model,
    legend=True,
    font=font,
    to_file='output.png'
).show()

