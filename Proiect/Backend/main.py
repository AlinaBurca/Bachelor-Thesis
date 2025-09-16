import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model2 = tf.keras.models.load_model("InceptionV3_3_model.keras")
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_stage_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_expanded /= 255.0
    prediction = model2.predict(img_array_expanded)
    procentage = np.max(prediction) * 100
    print(procentage)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, round(float(procentage), 2)



@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    image_bytes = await image.read()

    with open("received_image.jpg", "wb") as f:
        f.write(image_bytes)
        print("Imagine salvată local ca 'received_image.jpg'")

    result, procentage = predict_stage_from_bytes(image_bytes)
    print(f"Predicție: {result}")
    return {"prediction": result, "procentage": procentage}


