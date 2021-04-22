from fastapi import FastAPI, File, UploadFile
import subprocess

#def install():
#    subprocess.call(['pip', 'install', "tensorflow=2.2.2"])
#install()
import tensorflow as tf

import numpy
from PIL import Image
from io import BytesIO

app = FastAPI()

model_path = 'my_model.h5'
loaded_model = None
loaded_model = tf.keras.models.load_model(model_path)

@app.post("{full_path:path}")
async def predict(image: UploadFile = File(...)):
    img = Image.open(BytesIO(await image.read()))

    # Resize image and convert to grayscale
    img = img.resize((150,150)).convert('RGB')
    print("one")
    img_array = numpy.array(img)
    print("two")
    img_array = img_array * 1./255
    print("three")
    image_data = numpy.expand_dims(img_array, axis=0)
    print("four")

    global loaded_model
    # Check if model is already loaded
    if not loaded_model:
        loaded_model = tf.keras.models.load_model(model_path)
    print("five")
    # Predict with the model
    prediction = loaded_model.predict(image_data)
    print("six")
    p = prediction[0][0]
    print("seven")
    if p > 0.5:
        return "prediction: human "
    if p < 0.5:
        return "prediction: horse "
