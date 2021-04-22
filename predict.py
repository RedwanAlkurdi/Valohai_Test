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

@app.post("/{full_path:path}")
async def predict(image: UploadFile = File(...)):

    img = Image.open(BytesIO(await image.read()))

    # Resize image and convert to grayscale
    img = img.resize((150,150)).convert('RGB')

    img_array = numpy.array(img)

    img_array = img_array * 1./255

    image_data = numpy.expand_dims(img_array, axis=0)


    global loaded_model
    # Check if model is already loaded
    if not loaded_model:
        loaded_model = tf.keras.models.load_model(model_path)

    # Predict with the model
    prediction = loaded_model.predict(image_data)

    p = prediction[0][0]

    if p > 0.5:
        return "prediction: human"
    if p < 0.5:
        return "prediction: horse"


"""
curl -X 'POST' \
  'https://valohai.cloud/evoila/horses-vs-humans/test-europe/20210422.9/Horses_Vs_Humans/scorefile/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@horse1-241.png;type=image/png'"""
