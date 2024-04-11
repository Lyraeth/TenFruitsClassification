import io
import os
import PIL
import json
import keras
import numpy as np
from keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

model = keras.models.load_model("./model/80-20/Model-Skripsi_100.keras")

fruit_names = [
    "Apple",
    "Avocado",
    "Banana",
    "Grape",
    "Guava",
    "Mango",
    "Orange",
    "Rambutan",
    "Salak",
    "Watermelon",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    img = PIL.Image.open(io.BytesIO(file.read())) # type: ignore
    img = img.resize((100, 100))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr /= 255.0

    prediction = model.predict(img_arr) # type: ignore
    predicted_class = fruit_names[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]

    return json.dumps(
        {
            "predicted_class": predicted_class,
            "confidence": "{:.2f}".format(confidence * 100),
        },
        indent=4,
    )


if __name__ == "__main__":
    app.run(debug=True)
