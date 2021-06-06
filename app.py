import os
from model_prediction import img_pred

from tensorflow.keras.models import load_model

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model("./model_weights/vgg9.h5")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def img_upload():
    if request.method == "POST":
        # get the image and preprocess with the above ftn.
        img_file = request.files["file"]  # file keyword given in index.html

        # save the input img to uploads folder
        upload = os.path.dirname(__file__)  # get the root dir of this folder
        file_path = os.path.join(upload, "uploads", secure_filename=(img_file.name))
        img_file.save(file_path)

        # Call the function to predict
        true, pred = img_pred(file_path, model)
        result = "TRUE: " + true + " and PREDICTION: " + pred
        return result
    return None


if __name__ == "__main__":
    app.run(debug=True)
