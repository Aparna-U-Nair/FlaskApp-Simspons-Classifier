import cv2
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from os.path import join, isfile
from os import listdir

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("./model_weights/vgg9.h5")


def draw_test_img(name, pred, input_img, true_label):
    black = [0, 0, 0]
    final_img = cv2.copyMakeBorder(
        input_img, 160, 0, 0, 300, cv2.BORDER_CONSTANT, value=black
    )
    cv2.putText(
        final_img,
        "Predicted: " + pred,
        (20, 60),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        final_img,
        "True: " + true_label,
        (20, 120),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        2,
        (0, 255, 0),
        2,
    )
    plt.savefig(save_path + name + ".png")
    plt.imshow(final_img)
    plt.axis("off")
    plt.show()


def get_rand_img(path, wid, hei):
    """
    Load a random image from the images folder
    """
    print("Inside get_rand_img")
    file_names = [f for f in listdir(path) if isfile(join(path, f))]
    rand_idx = np.random.randint(len(file_names))
    img_name = file_names[rand_idx]
    name = re.split("_\d+", img_name)
    true_label = name[0]
    print(true_label)
    final_path = path + "/" + img_name
    print(f"Final Path: {final_path}")
    return image.load_img(final_path, target_size=(wid, hei)), final_path, true_label


save_path = "./uploads/"
path = "./images/"
width, height = 32, 32
files = []
preds = []
true_lab = []
class_lab = {
    0: "abraham_grampa_simpson",
    1: "apu_nahasapeemapetilon",
    2: "bart_simpson",
    3: "charles_montgomery_burns",
    4: "chief_wiggum",
    5: "comic_book_guy",
    6: "edna_krabappel",
    7: "homer_simpson",
    8: "kent_brockman",
    9: "krusty_the_clown",
    10: "lenny_leonard",
    11: "lisa_simpson",
    12: "marge_simpson",
    13: "mayor_quimby",
    14: "milhouse_van_houten",
    15: "moe_szyslak",
    16: "ned_flanders",
    17: "nelson_muntz",
    18: "principal_skinner",
    19: "sideshow_bob",
}


def save_images():
    for i in range(5):
        img, fin_path, true_label = get_rand_img(path, width, height)
        files.append(fin_path)
        true_lab.append(true_label)
        x = image.img_to_array(img)
        x = x * 1.0 / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        preds.append(classes)

    for i in range(len(files)):
        image = cv2.imread(files[i])
        image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        draw_test_img(i, class_lab[preds[i][0]], image, true_lab[i])


save_images()
