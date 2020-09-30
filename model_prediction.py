import numpy as np
import os
import re

# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# model = load_model("./model_weights/vgg9.h5")
# path = "./images/abraham_grampa_simpson_1.jpg" 

class_lab = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: \
        'bart_simpson', 3: 'charles_montgomery_burns', 4: 'chief_wiggum', \
        5: 'comic_book_guy', 6: 'edna_krabappel', 7: 'homer_simpson', \
        8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', \
        11: 'lisa_simpson', 12: 'marge_simpson', 13: 'mayor_quimby', \
        14: 'milhouse_van_houten', 15: 'moe_szyslak', 16: 'ned_flanders', \
        17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}

def img_pred(path,model):
    img = image.load_img(path, target_size=(32,32))
    x = image.img_to_array(img)
    x = x* 1./255
    x = np.expand_dims(x, axis = 0)
    images= np.vstack([x])
    classes = model.predict_classes(images, batch_size=10) #will give the index

    full = os.path.splitext(path)[0]
    img_name = full.split("/")[-1]
    true_lab = re.split("_\d+",img_name)[0]

    return true_lab,class_lab[classes[0]]
