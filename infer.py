import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import sys, os

# Usage: python infer.py path/to/image.png
if len(sys.argv) != 2:
    print('Usage: python infer.py <image_path>')
    sys.exit(1)

img_path = sys.argv[1]
model = models.load_model('bone_fracture_detector.h5')

# Preprocess
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = tf.keras.applications.resnet.preprocess_input(x)
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)[0][0]
label = 'fracture' if pred > 0.5 else 'normal'
print(f'Prediction: {label} (prob={pred:.3f})')
