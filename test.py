import numpy as np
import numpy as np
import cv2
from tensorflow.keras.models import load_model


image_path = 'data/test1/1.jpg'
model_path = 'checkpoints/MobileNet_transfer_learning_002.h5'
image_sz = (256, 256)
Class_label = ['class1', 'class2']

model = load_model(model_path)

image_np = cv2.imread(image_path)

image_np = cv2.resize(image_np, image_sz)
image_np = np.expand_dims(image_np, axis=0)
image_np = image_np / 255
score = model.predict(image_np)[0]
class_id = np.argmax(score)
print(f"class: {Class_label[class_id]} | predicted confidence: {score[class_id]}")
