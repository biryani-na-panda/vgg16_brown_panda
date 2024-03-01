from keras.applications.vgg16 import VGG16
model = VGG16()

# model.summary(line_length=120)

from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

testimg = load_img("images/test.jpg", target_size=(224,224))
data = img_to_array(testimg)
data = np.expand_dims(data, axis=0)
data = preprocess_input(data)
predicts = model.predict(data)
results = decode_predictions(predicts, top=5)[0]
for r in results:
    name = r[1]
    pct = r[2]
    print(f"これは、「{name}」です。（{pct:.1%})")

plt.imshow(testimg)
plt.show()