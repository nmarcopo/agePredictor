from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet',include_top=False)
model.summary()


img_path = 'UTKFace/26_1_4_20170117174327119.jpg.chip.jpg'
img = image.load_img(img_path, target_size=(244,244))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data,axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model.predict(img_data)

print(vgg16_feature.shape)
