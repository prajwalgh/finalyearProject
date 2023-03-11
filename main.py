import cv2
from keras.models import load_model
import os

model = load_model("model.h5")


classes = {
    4: ("nv", " melanocytic nevi"),
    6: ("mel", "melanoma"),
    2: ("bkl", "benign keratosis-like lesions"),
    1: ("bcc", " basal cell carcinoma"),
    5: ("vasc", " pyogenic granulomas and hemorrhage"),
    0: ("akiec", "Actinic keratoses and intraepithelial carcinomae"),
    3: ("df", "dermatofibroma"),
}
srcdir = ".\h"
count = 0
for temp in os.listdir(srcdir):
    img = cv2.imread(os.path.join(srcdir, temp))
    cv2.imwrite(temp, img)
    img = cv2.resize(img, (28, 28))
    result = model.predict(img.reshape(1, 28, 28, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]
    print(class_name)
    count += 1
    if count > 10:
        break
