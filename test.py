# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

def cvimshow(title, img):
    cv2.imshow(title, img)
    key = cv2.waitKey(5000)
    if key == 27:
        cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

## Description:

global_imgformat = (28, 28, 3)
gwidth, gheight, gdepth = global_imgformat

# pre-process the image for classification
image = cv2.resize(image, (gwidth, gheight))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
result = model.predict(image)
print 'Prediction : ', result
result = result[0]
class_ = {"sharingan": 0, "byakugan": 1, "sage": 2, "others": 3}
output = imutils.resize(orig, width=400)
line = 1
for a, b in zip(class_.keys(), result):
    cv2.putText(output, "{}: {:.2f}%".format(a, b * 100), (10, 25+line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    line += 15

# show the output image
cvimshow ("output", output)
