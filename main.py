import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()


training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Define class names
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


model=models.load_model('image_classifier.keras')
img = cv.imread('carimages (1).jpeg') 
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.title("Input Image")
plt.show()


img = img / 255.0
prediction = model.predict(np.expand_dims(img, axis=0))  


index = np.argmax(prediction)
print(f'Prediction is: {class_names[index]}')
