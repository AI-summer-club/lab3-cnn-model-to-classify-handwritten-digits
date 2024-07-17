# Lab 3: Building a CNN Model to Classify Handwritten Digits (MNIST Dataset)

## Introduction

In this lab, we will build a Convolutional Neural Network (CNN) model to classify handwritten digits from the MNIST dataset. The MNIST dataset is a widely used benchmark dataset in the field of computer vision and machine learning, consisting of 60,000 training images and 10,000 testing images of handwritten digits (0-9).

## Notes

- **ChatGPT Encouraged:** Feel free to use ChatGPT during this lab session to ask questions about installation procedures or Python code.
- **Screen Recording for Credit:** Please prepare a short screen recording demonstrating your work before the end of the lab. The lab assistant will review your recording and may ask questions to confirm your understanding and log your credits.

## Prerequisites

Before starting the lab, **IF YOU ARE USING YOUR MAC** make sure you have the following prerequisites installed:
- Python (version 3.6 or higher)
- TensorFlow or PyTorch (deep learning library)
- Numpy, Matplotlib (data manipulation and visualization libraries)

You can install these prerequisites using the following lines:

```bash
# For TensorFlow users
pip install numpy matplotlib tensorflow

# For PyTorch users
pip install numpy matplotlib torch
```

If not, use the **google colab** environment for this lab found here: https://colab.research.google.com/ 
Sign in with your google account and create a new notebook for this lab.

## Dataset Preparation

The MNIST dataset is readily available in many deep learning libraries, including TensorFlow and PyTorch. 

### Data Exploration

Let's start by exploring the dataset:
- Load the training and testing data
- Print the shape of the training and testing data
- Visualize a few sample images from the dataset

```python
# Using TensorFlow, the MNIST dataset can be accessed and loaded by running the lines:
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print shapes of the training and testing datasets using the lines:
print("Training data shape:", x_train.shape)  # (60000, 28, 28) - 60,000 images of size 28x28 pixels
print("Training labels shape:", y_train.shape)  # (60000,) - 60,000 labels (digits 0-9) 
print("Testing data shape:", x_test.shape)  # (10000, 28, 28) - 10,000 images of size 28x28 pixels
print("Testing labels shape:", y_test.shape)  # (10000,) - 10,000 labels (digits 0-9)

# To visualize the sample images from the dataset, run the following lines:
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## Building the CNN Model

CNNs are particularly effective for image classification tasks due to their ability to learn and extract features from the input images. Here's a typical CNN architecture for the MNIST dataset:

- Convolutional Layer: Applies filters to the input image to extract features
- Activation Function (e.g., ReLU): Introduces non-linearity
- Pooling Layer (e.g., Max Pooling): Reduces spatial dimensions
- Flatten Layer: Converts the 2D feature maps into a 1D vector
- Fully Connected Layer: Performs the classification task
- Output Layer (e.g., Softmax): Produces the final class probabilities

You can build this architecture using your preferred deep learning library (TensorFlow or PyTorch).
    
If using TensorFlow, the following lines can be used to build a CNN model:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

model.summary()
```

## Training the Model

Once the model is defined, you can proceed with training:
- Define the loss function (e.g., cross-entropy loss)
- Choose an optimizer (e.g., Adam, SGD)
- Set the number of epochs and batch size
- Train the model on the training data
- Monitor the training and validation accuracy/loss

You can train the model by running the following lines:

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc*100:.2f}%")

import numpy as np

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
correct_indices = np.where(predicted_labels == y_test)[0]
incorrect_indices = np.where(predicted_labels != y_test)[0]

def plot_images(images, labels, predictions, title):
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {labels[i]}\nPredicted: {predictions[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_images(x_test[correct_indices], y_test[correct_indices], predicted_labels[correct_indices], title="Correctly Classified Examples")

plot_images(x_test[incorrect_indices], y_test[incorrect_indices], predicted_labels[incorrect_indices], title="Incorrectly Classified Examples")
```

## Evaluation and Prediction

After training, you can evaluate the model's performance on the testing data:
- Plot the model accuracy and model loss
- Calculate the test accuracy
- Visualize some correctly and incorrectly classified examples
- Make predictions on new handwritten digit images

```python
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
```
