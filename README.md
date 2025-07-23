# Mnist-binary-digit-classifier
# A simple neural network using kerbs to classify handwritten digits 0 and 1 from MNIST dataset
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.utils import to_categorical

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# filter only 0 and 1
train_filter = (y_train == 0) | (y_train == 1)
test_filter = (y_test == 0) | (y_test == 1)

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# Normalize pixels(0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

#one hot encode labels: 0 -> [1,0], 1 -> [0,1]
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# build simple NN
model = Sequential([ 
    Flatten(input_shape=(28, 28)),         #convert 28*28 image into 784 vector
    Dense(16, activation='relu'),          #1 hidden layer with 16 neurons
    Dense(2, activation='softmax')        # 2 output classes (0 or 1)
])

# compile model(optimizer + loss + metric)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")






"""
 training step does this under the hood:
	1.	Forward pass:
Compute prediction:
z = w \cdot x + b \quad \rightarrow \quad a = \text{ReLU}(z)
\quad \rightarrow \quad \hat{y} = \text{Softmax}(Wa + b)
	2.	Loss:
Compare prediction with true label using categorical cross-entropy.
	3.	Backpropagation:
Use chain rule to compute:
\frac{dL}{dw}, \frac{dL}{db}
	4.	Update weights using:
w = w - \eta \cdot \frac{dL}{dw}


all this is done by:
model.compile (...)
model.fit(...)
"""

