# Run: python3 Training_MNIST.py

# ----- Thu vien ------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy as np
# Khai bao 

#  -----Data-----
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()


train_data = train_data.reshape((60000,28,28,1))
# Chuyen data ve numpy array [0,1]
train_data = train_data.astype('float32')/255

test_data = test_data.reshape((10000,28,28,1))
# Chuyen data ve numpy array [0,1]
test_data = test_data.astype('float32')/255

train_labels=to_categorical(train_labels,10)
test_labels=to_categorical(test_labels,10)


# -----MODEL------
# Tao model
model = Sequential()
# Tao lop CNN
model.add(Conv2D(254, kernel_size=(3,3), input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128, kernel_size=(3,3)))
model.add(MaxPool2D((2,2)))
# Chuyen 2D thanhf 1D, (lop nay dung dinh dang du lieu)
# 28x28 => 784 
model.add(Flatten())
# Lop FC (lop layer neural co 140 nut)
model.add(Dense(140, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
# Lop output (lop co 10 nut = num_class)
# Moi nut la xac suat cua class 
model.add(Dense(units=10, activation='sigmoid'))

# --- Compile model ---
batch_size = 128
NUM_EPOCHS = 5

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
H = model.fit(train_data,train_labels,batch_size=batch_size,
epochs=NUM_EPOCHS,validation_data =(test_data,test_labels))

# Luu trong so 
model.save("weights/model_sudoku.h5")
model.save_weights("weights/weight_sudoku.hdf5")


# Ve do thi training
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

## luu do thi
plt.savefig("dothi/train.png")
