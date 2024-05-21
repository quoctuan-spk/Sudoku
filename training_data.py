# Run: python3 Training_MNIST.py

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.layers import MaxPool2D # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
NUM_EPOCHS = 1

def pre_data():
    # Load data from mnist library
    (train_data,train_labels),(test_data,test_labels)=mnist.load_data()

    train_data = train_data.reshape((60000,28,28,1))
    train_data_1d = train_data.astype('float32')/255

    test_data = test_data.reshape((10000,28,28,1))
    test_data_1d = test_data.astype('float32')/255

    train_labels=to_categorical(train_labels,10)
    test_labels=to_categorical(test_labels,10)

    return (train_data_1d,train_labels,test_data_1d,test_labels)

def create_model():
    model = Sequential()

    # Create CNN layer
    model.add(Conv2D(254, kernel_size=(3,3), input_shape=(28,28,1)))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(128, kernel_size=(3,3)))
    model.add(MaxPool2D((2,2)))

    # Convert 2D to 1D, this layer use to format data
    # 28x28 => 784 
    model.add(Flatten())

    # FC layer, layer neural have 140 node)
    model.add(Dense(140, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=10, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train(model, train_data, train_labels, test_data, test_labels, batch_size, weight_path):
    History = model.fit(train_data,train_labels,batch_size=batch_size,epochs=NUM_EPOCHS,validation_data =(test_data,test_labels))
    model.save("weights/model_sudoku_mac.weights.h5")
    return History
    
def draw_plot(History, plot_path):
    N = np.arange(0, NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, History.history["loss"], label="train_loss")
    plt.plot(N, History.history["val_loss"], label="val_loss")
    plt.plot(N, History.history["accuracy"], label="train_acc")
    plt.plot(N, History.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    plt.savefig("dothi/train.png")

def main():
    (train_data_1d,train_labels,test_data_1d,test_labels) = pre_data()
    model = create_model()
    history = train(model,train_data_1d,train_labels,test_data_1d,test_labels, batch_size)
    draw_plot(history)

if __name__ == "__main__":
    main()

