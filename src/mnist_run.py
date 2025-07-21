import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.resource import *

# run mnist model
def run_mnist(selected_optimizer= 'adam', selected_metric = 'accuracy', selected_epochs = 5):
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary, aspect='auto')

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)
        
    
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        plt.xticks([])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
    
    prepare_running()
    train_images, train_labels, test_images, test_labels, class_names = load_data()
    col1, col2 = st.columns((1, 1))

    # Save selected hyperparameters
    selections = f'{selected_optimizer}, {selected_metric}, {selected_epochs}'

    # Normalize the images
    train_images = train_images / 255.0 
    test_images = test_images / 255.0

    # build the model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile the model with the selected optimizer and metric
    model.compile(optimizer=selected_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[selected_metric])
    
    # Train the model with the training data
    model.fit(train_images, train_labels, epochs=selected_epochs) 

    # Evaluate the model with the test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest Loss:', test_loss)
    st.success(f"Completed training with {selected_epochs} epochs, optimizer: {selected_optimizer}, metric: {selected_metric}")

    # Predict the model with the test data
    predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    fig5 = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    fig5.tight_layout()

    img = test_images[0]
    img = np.expand_dims(img, 0)  # Add batch dimension
    predictions_single = model.predict(img)

    fig6 = plt.figure(figsize=(8, 6))
    plot_value_array(0, predictions_single, test_labels)
    fig6.tight_layout()
    _ = plt.xticks(range(10), class_names, rotation=45)
    
    # Save plot if it dosn't exist in SAVE_IMAGES
    if selections not in SAVE_IMAGES:
        SAVE_IMAGES[selections] = [fig5, fig6]
    
    print(SAVE_IMAGES)
    st.success(f"Completed running model")

# Load the MNIST dataset and prepare the data
@st.cache_data
def load_data():
    fmnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fmnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names

@st.cache_resource
def prepare_running():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
            print(len(gpu_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices found.")


def show_data(train_images):
    st.subheader("Training Images")
    fig1 = plt.figure(figsize=(10, 10))
    plt.imshow(train_images[0], aspect='auto')
    plt.colorbar()
    plt.grid(False)
    st.pyplot(fig1)

def show_data_labels(train_images, train_labels, class_names):
    st.subheader("Training Images with Labels")
    fig2 = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary, aspect='auto')
        plt.xlabel(class_names[train_labels[i]])
    st.pyplot(fig2)