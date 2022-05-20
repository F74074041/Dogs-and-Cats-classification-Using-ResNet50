from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np
image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=(image_size[0],image_size[1],3))
# cd python_qtdesigner\project1\Hw2_5
# python hw2_5.py
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btn1_clicked)
        self.ui.pushButton_2.clicked.connect(self.btn2_clicked)
        self.ui.pushButton_3.clicked.connect(self.btn3_clicked)
        self.ui.pushButton_4.clicked.connect(self.btn4_clicked)

    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:2]:
        layer.trainable = False
    for layer in net_final.layers[2:]:
        layer.trainable = True
    net_final.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy', metrics=['accuracy'])
        
    def btn1_clicked(self):
        print(self.net_final.summary())
        
    def btn2_clicked(self):
        img = mpimg.imread("./pic/1.png")
        plt.imshow(img)
        plt.show()

    def btn3_clicked(self):
        model = tf.keras.models.load_model('model.h5')
        s=self.ui.textEdit.toPlainText()
        for images, _ in val_ds.take(1):
            plt.imshow(images[int(s)%10].numpy().astype("uint8"))
            img_array = keras.preprocessing.image.img_to_array(images[int(s)%10])
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            score = predictions[0]
            if(score>0.5):
                plt.title("class:dog")
            else:
                plt.title("class:cat")
            plt.axis("off")
            plt.show()
    
    def btn4_clicked(self):
        data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal"),layers.experimental.preprocessing.RandomRotation(0.1),])
        net2 = ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=(image_size[0],image_size[1],3))
        inputs = keras.Input(shape=(180,180)+(3,))
        x = data_augmentation(inputs)
        x = net2.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        net_final2 = Model(inputs=net2.input, outputs=output_layer)
        for layer in net_final2.layers[:2]:
            layer.trainable = False
        for layer in net_final2.layers[2:]:
            layer.trainable = True
        net_final2.compile(optimizer=Adam(lr=1e-5),loss='binary_crossentropy', metrics=['accuracy'])
        # print(net_final2.summary())
        img = mpimg.imread("./pic/2.png")
        plt.imshow(img)
        plt.show()
            
            
        
        

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())