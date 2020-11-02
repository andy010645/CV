import numpy as np
import sys
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import cifar10 
from keras import regularizers,optimizers
from keras.layers.normalization import BatchNormalization

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_test=x_test/255
x_train=x_train/255
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
np.reshape(x_train,(50000,32,32,3))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def VGG16():
    weight_decay = 0.005
    dropout_rate = 0.5
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32,32,3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(4096,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

def train():
    model=VGG16()
    print(model.summary())
    model.compile(loss=categorical_crossentropy,optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    history=model.fit(x_train,y_train,validation_split=0.33,batch_size=128,epochs=30)
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('accuracy.png')
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig('loss.png')
    model.save('my_model.h5')

    result=model.evaluate(x_test,y_test)
    test=model.evaluate(x_train,y_train)
    print("\nTrain Loss",test[0],"\nTrain ACC",test[1])
    print("\nLoss:",result[0],"\nTest ACC",result[1])

def show_train_images():
    save=[np.random.randint(0,49999)]
    img=[x_train[save[0]]]
    for i in range(1,10):
        save.append(np.random.randint(0,49999))
        img.append(x_train[save[i]])
    imgs=np.hstack(img)
    cv2.imshow("training set",imgs)
    for i in range(10):
        print(list(y_train[save[i]]).index(1))

def show_hyperparameters():
    #print("hyperparameters:\nbatch size: 100\nlearning rate:0.001\noptimizer: SGD")
    model=VGG16()
    print(model.get_config())

def show_model_structure():
    model=VGG16()
    print(model.summary())

def show_accuracy():
    img=cv2.imread('accuracy.png')
    img2=cv2.imread('loss.png')
    cv2.imshow("accuracy",img)
    cv2.imshow("loss",img2)
    
def test_model(x):
    print("載入模型中...")
    model=load_model('my_model.h5')
    img_x=x_test[int(x)].reshape(1,32,32,3)
    #reduce dim
    points=model.predict(img_x)[0]
    print(points)
    plt.figure(figsize=(8,5))
    plt.xlabel('classes')
    plt.ylabel('prob')
    classes=["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
    plt.bar(classes,points)
    plt.show()
    
    
    img = x_test[int(x)]
    img=cv2.resize(img,(128,128))
    cv2.imshow("image",img)
    


    
    

    

