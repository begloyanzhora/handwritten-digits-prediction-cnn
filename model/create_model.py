import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
n_samples, nx, ny = x_train.shape
n_samples_test, nx_test, ny_test = x_test.shape

x_train = x_train.reshape((n_samples, nx, ny, 1))
x_train = x_train.astype('float32')
x_train /= 255

y_train = to_categorical(y_train)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1, random_state=2)

datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15)
datagen.fit(x_train)

def get_model():
    cnn = Sequential()
    cnn.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(nx, ny, 1)))
    cnn.add(Conv2D(32, 3, padding='same', activation='relu'))
    cnn.add(MaxPool2D())

    cnn.add(Conv2D(64, 3, padding='same', activation='relu'))
    cnn.add(Conv2D(64, 3, padding='same', activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())

    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))

    cnn.add(Dense(10, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn

callbacks_list = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = 5
        )
]

model = get_model()
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                              epochs = 25, validation_data = (x_val,y_val),
                              steps_per_epoch=len(x_train) / 128,
                              callbacks=callbacks_list)

model.save("model.h5")
