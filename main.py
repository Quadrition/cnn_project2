import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    IMG_WIDTH = 150
    IMG_HEIGHT = 150
    EPOCHS = 30
    BATCH_SIZE = 100

    # Ucitavanje i uklanjanje nula i smoking-a
    df = pd.read_csv("./data/metadata/chest_xray_metadata.csv")
    df = df.fillna(value="Normal")
    df = df[df["Label_1_Virus_category"] != "Stress-Smoking"]

    print(set(df.Label_1_Virus_category.values))

    # Podela na train : test = 80% : 20%
    msk = np.random.rand(len(df)) < 0.8
    traindf = df[msk]
    testdf = df[~msk]

    # Broj uzoraka
    NO_VAL_SAMPLES = int(len(testdf) * 0.20)
    NO_TEST_SAMPLES = int(len(testdf) * 0.80)
    NO_TRAIN_SAMPLES = len(traindf)
    

    train_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
            traindf,
            directory="./data/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical'
            )

    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.20)

    val_generator = test_datagen.flow_from_dataframe(
            testdf,
            directory="./data/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical',
            subset="validation"
            )

    test_generator = test_datagen.flow_from_dataframe(
            testdf,
            directory="./data/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical',
            shuffle=False,
            subset="training"
            )

    optimizer = "adam"
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    # 150x150 x rgb
    model = Sequential([
        Conv2D(32, padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu', kernel_size=(3, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, activation='relu', kernel_size=(3, 3)),
        Conv2D(64, activation='relu', kernel_size=(3, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
        ])
    model.summary()
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # Load prosle weights
    do_load = True

    if do_load:

        model.load_weights("model.h5")

    else:
        
        history = model.fit(
                train_gen,
                epochs=EPOCHS,
                steps_per_epoch=NO_TRAIN_SAMPLES//BATCH_SIZE+1,
                validation_data=val_generator,
                validation_steps=NO_VAL_SAMPLES//BATCH_SIZE+1
                )
        model.save_weights("model.h5")

    model.compile(optimizer=optimizer,
            loss = loss,
            metrics=metrics)

    print("Evaluating...")
    score = model.evaluate(val_generator, batch_size=BATCH_SIZE)
    print("Accuracy: ", score[1], " Loss: ", score[0])

if __name__ == "__main__":
    main()
