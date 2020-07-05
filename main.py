import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
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
    traindf = df

    testdf = pd.read_csv("./test/chest_xray_test_dataset.csv")
    # Uklanjanje praznih redova
    testdf = testdf[testdf["Label_1_Virus_category"] != "Stress-Smoking"]
    testdf = testdf.dropna(subset=["X_ray_image_name"])
    testdf = testdf.fillna(value={"Label_1_Virus_category" : "Normal"})

    # Broj uzoraka
    NO_TRAIN_SAMPLES = int(len(traindf) * 0.8)
    NO_VAL_SAMPLES = int(len(traindf) * 0.20)
    

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split = 0.2)
    
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_dataframe(
            traindf,
            directory="./data/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical',
            subset="training",
            )

    val_generator = train_datagen.flow_from_dataframe(
            traindf,
            directory="./data/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical',
            subset="validation",
            shuffle=False
            )

    test_generator = test_datagen.flow_from_dataframe(
            testdf,
            directory="./test/test/",
            x_col="X_ray_image_name",
            y_col="Label_1_Virus_category",
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            class_mode='categorical',
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

    # Load postojeci
    load = True

    if load:
        print("Loading model...")
        model=load_model("trainedmodel")
        print("Loading history...")
        historydf = pd.read_json("history.json")
        print(historydf)

    else:
        model.compile(optimizer=optimizer,
                loss = loss,
                metrics=metrics)
        history = model.fit(
                train_generator,
                epochs=EPOCHS,
                steps_per_epoch=NO_TRAIN_SAMPLES//BATCH_SIZE+1,
                validation_data=val_generator,
                validation_steps=NO_VAL_SAMPLES//BATCH_SIZE+1
                )
        model.save("trainedmodel")
        history_df = pd.DataFrame(history.history)
        history_json = "history.json"
        with open(history_json, mode='w') as f:
            history_df.to_json(f)

    print("Evaluating...")
    score = model.evaluate(test_generator, batch_size=BATCH_SIZE)
    print("Accuracy: ", score[1], " Loss: ", score[0])

if __name__ == "__main__":
    main()
