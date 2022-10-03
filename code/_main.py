# DATA VARIABLES

DATASET_IMG_ROWS = 512
DATASET_IMG_COLS = 512

RESIZED_IMG_ROWS = 128
RESIZED_IMG_COLS = 128

# DOWNLOAD, LOAD, PREPROCESS, SPLIT THE DATA

from pathlib import Path

imaging_url = "https://kits19.sfo2.digitaloceanspaces.com/"
imaging_name_tmplt = "master_{:05d}.nii.gz"
temp_f = Path(__file__).parent / "temp.tmp"

from _data import download_data, unzip_data, load_data_to_memory, resize_and_preprocess_images

download_data(imaging_url, imaging_name_tmplt, temp_f)
unzip_data()

X_train_tumour, Y_train_tumour = load_data_to_memory(img_row_size=DATASET_IMG_ROWS, 
                                                     img_col_size=DATASET_IMG_COLS,
                                                     range_end=209, step=30)

Xtrain_out, Ytrain_out = resize_and_preprocess_images(X_train_tumour, Y_train_tumour,
                                                      new_rows=RESIZED_IMG_ROWS,
                                                      new_cols=RESIZED_IMG_COLS)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xtrain_out, Ytrain_out, test_size=0.2)

#%%

# MODEL & TRAINING VARIABLES

MODEL_NAME = "xception"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"


# LOAD THE MODELS

import tensorflow.keras.backend as K
from _models import UnetXception, UnetResnet50, dice_coef, dice_loss
from tensorflow.keras.optimizers import Adam, SGD

if OPTIMIZER == "adam":
    opt = Adam(LEARNING_RATE)

elif OPTIMIZER == "sgd":
    opt = SGD(LEARNING_RATE)
    
else:
    raise Exception("no such optimizer option")
    
    
if MODEL_NAME == "resnet50":
    K.clear_session()
    model = UnetResnet50(input_shape=(RESIZED_IMG_ROWS, RESIZED_IMG_COLS, 3))
    model.compile(optimizer=opt, loss=dice_loss, metrics=[dice_coef])
    model.summary()

elif MODEL_NAME == "xception":
    K.clear_session()
    model = UnetXception(input_shape=(RESIZED_IMG_ROWS, RESIZED_IMG_COLS, 3))
    model.compile(optimizer=opt, loss=dice_loss, metrics=[dice_coef])
    model.summary()
    
else:
    raise Exception("no such model option")

#%%

# TRAIN THE MODEL

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_test, y_test))

model.save(f"{MODEL_NAME}.h5")

#%%

# PLOT DICE LOSS AND DICE COEF

import matplotlib.pyplot as plt

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.ylim((0,1.1))
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("dice_coef.png")
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model dice loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("dice_loss.png")

#%%

# TEST THE MODEL

from tensorflow.keras.models import load_model
import numpy as np
import random

model = load_model(f"{MODEL_NAME}.h5", compile=False)

sample_indices = random.sample(range(X_test.shape[0]), 10)
print("sample indices are:", sample_indices)

xs, ys, preds = [], [], []

for i in sample_indices:    
    print(f"predicting {i}")
    pred = model.predict(X_test[i:i+1])
    pred[pred>0.2]=1
    pred[pred!=1]=0
    
    xs.append(X_test[i])
    ys.append(y_test[i])
    preds.append(pred[0])


#%%

# PLOT TEST RESULTS

n_samples = len(sample_indices)

fig, axs = plt.subplots(3, n_samples, figsize=(9,3))

for j in range(n_samples):
    for i in range(3):
        axs[i,j].axis("off")

    axs[0,j].imshow(np.squeeze(xs[j]),cmap="gray")
    axs[1,j].imshow(np.squeeze(ys[j]), cmap="gray")
    axs[2,j].imshow(np.squeeze(preds[j]), cmap="gray")

plt.savefig("preds.png")

