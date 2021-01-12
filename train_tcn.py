# train NN to classify speech emotion 

# %% import stuff
# sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Network
from tcn import TCN, compiled_tcn, tcn_full_summary
from keras_one_cycle_clr.keras_one_cycle_clr import CLR, OneCycle

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, model_from_json
# print("Tensorflow - Número de GPUs encontradas: ", len(tf.config.experimental.list_physical_devices('GPU')) )

# utils
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

# %% load dataset
with open('features.pckl', 'rb') as f:
    X, y = pickle.load(f)

# %% Filter inputs and targets
# Split between train and test 
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    shuffle=True,
                                                    random_state=42)
# %% Input normalization
def scale_dataset(x_in, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
    y_out = (x_in - mean)/std
    scaler = MinMaxScaler(feature_range=(0,1))
    y_out = scaler.fit_transform(y_out)
    return y_out, mean, std

x_train, mean_in, std_in = scale_dataset(x_train)
x_test = scale_dataset(x_test, mean_in, std_in)[0]

# save for  inference
with open('Network/input_preprocess.pckl', 'wb') as f:
    pickle.dump([mean_in, std_in], f)

# %% Reshape to keras tensor
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
 
# lb = LabelEncoder()
# y_train = to_categorical(lb.fit_transform(y_train))
# y_test = to_categorical(lb.fit_transform(y_test))
# y_train = np.expand_dims(y_train, axis=2)
# y_test = np.expand_dims(y_test, axis=2)


# %% CLR parameters
batch_size = 16
epochs = 20
clr = CLR(cyc=2,
            lr_range=(0.01, 0.00005),
            momentum_range=(0.85, 0.98),
            record_frq=10)

clr.test_run(epochs)
plt.savefig("Network/LR_schedule.pdf", bbox_inches='tight')


# %% Train TCN
model = compiled_tcn(return_sequences=False,
                    num_feat=x_train.shape[2],
                    num_classes=7,
                    nb_filters=64,
                    kernel_size=8,
                    dilations=[2 ** i for i in range(8)], 
                    nb_stacks=1,
                    dropout_rate=0.25,
                    use_batch_norm=True,
                    max_len=x_train[0:1].shape[1],
                    use_skip_connections=True,
                    use_layer_norm=True,
                    opt='adam')
model.summary()
cnnhistory = model.fit(x_train, y_train,
                        batch_size = batch_size,
                        validation_data=(x_test, y_test),
                        epochs = epochs)


# %% Save it all
# get model as json string and save to file
model_as_json = model.to_json()
with open('Network/model.json', "w") as json_file:
    json_file.write(model_as_json)
# save weights to file (for this format, need h5py installed)
model.save_weights('Network/weights.h5')


# %% Plot accuracy n loss
h = plt.figure()
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()
plt.show()
h.savefig("Network/Loss.pdf", bbox_inches='tight')


h = plt.figure()
plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid()
plt.show()
h.savefig("Network/Accuracy.pdf", bbox_inches='tight')


# %% reload saved model 
# load model from file
loaded_json = open('Network/model.json', 'r').read()
reloaded_model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

# restore weights
reloaded_model.load_weights('Network/weights.h5')


# %% Confusion Matrix
lb = LabelEncoder()
pred = np.round(reloaded_model.predict(x_test, verbose=1))
pred = pred.squeeze().argmax(axis=1)
new_y_test = y_test.astype(int)

mtx = confusion_matrix(new_y_test, pred)
labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
h = plt.figure()
sb.heatmap(mtx, annot = True, fmt ='d',
           yticklabels=labels,
           xticklabels=labels,
           cbar=False)
plt.title('Confusion matrix')
h.savefig("Network/Confusion.pdf", bbox_inches='tight')


# %%
