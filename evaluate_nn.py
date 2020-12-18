
# %%
from tcn import TCN
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import Model
import tensorflow as tf
print("Tensorflow - Número de GPUs encontradas: ", len(tf.config.experimental.list_physical_devices('GPU')) )

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from features_extraction import load_data
import pickle
from sklearn.preprocessing import LabelEncoder


# %% Load test data
test_data_path = 'dataset/test_data'
lst = load_data(test_data_path)

# %% scale data
# Array conversion
x_test, y = zip(*lst)
x_test = np.asarray(x_test)
def scale_dataset(x_in, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
    y_out = (x_in - mean)/std
    return y_out, mean, std

f = open('input_preprocess.pckl', 'rb')
mean_in, std_in = pickle.load(f)
x_test = scale_dataset(x_test, mean_in, std_in)[0]


# %% load saved model 
with open("Network/model.json", 'r') as json_file:
    loaded_json = json_file.read()
# loaded_json = open('Network/model.json', 'r').read()
model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

# restore weights
model.load_weights('Network/weights.h5')


# %%
lb = LabelEncoder()
pred = np.round(model.predict(x_test, verbose=1))
print(pred)

predi = pred.argmax(axis=1)
print(labels)
for i, n in enumerate(predi):
        print(labels[n])
# %%
