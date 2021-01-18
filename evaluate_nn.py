
# %% Import
from tcn import TCN
from tensorflow.keras.models import model_from_json

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from features_extraction import load_data
import pickle
from sklearn.preprocessing import MinMaxScaler



# %% load saved model 
with open("Network/model.json", 'r') as json_file:
    loaded_json = json_file.read()
# loaded_json = open('Network/model.json', 'r').read()
model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

# restore weights
model.load_weights('Network/weights.h5')


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
    scaler = MinMaxScaler(feature_range=(0,1))
    y_out = scaler.fit_transform(y_out)
    return y_out, mean, std

f = open('Network/input_preprocess.pckl', 'rb')
mean_in, std_in = pickle.load(f)

x_t = scale_dataset(x_test, mean_in, std_in)[0]
x_t = np.expand_dims(x_t, axis=2)


# %% Prediction
pred = np.round(model.predict(x_t, verbose=0))
print(pred)
labels = ['Irritado', 'Nojo', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpresa']
predi = pred.argmax(axis=1)
for i, n in enumerate(predi):
        print("Audio " + str(i) + ": " + labels[n])


# %%
