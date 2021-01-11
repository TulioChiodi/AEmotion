"""
Real time audio emotion recognition 
"""
# %% Import libs
import pyaudio
import numpy as np
import pickle
import librosa

from tensorflow.keras.models import load_model, model_from_json
from tcn import TCN


# %% load saved model 
with open("Network/model.json", 'r') as json_file:
    loaded_json = json_file.read()
# loaded_json = open('Network/model.json', 'r').read()
model = model_from_json(loaded_json, custom_objects={'TCN': TCN})

# restore weights
model.load_weights('Network/weights.h5')


# %% Pre-process input
with open('input_preprocess.pckl', 'rb') as f:
    mean_in, std_in = pickle.load(f)

def scale_dataset(x_in, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
    y_out = (x_in - mean)/std
    return y_out

def input_prep(data):
    global RATE, mean_in, std_in
    # Obtain mfcss
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=RATE,
                                         n_mfcc=40).T, axis=0)
    y = scale_dataset(mfccs, mean_in, std_in)
    return np.expand_dims(y, axis=[0,2])


# %% Identificar dispositivos de audio do sistema
p = pyaudio.PyAudio()

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))



# %% Time streaming
RATE = 44100 # Sample rate
CHUNK = RATE*3 # Frame size

print('janela de análise da RNN é de: {0} segundos'.format(CHUNK/RATE))
#input stream setup
# pyaudio.paInt16 : representa resolução em 16bit 
stream=p.open(format = pyaudio.paInt16,rate=RATE,channels=1, input_device_index = 1, input=True, frames_per_buffer=CHUNK)
# tocador
# player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)
labels = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpresa']
while True:
    data=np.fromstring(stream.read(CHUNK,exception_on_overflow = False),dtype=np.float16)
    data = np.nan_to_num(np.array(data))
    x_infer = input_prep(data)
    pred = np.round(model.predict(x_infer, verbose=0))
    predi = pred.argmax(axis=1)
    print(labels[predi[0]])
# print(max(data))
# player.write(data,CHUNK)

stream.stop_stream()
stream.close()
p.terminate()
# %%
