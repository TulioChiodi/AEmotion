"""
Real time AEmotion
"""
# %% Import libs
import pyaudio
import numpy as np
import pickle
import librosa
import keract

from tensorflow.keras.models import model_from_json
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime as dtime

# from pahoclass import paho_client

# %% load saved model 
with open("Network/model_en.json", 'r') as json_file:
    loaded_json = json_file.read()
    model = model_from_json(loaded_json, custom_objects={'TCN': TCN})
    # restore weights
    model.load_weights('Network/weights_en.h5')


# %% Pre-process input
with open('Network/input_preprocess_en.pckl', 'rb') as f:
    mean_in, std_in = pickle.load(f)

def input_prep(data, RATE, mean, std):
    # Obtain mfcss
    # normalize input
    data = np.divide(data, np.amax(np.absolute(data)))
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=RATE,
                                         n_mfcc=40).T, axis=0) 
    y = (mfccs - mean)/std
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(np.expand_dims(y, axis=1))
    return np.expand_dims(y, axis=0)


# %% Identificar dispositivos de audio do sistema
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


#  Time streaming #############################################
RATE = 44100 # Sample rate
nn_time = 3 # signal length send to the network
CHUNK = round(RATE*nn_time) # Frame size

print('janela de análise é de: {0} segundos'.format(CHUNK/RATE))
#input stream setup
# pyaudio.paInt16 : representa resolução em 16bit 
stream=p.open(format = pyaudio.paInt16,
                       rate=RATE,
                       channels=1, 
                       input_device_index = 1,
                       input=True,  
                       frames_per_buffer=CHUNK)


# labels = ['Irritação', 'Aversão', 'Medo', 'Alegria', 'Neutro', 'Tristeza', 'Surpresa']
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
history_pred = []
hist_time = []
while True:
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    data = np.nan_to_num(np.array(data))
    x_infer = input_prep(data, RATE, mean_in, std_in)
    pred = np.round(model.predict(x_infer, verbose=0))
    if pred.any() != 0:
        predi = pred.argmax(axis=1)
        history_pred = np.append(history_pred, predi[0])
        # hist_time = np.append(hist_time, dtime.now().strftime('%H:%M:%S'))
        # print(labels[predi[0]] + "  --  (raw data peak: " + str(max(data))+")")
        
        # GET ACTIVATIONS
        layername = 'activation' 
        l_weights = keract.get_activations(model, x_infer, layer_names=layername)
        w_values = np.squeeze(l_weights[layername])

        # SEND TO MQTT BrOKER
        # for k in range(len(labels)):
        #     mqtt_client.publish_single(float(w_values[k]), topic=labels[k])

        # plot
        clear_output(wait=True)
        plt.plot(w_values, 'b-')
        plt.title(labels[predi[0]])
        plt.yticks(ticks=np.arange(0,1.1,0.1))
        plt.xticks(ticks=np.arange(0,7), labels=labels)
        plt.xlabel('Emotion')
        plt.ylabel('NN certainty')
        plt.grid()
        plt.show()  


# %% Plot history 

h=plt.figure()
plt.scatter(range(0,len(history_pred)), history_pred)
plt.yticks(range(0,7) , labels=labels)
# plt.xticks(range(0,len(history_pred)) , labels=hist_time, rotation=90)


plt.xlabel('Time (each dot represents a ' +str(nn_time)+ 's iteration)')
plt.ylabel('Emotion')
plt.title('AEmotion classification')
plt.grid()
plt.show()
h.savefig("Network/hist.pdf", bbox_inches='tight')

