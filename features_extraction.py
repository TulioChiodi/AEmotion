# %% Import audios, convert to mfcc and label it all
import librosa
import os
import numpy as np
import time
import pickle
from sklearn.preprocessing import MinMaxScaler


# %% Load data
def load_data(path):
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X = []
  lst = []
  cnt = 0
  i = -2
  start_time = time.time()
  for subdir, dirs, files in os.walk(path):
    i=i+1
    print(subdir)
    print(i)
    for file in files:
          #Load librosa array, obtain mfcss, add them to array and then to list.
          data, sample_rate = librosa.load(os.path.join(subdir,file),
                                          res_type='kaiser_best')
          
          data = np.squeeze(scaler.fit_transform(np.expand_dims(data, axis=1)))

          mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate,
                                              n_mfcc=40).T, axis=0)
          arr = mfccs, i
          lst.append(arr)
  print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
  return lst


# %% 
if __name__ == "__main__":
  path = 'dataset/archive/Emotions'
  lst = load_data(path)

  ## SAVE DATA ##
  # Array conversion
  X, y = zip(*lst)
  X, y = np.asarray(X), np.asarray(y)
  f = open('features.pckl', 'wb')
  pickle.dump([X, y], f)
  f.close()
  print("All done!")


# %%
