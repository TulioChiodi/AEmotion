# %% Import audios, convert to mfcc and label it all

import librosa
import os
import numpy as np
import time
import pickle

# %% Load data
path = 'dataset/archive/Emotions'

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
        X, sample_rate = librosa.load(os.path.join(subdir,file),
                                      res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                            n_mfcc=40).T, axis=0)

        arr = mfccs, i
        lst.append(arr)


print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

# %% SAVE DATA
# Array conversion
X, y = zip(*lst)
X, y = np.asarray(X), np.asarray(y)
f = open('features.pckl', 'wb')
pickle.dump([X, y], f)
f.close()
print("All done!")
# %%
