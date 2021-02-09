# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:26:40 2021

@author: rdavi

DEMoS and EmoFilm datasets feature extraction
"""

# %% Imports
from features_extraction import load_data
import os 
import pickle
import numpy as np

# %% Categorizar dados em pastas
path = 'dataset/DEMoS/DEMOS/'
# path = 'dataset/EmoFilm/wav_corpus/'
path_out = 'dataset/Italiano'
for subdir, dirs, files in os.walk(path):
    for file in files:
        emotion = file[8:11] # if DEMoS
        if file[0:2] == 'PR':
            if emotion == 'col': # guilt
                path_paste = path_out + '/0 - Guilt/'
            elif emotion == 'dis': # disgust 
                path_paste = path_out + '/1 - Disgust/'
            elif emotion == 'gio': # happy 
                path_paste = path_out + '/2 - Happy/'
            elif emotion == 'pau' or emotion == 'ans': # fear 
                path_paste = path_out + '/3 - Fear/'
            elif emotion == 'rab': # anger
                path_paste = path_out + '/4 - Anger/'
            elif emotion == 'sor': # surprise
                path_paste = path_out + '/5 - Surprise/'
            elif emotion == 'tri': # sadness
                path_paste = path_out + '/6 - Sad/'
            elif emotion == 'neu': # neutral
                path_paste = path_out + '/7 - Neutral/'
            
        # Criar caminho caso não exista
        if not os.path.exists(path_paste):
            os.makedirs(path_paste)
        # Colar arquivos
        os.replace(path + file, path_paste + file)
        
        
# %% Preparar MFCCs
path_out = 'dataset/Italiano'
lst = load_data(path_out)

# Array conversion
X, y = zip(*lst)
X, y = np.asarray(X), np.asarray(y)
f = open('Network/features_it.pckl', 'wb')
pickle.dump([X, y], f)
f.close()
print("All done!")
# %%
