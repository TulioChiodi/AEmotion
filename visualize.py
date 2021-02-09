# %% Network
from tcn import TCN
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf


# utils
import numpy as np
import matplotlib.pyplot as plt
import keract


# %% 
temporal_conv_net = TCN(nb_filters=128,
                        kernel_size=7,
                        dilations=[2 ** i for i in range(6)], 
                        nb_stacks=1,
                        use_batch_norm=True,
                        use_skip_connections=True,
                        use_layer_norm=False,
                        name = 'temp')

inputs1 = Input(shape=(40, 1))
tcn = temporal_conv_net(inputs1)
dense_end = Dense(7, activation='sigmoid',name='classification')(tcn)

model = Model(inputs=inputs1, outputs=dense_end)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()



# %% 7
x = np.ones(shape=(1, 40, 1))
prefix = 'x_test'
tcn_layer_outputs = list(temporal_conv_net.layers_outputs)

print(keract.get_activations(model, x, layer_names='temp'))


# %%
