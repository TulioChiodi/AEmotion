# train NN to classify speech emotion 

# %% import stuff
import pandas as pd
import numpy as np

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# others
from tcn import TCN, compiled_tcn
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# %% load dataset
df = pd.read_csv('RAVDESS_complete.csv')
df = df.dropna()
# df.isnull().any() # print

# %% Filter inputs and targets
# Split between train and test 
x_train, x_test, y_train, y_test = train_test_split(df.drop(['path','labels','gender','emotion'], axis=1),
                                                    df.labels,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)
# Input normalization
def scale_dataset(x_in, mean=None, std=None):
    print(mean)
    if mean is None or std is None:
        mean = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
    y_out = (x_in - mean)/std
    return y_out, mean, std

x_train, mean_in, std_in = scale_dataset(x_train)
x_test = scale_dataset(x_test, mean_in, std_in)[0]

#  Reshape to keras tensor
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))
y_train = np.expand_dims(y_train, axis=2)
y_test = np.expand_dims(y_test, axis=2)


# %% Train TCN
model = compiled_tcn(return_sequences=False,
                    num_feat=1,
                    num_classes=y_train.shape[1],
                    nb_filters=10,
                    kernel_size=5,
                    dilations=[2 ** i for i in range(9)],
                    nb_stacks=6,
                    max_len=x_train[0:1].shape[1],
                    use_skip_connections=True,
                    opt='rmsprop',
                    lr=5e-4,
                    )

print(f'x_train.shape = {x_train.shape}')
print(f'y_train.shape = {y_train.shape}')
print(f'x_test.shape = {x_test.shape}')
print(f'y_test.shape = {y_test.shape}')

model.summary()
model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
        validation_data=(x_test, y_test.squeeze().argmax(axis=1)))

# %%
