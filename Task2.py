from pandas import Series
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Item 1

dataset = pd.read_csv('Kvalitet_vina.csv')
print("\nPrikaz dataseta: \n")
print(dataset)
print("\nPrikaz statistike dataseta: \n")
print(dataset.describe())
print("\nPrikaz dimenzija dataseta: \n")
print(dataset.shape)
print("\nPrikaz informacija o datasetu: \n")
print(dataset.info())
print("\nBroja targeta po klasama: \n")
print(dataset.groupby('quality').size())

X=dataset.iloc[:,:11]
Y=dataset.iloc[:,11]
pd.plotting.scatter_matrix(X)
plt.show()

column=dataset.columns

for i in range(0,11):
    fig = plt.figure()
    Series.hist(dataset.iloc[:,i])
    plt.title(column[i])
    plt.show()
    fig.savefig('2.1.' + column[i] + '.png')

# Item 2

dataset[dataset['quality'].isin([3,4,7,8])]=0
dataset[dataset['quality'].isin([5])]=1
dataset[dataset['quality'].isin([6])]=2
# dataset[dataset['quality'].isin([7])]=3
print(dataset)
print("\nBroj targeta po klasama nakon sređivanja: \n")
print(dataset.groupby('quality').size())

skup=dataset[dataset['quality'].isin([0,1,2])]
print(skup)
predikt=skup.iloc[:9,:]
y_predikcija=predikt.iloc[:,11]
y_predikcija.index = range(9)

obucavajuci=skup.iloc[9:,:]
X=obucavajuci.iloc[:,0:11].values
Y=obucavajuci.iloc[:,11].values

X_obucavajuci, X_testirajuci, Y_obucavajuci, Y_testirajuci = train_test_split(X, Y, test_size=0.27, random_state= 1)
print(X_obucavajuci.shape, X_testirajuci.shape,Y_obucavajuci.shape,Y_testirajuci.shape)

n_features=X_obucavajuci.shape[1]

# definisanje modela

model = Sequential()
model.add(Dense(4, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_obucavajuci, Y_obucavajuci, epochs=300, batch_size=32, verbose=0)

# evaluacija modela

loss, acc = model.evaluate(X_testirajuci, Y_testirajuci, verbose=0)
print('Tačnost  modela: %.3f' % acc)

# Item 3

import warnings
warnings.filterwarnings("ignore")
model.save('bestmod.model')
nov_model = tf.keras.models.load_model('bestmod.model')
predikcije = nov_model.predict(predikt.iloc[:,:11])

from numpy import argmax
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[0]),'Prava vrednost cifre je: %.3f' % y_predikcija[0])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[1]),'Prava vrednost cifre je: %.3f' % y_predikcija[1])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[2]),'Prava vrednost cifre je: %.3f' % y_predikcija[2])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[3]),'Prava vrednost cifre je: %.3f' % y_predikcija[3])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[4]),'Prava vrednost cifre je: %.3f' % y_predikcija[4])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[5]),'Prava vrednost cifre je: %.3f' % y_predikcija[5])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[6]),'Prava vrednost cifre je: %.3f' % y_predikcija[6])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[7]),'Prava vrednost cifre je: %.3f' % y_predikcija[7])
print('Model je klasifikovao cifru kao: %.3f' % argmax(predikcije[8]),'Prava vrednost cifre je: %.3f' % y_predikcija[8])