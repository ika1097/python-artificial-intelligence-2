import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Item 1

np.random.seed(0)
imena = ['pca1', 'pca2', 'pca3', 'klase']
dataset = pd.read_csv('Klasteri.csv', names=imena)
a = dataset[dataset['klase'] == 1].values
x1 = a[:, 0]
y1 = a[:, 1]
z1 = a[:, 2]

b = dataset[dataset['klase'] == 2].values
x2 = b[:, 0]
y2 = b[:, 1]
z2 = b[:, 2]

c = dataset[dataset['klase'] == 3].values
x3 = c[:, 0]
y3 = c[:, 1]
z3 = c[:, 2]

d = dataset[dataset['klase'] == 4].values
x4 = d[:, 0]
y4 = d[:, 1]
z4 = d[:, 2]

fig = plt.figure()
plt.scatter(z1, x1, y1, label='I klasa')
plt.scatter(z2, x2, y2, label='II klasa')
plt.scatter(z3, x3, y3, label='III klasa')
plt.scatter(z4, x4, y4, label='IV klasa')
plt.legend()
plt.title('Četiri klase u 2d prostoru')
plt.show()
fig.savefig('1.1.png')

print(len(a))
print(len(b))
print(len(c))
print(len(d))
print("\r\n")

# Item 2

XYZ = dataset.values

m11 = np.average(x1)
m12 = np.average(y1)
m13 = np.average(z1)

m21 = np.average(x2)
m22 = np.average(y2)
m23 = np.average(z2)

m31 = np.average(x3)
m32 = np.average(y3)
m33 = np.average(z3)

m41 = np.average(x4)
m42 = np.average(y4)
m43 = np.average(z4)

M1 = np.array([5.0, 5.0, 6.0])
M2 = np.array([3.5, 2.0, 1.0])
M3 = np.array([6.0, 8.0, 8.0])
M4 = np.array([3.0, 10.0, 4.0])
m11 = M1[0]
m12 = M1[1]
m13 = M1[2]

m21 = M2[0]
m22 = M2[1]
m23 = M2[2]

m31 = M3[0]
m32 = M3[1]
m33 = M3[2]

m41 = M4[0]
m42 = M4[1]
m43 = M4[2]


def najmanje(a, b, c, d):
    min = a
    if min > b: min = b
    if min > c: min = c
    if min > d: min = d
    return min;


a = 0
l = 0
while (a == 0) & (l < 100):
    a = 1
    l = l + 1
    X1 = []
    Y1 = []
    Z1 = []

    X2 = []
    Y2 = []
    Z2 = []

    X3 = []
    Y3 = []
    Z3 = []

    X4 = []
    Y4 = []
    Z4 = []

    d11 = ((x1 - m11) ** 2 + (y1 - m12) ** 2 + (z1 - m13) ** 2) ** 0.5
    d12 = ((x1 - m21) ** 2 + (y1 - m22) ** 2 + (z1 - m23) ** 2) ** 0.5
    d13 = ((x1 - m31) ** 2 + (y1 - m32) ** 2 + (z1 - m33) ** 2) ** 0.5
    d14 = ((x1 - m41) ** 2 + (y1 - m42) ** 2 + (z1 - m43) ** 2) ** 0.5

    d21 = ((x2 - m11) ** 2 + (y2 - m12) ** 2 + (z2 - m13) ** 2) ** 0.5
    d22 = ((x2 - m21) ** 2 + (y2 - m22) ** 2 + (z2 - m23) ** 2) ** 0.5
    d23 = ((x2 - m31) ** 2 + (y2 - m32) ** 2 + (z2 - m33) ** 2) ** 0.5
    d24 = ((x2 - m41) ** 2 + (y2 - m42) ** 2 + (z2 - m43) ** 2) ** 0.5

    d31 = ((x3 - m11) ** 2 + (y3 - m12) ** 2 + (z3 - m13) ** 2) ** 0.5
    d32 = ((x3 - m21) ** 2 + (y3 - m22) ** 2 + (z3 - m23) ** 2) ** 0.5
    d33 = ((x3 - m31) ** 2 + (y3 - m32) ** 2 + (z3 - m33) ** 2) ** 0.5
    d34 = ((x3 - m41) ** 2 + (y3 - m42) ** 2 + (z3 - m43) ** 2) ** 0.5

    d41 = ((x4 - m11) ** 2 + (y4 - m12) ** 2 + (z4 - m13) ** 2) ** 0.5
    d42 = ((x4 - m21) ** 2 + (y4 - m22) ** 2 + (z4 - m23) ** 2) ** 0.5
    d43 = ((x4 - m31) ** 2 + (y4 - m32) ** 2 + (z4 - m33) ** 2) ** 0.5
    d44 = ((x4 - m41) ** 2 + (y4 - m42) ** 2 + (z4 - m43) ** 2) ** 0.5

    print('Pre for')

    for i in range(0, len(x1)):
        min = najmanje(d11[i], d12[i], d13[i], d14[i])
        if d11[i] == min:
            X1.append(x1[i])
            Y1.append(y1[i])
            Z1.append(z1[i])
        elif d12[i] == min:
            X2.append(x1[i])
            Y2.append(y1[i])
            Z2.append(z1[i])
            a = 0
        elif d13[i] == min:
            X3.append(x1[i])
            Y3.append(y1[i])
            Z3.append(z1[i])
            a = 0
        elif d14[i] == min:
            X4.append(x1[i])
            Y4.append(y1[i])
            Z4.append(z1[i])
            a = 0
    for i in range(0, len(x2)):
        min = najmanje(d21[i], d22[i], d23[i], d24[i])
        if d21[i] == min:
            X1.append(x2[i])
            Y1.append(y2[i])
            Z1.append(z2[i])
            a = 0
        elif d22[i] == min:
            X2.append(x2[i])
            Y2.append(y2[i])
            Z2.append(z2[i])
        elif d23[i] == min:
            X3.append(x2[i])
            Y3.append(y2[i])
            Z3.append(z2[i])
            a = 0
        elif d24[i] == min:
            X4.append(x2[i])
            Y4.append(y2[i])
            Z4.append(z2[i])
            a = 0
    for i in range(0, len(x3)):
        min = najmanje(d31[i], d32[i], d33[i], d34[i])
        if d31[i] == min:
            X1.append(x3[i])
            Y1.append(y3[i])
            Z1.append(z3[i])
            a = 0
        elif d32[i] == min:
            X2.append(x3[i])
            Y2.append(y3[i])
            Z2.append(z3[i])
            a = 0
        elif d33[i] == min:
            X3.append(x3[i])
            Y3.append(y3[i])
            Z3.append(z3[i])
        elif d34[i] == min:
            X4.append(x3[i])
            Y4.append(y3[i])
            Z4.append(z3[i])
            a = 0
    for i in range(0, len(x4)):
        min = najmanje(d41[i], d42[i], d43[i], d44[i])
        if d41[i] == min:
            X1.append(x4[i])
            Y1.append(y4[i])
            Z1.append(z4[i])
            a = 0
        elif d42[i] == min:
            X2.append(x4[i])
            Y2.append(y4[i])
            Z2.append(z4[i])
            a = 0
        elif d43[i] == min:
            X3.append(x4[i])
            Y3.append(y4[i])
            Z3.append(z4[i])
            a = 0
        elif d44[i] == min:
            X4.append(x4[i])
            Y4.append(y4[i])
            Z4.append(z4[i])

    del x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4
    print(len(X1))
    print(len(X2))
    print(len(X3))
    print(len(X4))
    x1 = X1
    y1 = Y1
    z1 = Z1

    x2 = X2
    y2 = Y2
    z2 = Z2

    x3 = X3
    y3 = Y3
    z3 = Z3

    x4 = X4
    y4 = Y4
    z4 = Z4

    print("\r\nIspisujemo vrednosti centroida da vidimo kako se pomeraju kroz iteracije\r\n")

    print(m11)
    print(m12)
    print(m13)

    print(m21)
    print(m22)
    print(m23)

    print(m31)
    print(m32)
    print(m33)

    print(m41)
    print(m42)
    print(m43)
    fig = plt.figure()

    plt.scatter(x1, y1, z1, label='I klasa')
    plt.scatter(x2, y2, z2, label='II klasa')
    plt.scatter(x3, y3, z3, label='III klasa')
    plt.scatter(x4, y4, z4, label='IV klasa')
    plt.legend()

    plt.scatter(m11, m12, m13, color='black')
    plt.scatter(m21, m22, m23, color='black')
    plt.scatter(m31, m32, m33, color='black')
    plt.scatter(m41, m42, m43, color='black')
    plt.title('Iteracija broj ' + str(l))
    plt.show()
    fig.savefig('1.2.iteracija' + str(l) + '.png')

    print('\r\nPosle iteracije\r\n')
    m11 = np.average(x1)
    m12 = np.average(y1)
    m13 = np.average(z1)

    m21 = np.average(x2)
    m22 = np.average(y2)
    m23 = np.average(z2)

    m31 = np.average(x3)
    m32 = np.average(y3)
    m33 = np.average(z3)

    m41 = np.average(x4)
    m42 = np.average(y4)
    m43 = np.average(z4)

# Item 4

fig = plt.figure()

M1klast = np.array([m11, m12])
M2klast = np.array([m21, m22])
x = [M1klast, M2klast]
y = ((m22 - m12) / (m21 - m11)) * (x - m11) + m12
srvr = [((m11 + m21) / 2), ((m12 + m22) / 2)]
z = -(((m21 - m11) / (m22 - m12)) * (x - srvr[0])) + srvr[1]
plt.plot(x, y, '-.k', linewidth=3)
plt.plot(x, z, c='k', linewidth=3)

plt.scatter(x1, y1, z1, label='I klasa')
plt.scatter(x2, y2, z2, label='II klasa')
plt.scatter(x3, y3, z3, label='III klasa')
plt.scatter(x4, y4, z4, label='IV klasa')
plt.legend()

M1klast = np.array([m11, m12])
M2klast = np.array([m41, m42])
x = [M1klast, M2klast]
y = ((m42 - m12) / (m41 - m11)) * (x - m11) + m12
srvr = [((m11 + m41) / 2), ((m12 + m42) / 2)]
z = -(((m41 - m11) / (m42 - m12)) * (x - srvr[0])) + srvr[1]
plt.plot(x, y, '-.k', linewidth=3)
plt.plot(x, z, c='k', linewidth=3)

M1klast = np.array([m31, m32])
M2klast = np.array([m41, m42])
x = [M1klast, M2klast]
y = ((m42 - m32) / (m41 - m31)) * (x - m31) + m32
srvr = [((m31 + m41) / 2), ((m32 + m42) / 2)]
z = -(((m41 - m31) / (m42 - m32)) * (x - srvr[0])) + srvr[1]

plt.plot(x, y, '-.k', linewidth=3)
plt.plot(x, z, c='k', linewidth=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Klasifikacione linije')
plt.show()
fig.savefig('1.4.png')

# Item 3

import sklearn
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

# Fitovanje modela
kmeans = kmeans.fit(XYZ)
labels = kmeans.predict(XYZ)

centroidi = kmeans.cluster_centers_

Prvaklasax = []
Prvaklasay = []
Prvaklasaz = []

Drugaklasax = []
Drugaklasay = []
Drugaklasaz = []

Trecaklasax = []
Trecaklasay = []
Trecaklasaz = []

Cetvrtaklasax = []
Cetvrtaklasay = []
Cetvrtaklasaz = []

for i in range(0, len(labels)):
    if (labels[i] == 0):
        Prvaklasax.append(XYZ[i, 0])
        Prvaklasay.append(XYZ[i, 1])
        Prvaklasaz.append(XYZ[i, 2])

    elif (labels[i] == 1):
        Drugaklasax.append(XYZ[i, 0])
        Drugaklasay.append(XYZ[i, 1])
        Drugaklasaz.append(XYZ[i, 2])

    elif (labels[i] == 2):
        Trecaklasax.append(XYZ[i, 0])
        Trecaklasay.append(XYZ[i, 1])
        Trecaklasaz.append(XYZ[i, 2])

    elif (labels[i] == 3):
        Cetvrtaklasax.append(XYZ[i, 0])
        Cetvrtaklasay.append(XYZ[i, 1])
        Cetvrtaklasaz.append(XYZ[i, 2])

fig = plt.figure()

plt.scatter(Prvaklasax, Prvaklasay, Prvaklasaz, label='I klasa')
plt.scatter(Drugaklasax, Drugaklasay, Drugaklasaz, label='II klasa')
plt.scatter(Trecaklasax, Trecaklasay, Trecaklasaz, label='III klasa')
plt.scatter(Cetvrtaklasax, Cetvrtaklasay, Cetvrtaklasaz, label='IV klasa')
plt.legend()

plt.scatter(centroidi[0, 0], centroidi[0, 1], centroidi[0, 2], c='k')
plt.scatter(centroidi[1, 0], centroidi[1, 1], centroidi[1, 2], c='k')
plt.scatter(centroidi[2, 0], centroidi[2, 1], centroidi[2, 2], c='k')
plt.scatter(centroidi[3, 0], centroidi[3, 1], centroidi[3, 2], c='k')

print(centroidi[0, 0], centroidi[0, 1], centroidi[0, 2])
print(centroidi[1, 0], centroidi[1, 1], centroidi[1, 2])
print(centroidi[2, 0], centroidi[2, 1], centroidi[2, 2])
print(centroidi[3, 0], centroidi[3, 1], centroidi[3, 2])
print(m11, m12, m13)
print(m21, m22, m23)
print(m31, m32, m33)
print(m41, m42, m43)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prikaz klasa nakon klasterizacije pomoću Scikit-learn biblioteke')
plt.show()
fig.savefig('1.3.png')
