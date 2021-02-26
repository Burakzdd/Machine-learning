#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:37:33 2021

@author: burakzdd
"""

#kutuphaneler tanımlanır
import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

#doğrusal olmayan bir fonksiyon tanımlanır
def f(x):
    return np.exp(3 * x)
x_tr = np.linspace(0., 2, 200)
y_tr = f(x_tr)

#0 ile 2 arasında değerler üretilir
x = np.array([0, .1, .3, .5, .8, 1, 1.3])
y = f(x) + 2 * np.random.randn(len(x))

# Üretilen değerlerden 0-1 arasındaki veriler işaretlenir
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(x_tr, y_tr, '--k')
ax.plot(x, y, '.', ms=10)
ax.set_xlim(0, 1.5)
ax.set_ylim(-10, 80)
ax.set_title('Uretilen model')

#ridge resgresyonu uygulanır
ridge = lm.RidgeCV()

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(x_tr, y_tr, '--k')

for deg, s in zip([2, 4], ['-', '.']):
    ridge.fit(np.vander(x, deg + 1), y)
    y_ridge = ridge.predict(np.vander(x_tr, deg + 1))
    ax.plot(x_tr, y_ridge, s,
            label='derece ' + str(deg))
    ax.legend(loc=2)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-10, 80)
   #modelin katsayilarini bastirma
    print(f'Katsayılar, Derece {deg} için:',
          ' '.join(f'{c:.2f}' for c in ridge.coef_))

ax.plot(x, y, '.', ms=20)
ax.set_title("Ridge Regresyonu")