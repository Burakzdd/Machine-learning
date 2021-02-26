#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:07:48 2021
@author: burakzdd
"""
#kütüphaneleri tanımlıyoruz
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
#boston şehrindeki ev fiyatları veri setini alıyoruz load_boston kütüphanesi içinde yüklü geliyor
boston = load_boston()
x, y = boston.data, boston.target
#veri kümesinin %15'ini test verileri olarak çıkarıyoruz
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

#model için en iyi alfa değerini bulmak için birden çok alfa değeri tanımlıyoruz
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
#ElasticNet modelini her bir alfa için ayarlıyor ve x,y verileri ile eğitiyoruz
#Ardından, her alfa için R-kare, MSE ve RMSE ölçümlerini kontrol ediyoruz
for a in alphas:
    model = ElasticNet(alpha=a).fit(x,y)   
    score = model.score(x, y)
    pred_y = model.predict(x)
    mse = mean_squared_error(y, pred_y)   
    print("Alfa={0:.4f}, R2={1:.2f}, MSE={2:.2f}, RMSE={3:.2f}"
       .format(a, score, mse, np.sqrt(mse)))
#elastiknet regresyon modelini yukarıda alınan bir alfa değeri le tanımlıyoruz
elastic=ElasticNet(alpha=0.01).fit(xtrain, ytrain)
ypred = elastic.predict(xtest)
score = elastic.score(xtest, ytest)
mse = mean_squared_error(ytest, ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
      .format(score, mse, np.sqrt(mse)))
#modeli görselleştiriyoruz
x_ax = range(len(xtest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="Orjinal")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="Tahmin")
plt.legend()
plt.show()

# --- ElasticNetCV ----
elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(xtrain, ytrain)
print(model.alpha_)
print(model.intercept_)

ypred = model.predict(xtest)
score = model.score(xtest, ytest)
mse = mean_squared_error(ytest, ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
      .format(score, mse, np.sqrt(mse)))

x_ax = range(len(xtest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="Orjinal")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="Tahmin")
plt.legend()
plt.show()