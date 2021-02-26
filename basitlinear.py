#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:20:58 2021
@author: burakzdd
"""
#kütüphaneler tanımlanır
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#veri seti alınır
dataset = pd.read_csv('/home/burakzdd/Desktop/makine öğrenmesi/basitlinear.csv')
dataset.shape
(25, 2)
dataset.head()
dataset.describe()
#veri setindeki tanım ve görüntü kümesi elemanları ayrıştırılır
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#Basit linear regresyon modeli eğitilir
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#hata değerleri bastırılır
from sklearn import metrics
print('Ortalama Mutlak Hata:', metrics.mean_absolute_error(y_test, y_pred))
print('Kare Ortalama Hata:', metrics.mean_squared_error(y_test, y_pred))
print('Karekök ortalama hata:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#veri setindeki noktalar ve linear regresyon modeli görüntülenir
dataset.plot(x='KisiSayisi', y='YuzdeBasari', style='.',color='green',ms= '10')
plt.title('Kürek Çekme Yarışı Kişi ve Başarı')
plt.xlabel('Kişi Sayisi')
plt.ylabel('Başarı Yüzdesi')
plt.plot(X_test,y_test)
plt.show()
