#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:35:18 2021

@author: burakzdd
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('/home/burakzdd/Desktop/makine öğrenmesi/ilsicaklik.csv')
iller = data.iloc[:, 1:2].values
sicaklik = data.iloc[:, -1:].values

lr = LinearRegression()

poly = PolynomialFeatures(degree=4)
iller_poly = poly.fit_transform(iller)
lr.fit(iller_poly, sicaklik)
predict = lr.predict(iller_poly)

plt.title('iller ve ortalama sicaklik')
plt.xlabel('iller')
plt.ylabel('sicaklik')
plt.scatter(iller, sicaklik, color='red')
plt.plot(iller,predict, color='blue')
plt.show()
