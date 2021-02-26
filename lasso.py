#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:15:18 2021

@author: burakzdd
"""
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

# Lasso Regresyon
class LassoRegression() : 
	
	def __init__( self, learning_rate, iterations, l1_penality ) : 
		self.learning_rate = learning_rate 
		self.iterations = iterations 
		self.l1_penality = l1_penality 
	# Eğitim modeli fonksiyonu
			
	def fit( self, X, Y ) : 
		self.m, self.n = X.shape 
		# ağırlaklara göre düzenleme
		self.W = np.zeros( self.n ) 
		self.b = 0
		self.X = X 
		self.Y = Y 
		# gradyan iniş öğrenimi
		for i in range( self.iterations ) : 
			self.update_weights() 
		return self
	
	# Gradyan inişte ağırlıkları güncellemek için yardımcı fonksiyon
	def update_weights( self ) : 
		Y_pred = self.predict( self.X ) 
		# gradyanları hesaplama
		dW = np.zeros( self.n ) 
		for j in range( self.n ) : 
			if self.W[j] > 0 : 
				dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) )
						+ self.l1_penality ) / self.m 
			else : 
				dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
						- self.l1_penality ) / self.m 
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
		# ağırlıkları güncelleme
		self.W = self.W - self.learning_rate * dW 
		self.b = self.b - self.learning_rate * db 
		return self
	
	# Varsayımsal fonksiyon h( x ) 
	def predict( self, X ) : 
		return X.dot( self.W ) + self.b 
	
def main() : 
	# Verileri alma
	df = pd.read_csv( "/home/burakzdd/Desktop/makine öğrenmesi/salary_data.csv" )
	X = df.iloc[:, :-1].values 
	Y = df.iloc[:, 1].values 
	
	# Ayrılan veri setinieğitme ve test etme
	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1 / 3, random_state = 0 ) 
	
    # Eğitim modeli 
	model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500 )
	model.fit( X_train, Y_train ) 

	# Test setinde tahmin
	Y_pred = model.predict( X_test ) 
	print( "Öngörülen değerler ", np.round( Y_pred[:3], 2 ) ) 
	print( "Gerçek değerler	 ", Y_test[:3] ) 
	print( "Eğitilen W	 ", round( model.W[0], 2 ) ) 
	print( "Eğitilen b	 ", round( model.b, 2 ) ) 
	
	# Test setini görselleştirme
	plt.scatter( X_test, Y_test, color = 'blue' ) 
	plt.plot( X_test, Y_pred, color = 'orange' ) 
	plt.title( 'Maaş ve Deneyim' )	 
	plt.xlabel( 'Deneyim Senesi' )	 
	plt.ylabel( 'Maaş' )	 
	plt.show() 
	
if __name__ == "__main__" : 
	
	main()
