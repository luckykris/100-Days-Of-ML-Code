import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)
print(X_train)
print(Y_train)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


Y_pred = regressor.predict(X_test)

#可视化训练结果,红为训练集,蓝线为预测训练集的Y集合
regressor.predict(X_test)
#可视化查看测试结果
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()