import numpy as np
import pandas as pd

dataset = pd.read_csv("../datasets/Data.csv")
print(dataset)


#[row , column]
X = dataset.iloc[:, :-1].values
print(X)
Y = dataset.iloc[:, 3].values
print(Y)

#handle missing data
# 众数：数据中出现次数最多个数
# 均值：数据的求和平均。
# 中位数：数据排序后的中间数据。
# strategy : string, optional (default="mean") The imputation strategy.
# - If "mean", then replace missing values using the mean along the axis.
# - If "median", then replace missing values using the median along the axis.
# - If "most_frequent", then replace missing using the most frequent value along the axis.



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)





#将某一列改成标签编码特征值
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print(X)

#对数据进行亚编码,创建副本
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print(Y)


#Splitting the datasets into training sets and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


#特征值缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)