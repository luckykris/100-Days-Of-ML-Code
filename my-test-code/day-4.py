##logistic regression
import numpy as np
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
iris = datasets.load_iris()
print(iris)
X = iris.data[:, :2]  # 使用前两个特征
# print(X)
Y = iris.target
# print(Y)
#np.unique(Y)
#  out: array([0, 1, 2])
#  2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)
print(logreg.predict([[4.9, 3. ]]))


# predict_proba 预测X值属于某列Y的概率 --》分类的概率
prepro = logreg.predict_proba(X_test_std)
print(prepro)
acc = logreg.score(X_test_std,Y_test)
print(acc)
# 预测的十分不准确 ,之后详细学习各种参数的算法
