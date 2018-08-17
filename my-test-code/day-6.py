import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# print(X)
# print(y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(classifier.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

X_combined_std = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train, y_test))

#  用这个库生成图
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X_combined_std,y=y_combined,
                      clf=classifier)
plt.xlabel('age')
plt.ylabel('salary')
plt.title('logic regression')
plt.show()