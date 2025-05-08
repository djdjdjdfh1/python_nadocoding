# LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')
print('dataset.head:', dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('X,y: ', X,y)

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X,y) # 학습(모델 생성)
y_pred = reg.predict(X)
print('y_pred: ', y_pred)

plt.scatter(X,y, color='blue')
plt.plot(X, y_pred, color='green')
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
# plt.show()

print('9시간 공부했을때 예상 점수: ', reg.predict([[9], [8], [7]]))
print('reg.coef: ', reg.coef_) # 기울기 (m)
print('reg.intercept_', reg.intercept_) # y 절편 (b)
# y = mx + b