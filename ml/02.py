### 데이터 세트 분리
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X, len(X)) # 전체 데이터 X 개수
print(X_train, len(X_train)) # 훈련 세트 X
print(X_test, len(X_test)) # 테스트 세트 X 개수
print(y, len(y)) # 전체 데이터 y 개수
print(y_train, len(y_train)) # 훈련 세트 y
print(y_test, len(y_test)) # 테스트 세트 y 개수

### 분리된 데이터를 통한 모델링
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

### 데이터 시각화
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours (train data)')
plt.xlabel('hours')
plt.ylabel('score')
# plt.show()

### 데이터 테스트
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours (test data)')
plt.xlabel('hours')
plt.ylabel('score')
# plt.show()

print('reg.coef: ', reg.coef_) # 기울기 (m)
print('reg.intercept_', reg.intercept_) # y 절편 (b)

### 모델 평가
print(reg.score(X_test, y_test)) # 테스트 세트를 통한 모델 평가
print(reg.score(X_train, y_train)) # 훈련 세트를 통한 모델 평가

### 경사 하강법
from sklearn.linear_model import SGDRegressor
# max_iter = 훈련 세트 반복 횟수 (Epoch 횟수)
# eta0 = 학습률
sr = SGDRegressor(max_iter=500, eta0=1e-4, random_state=0, verbose=1)
sr.fit(X_train, y_train)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, sr.predict(X_train), color='green')
plt.title('Score by hours (train data, SGD)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

print(sr.coef_, sr.intercept_)
print(sr.score(X_test, y_test))
print(sr.score(X_train, y_train))