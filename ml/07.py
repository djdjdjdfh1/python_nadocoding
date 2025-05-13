import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('QuizData.csv')
print(dataset[:5])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X[:5], y[:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Wedding reception (train)')
plt.xlabel('total')
plt.ylabel('reception')
# plt.show()

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, reg.predict(X_test), color='red')
plt.title('Wedding reception (test)')
plt.xlabel('total')
plt.ylabel('reception')
plt.show()

print(reg.score(X_train, y_train)) # 훈련 세트 평가 점수
print(reg.score(X_test, y_test)) # 테스트 세트 평가 점수

total = 300 # 결혼식 참석 인원
y_pred = reg.predict([[total]])

print(f'결혼식 참석 인원 {total} 명에 대한 예상 식수 인원은 {np.around(y_pred[0]).astype(int)} 명입니다.')