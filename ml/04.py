### 3.Polynomial Regression 
# 공부시간에 따른 시험 점수 (우등생)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('PolynomialRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X,y)

# 3-1. 단순 선형 회귀
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y) # 전체 데이터 학습

plt.scatter(X,y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') #선 그래프
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
# plt.show()

print(reg.score(X, y))

# 3-2. 다항 회귀
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2) # 2차 다항식
X_poly = poly_reg.fit_transform(X)
print(X_poly[:5]) # [x] = [x^0, x^1, x^2] -> x가 3이라면 1, 3, 9
print(X[:5])
print(poly_reg.get_feature_names_out())

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 x 와 y로 모델 생성 (학습)

plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
# plt.show()

# X 의 최소값에서 최대값까지의 범위를 0.1단위로 잘라서 데이터 생성
X_range = np.arange(min(X),max(X), 0.1) 
print(X_range)
X_range = X_range.reshape(-1, 1)
X_range.shape

plt.scatter(X, y, color='blue')
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='orange')
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
