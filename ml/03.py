# Multiple Linear Regression
import pandas as pd
dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)
print(X)
# 1 0 : Home
# 0 1 : Library
# 0 0 : Cafe

# 데이터 세트 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(y_pred, y_test)
print(reg.coef_, reg.intercept_)

# 모델 평가
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))

### 다양한 평가 지표
# MAE (Mean Absolute Error) (실제 값과 예측 값) 차이의 절대값
# MSE (Mean Squared Error) 차이의 제곱
# RMSE (Root Mean Squared Error) 차이의 제곱의 루트
# R2: 결정 계수 , R2는 1에 가까울수록, 나머지는 0에 가까울수록 좋음
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print(mean_absolute_error(y_test, y_pred)) # 실제 값, 예측값 MAE
print(mean_squared_error(y_test, y_pred)) # MSE
# print(mean_squared_error(y_test, y_pred, squared=False)) # RMSE
print(r2_score(y_test, y_pred))