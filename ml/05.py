# 공부시간에 따른 자격증 시험 합격 가능성
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LogisticRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

classifier.predict([[6]])
# 결과 1 : 합격할 것으로 예측

classifier.predict([[4]])
classifier.predict_proba([[6]]) # 합격할 확률 출력
# 불합격 확률 14%, 합격 확률 86%

classifier.predict([[4]])
# 결과 0 : 불합격할 것으로 예측
classifier.predict_proba([[4]]) # 합격할 확률 출력
# 불합격 확률 62%, 합격 확률 38%

y_pred = classifier.predict(X_test)
print(y_pred) # 예측 값
print(y_test) # 실제 값 (테스트 세트)