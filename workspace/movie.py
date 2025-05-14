# 영화추천 시스템
#  1. 인구통계학적 필터링
#  2. 컨텐츠기반 필터링
#  3. 협업 필터링

import pandas as pd
import numpy as np

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
print(df1.head())
print(df2.head(3))
print(df1.shape, df2.shape)
print(df1['title'].equals(df2['title']))
df1.columns = ['id', 'title', 'cast', 'crew']
print(df1[['id', 'cast', 'crew']])
df2 = df2.merge(df1[['id', 'cast', 'crew']], on='id')
print(df2.head(3))

C = df2['vote_average'].mean()
print(C)
m = df2['vote_count'].quantile(0.9)
print(m)
q_movies = df2.copy().loc[df2['vote_count'] >= m]
print(q_movies.shape)
print(q_movies['vote_count'].sort_values())

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R + m / (m + v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
print(q_movies.head(3))
q_movies = q_movies.sort_values('score', ascending=False)
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()