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
# plt.show()

### 2. 컨텐츠 기반 필터링
print(df2['overview'].head(5))

# 1. TfidfVectorizer (TF-IDF 기반의 벡터화)
# 2. CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# print(ENGLISH_STOP_WORDS)

print(df2['overview'].isnull().values.any())
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
print(indices['The Dark Knight Rises'])
print(df2.iloc[[3]])

# 영화의 제목을 입력받으면 코사인 유사도를 통해 가장 유사도 높은 10개 반환
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]
    

test_idx = indices['The Dark Knight Rises'] # 영화제목을 통해 전체데이터 기준 인덱스 값 얻어옴
print(test_idx)
print(cosine_sim[3])
test_sim_scores = list(enumerate(cosine_sim[3])) # 코사인 유사도 매트릭스에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
test_sim_scores = sorted(test_sim_scores, key=lambda x: x[1], reverse=True)
print(test_sim_scores[1:11])

test_movie_indices = [i[0] for i in test_sim_scores[1:11]]
df2['title'].iloc[test_movie_indices]

print('추천영화', get_recommendations('Avatar'))

### 다양한 요소 기반 추천 (장르, 감독, 키워드)
df2.head(3)
# print(df2.loc[0, 'genres'])

from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
    print(df2[feature])
df2.loc[0, 'crew']

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

df2['director'] = df2['crew'].apply(get_director)
# print(df2['director'].isnull())
df2.loc[0, 'cast']
df2.loc[0, 'genres']

# 처음 3개의 데이터중에ㅓㅅ name에 해당하는 value만 추출
def get_list(x):
    if isinstance(x,list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

print(df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:   
        if isinstance(x, str):
            return str.lower(x.replace(" ", "")) 
        else:
            return ''
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

print(df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

def create_soup(x):
    return ' '.join(x['cast']) + ' ' + ' '.join(x['keywords']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)