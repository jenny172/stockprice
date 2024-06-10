# -*- coding: utf-8 -*-
"""
Created on Wed Jun 5 17:32:06 2024

@author: jenny, jacky, carol
"""

import os
import pandas as pd
import jieba
print(jieba.__version__)
os.chdir("D:\金融大數據\期末報告")
YahooNews=pd.read_csv('航運報導.csv')

###############################
###Part I: Identify the Noise###
###############################
import re

RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

def impurity(text, min_len=10):
    """returns the share of suspicious characters in a text"""
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

YahooNews['Impurity']=YahooNews['Context'].apply(impurity, min_len=10)
YahooNews.columns
YahooNews[['Context', 'Impurity']].sort_values(by='Impurity', ascending=False).head(6)

#####################################################
###Part II: Removing Noise with Regular Expressions###
#####################################################

#remark: html.unescape()，converts HTML-safe sequences (&lt;, &gt;, etc.) into their corresponding characters (<, >, etc.)
import html
p = '&lt;abc&gt;' #&lt; and &gt; are special symbols in html
#not showing in text example
txt= html.unescape(p)
print (txt)

import html

#定義清理(clean)函數
def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text) #in this example, this part does nothing
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', ' ', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    #清除逗號和-符號
    text = re.sub(r'[,-]', ' ', text)
    
    return text.strip()

#利用clean函數處理文本
YahooNews['Clean_text'] = YahooNews['Context'].apply(clean)
#加入不純度(Impurity)變數
YahooNews['Impurity']   = YahooNews['Clean_text'].apply(impurity, min_len=20)
#顯示清理後的前六個文本之不純度
YahooNews[['Clean_text', 'Impurity']].sort_values(by='Impurity', ascending=False).head(6)     


####################################################
###Part III: Character Normalization with textacy###
####################################################  
#No need for Chinese


#############################################
###Part IV: Character Masking with textacy###
#############################################
from textacy.preprocessing import replace

#移除url
YahooNews['Clean_text']=YahooNews['Clean_text'].apply(replace.urls)
#重新命名變數
YahooNews.rename(columns={'Context': 'Raw_text', 'Clean_text': 'Context'}, inplace=True)
YahooNews.drop(columns=['Impurity'], inplace=True)




##########################
###Liguistic Processing###
##########################
#1加入繁體詞典
import jieba

jieba.set_dictionary('dict.txt.big.txt')
stopwords1 = [line.strip() for line in open('stopWords.txt', 'r', encoding='utf-8').readlines()]

def remove_stop(text):
    c1=[]
    for w in text:
        if w not in stopwords1:
            c1.append(w)
    c2=[i for i in c1 if i.strip() != '']
    return c2

#def remove_stop(text):
    #words = jieba.cut(text)
    #filtered_words = [w for w in words if w not in stopwords1 and w.strip() != '']
    #return list(filtered_words)

#斷詞並移除停用詞
YahooNews['tokens']=YahooNews['Context'].apply(jieba.cut)
YahooNews['tokens_new']=YahooNews['tokens'].apply(remove_stop)
YahooNews.iloc[0,:]


#Freq charts
from collections import Counter
counter = Counter()#use a empty string first
YahooNews['tokens_new'].apply(counter.update)
print(counter.most_common(15))

import seaborn as sns
sns.set(font="SimSun")
min_freq=2
#transform dict into dataframe
freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
freq_df = freq_df.query('freq >= @min_freq')
freq_df.index.name = 'token'
freq_df = freq_df.sort_values('freq', ascending=False)
freq_df.head(15)

ax = freq_df.head(15).plot(kind='barh', width=0.95, figsize=(8,3))
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Top Words')

###Creating Word Clouds
from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(freq_df['freq'])
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)



#將斷詞結果轉為字串
def list_to_string(org_list, seperator=' '):
    return seperator.join(org_list)

YahooNews['News_seg']=YahooNews['tokens_new'].apply(list_to_string)
YahooNews['News_seg'][1]

#將YahooNews['News_seg']轉為文章-字詞矩陣
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(decode_error='ignore', min_df=2) 

dt01 = cv.fit_transform(YahooNews['News_seg'])
print(cv.get_feature_names_out())
fn=cv.get_feature_names_out()

# Convert the sparse matrix to a dense array and create a DataFrame
dtmatrix=pd.DataFrame(dt01.toarray(), columns=cv.get_feature_names_out())
print(dtmatrix)


#cosine_similarity是計算各向量間每行的相似度
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(dt01[20], dt01[24])

#計算文章間的相似性
sm = pd.DataFrame(cosine_similarity(dt01, dt01))


YahooNews['Context'][20]
YahooNews['Context'][24]



from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

tfidf_dt = tfidf.fit_transform(dt01)
tfidfmatrix = pd.DataFrame(tfidf_dt.toarray(), columns=cv.get_feature_names_out())

#cosine_similarity是計算各向量間每行的相似度
cosine_similarity(tfidf_dt[20], tfidf_dt[22])

#計算各字詞間的相似性
sm1 =pd.DataFrame(cosine_similarity(tfidf_dt, tfidf_dt))

#計算經過TF-IDF轉換後文章間的相似性
sm2 = pd.DataFrame(cosine_similarity(tfidf_dt.transpose(), tfidf_dt.transpose()))


YahooNews['Context'][20]
YahooNews['Context'][22]


from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

tfidfsum=tfidfmatrix.T.sum(axis=1)

wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white")
wordcloud.generate_from_frequencies(tfidfsum)
#plt.figure(figsize=(20,10)) 
plt.imshow(wordcloud)


#集群分析
from sklearn.cluster import KMeans

from sklearn import preprocessing 
distortions = []
for i in range(1, 31):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(preprocessing.normalize(tfidf_dt))
    distortions.append(km.inertia_)

# plot
from matplotlib import pyplot as plt
plt.plot(range(1, 31), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#分成五群
km = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(preprocessing.normalize(tfidf_dt))


# 將聚集類別標籤加入到DataFrame中
YahooNews['Cluster'] = y_km



###Creating Word Clouds
from matplotlib import pyplot as plt
from wordcloud import WordCloud ###
from collections import Counter ###

# 繪製各分群之文字雲
for cluster in range(5):
    cluster_data = YahooNews[YahooNews['Cluster'] == cluster]
    text = ' '.join(cluster_data['News_seg'])
    wordcloud = WordCloud(font_path="SimHei.ttf", background_color="white").generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {cluster}')
    plt.show()


#g0 = YahooNews['Context'][y_km==0]
#g0.head()
#g1 = YahooNews['Context'][y_km==1]
#g1.head()
#g2 = YahooNews['Context'][y_km==2]
#g2.head()
#g3 = YahooNews['Context'][y_km==3]
#g3.head()
#g4 = YahooNews['Context'][y_km==4]
#g4.head()




YahooNews['length']= YahooNews['Context'].str.len()

YahooNews.info()
YNews = YahooNews.copy()  

YNews.dtypes
YNews['DateTime'] = YNews['Date'].astype(str) + " " + YNews['Time']
type(YNews['DateTime'])
YNews['DateTime'].dtypes
YNews['DateTime'] = pd.to_datetime(YNews['DateTime'], format='%Y/%m/%d %H:%M:%S')
YNews['DateTime'][1]-YNews['DateTime'][0]

#轉成日期格式
YNews['DateOnly'] = YNews['DateTime'].dt.date
YNews['week_of_year'] = YNews['DateTime'].dt.isocalendar().week
YNews['day_of_week'] = YNews['DateTime'].dt.dayofweek
YNews['day'] = YNews['DateTime'].dt.day
YNews['month'] = YNews['DateTime'].dt.month
YNews['year'] = YNews['DateTime'].dt.year

YNews.describe()
#agg可計算新聞之長度，plot可將其畫成圖形
YNews.groupby('DateOnly').agg({'length':'mean'}).plot(rot=45)
#size可計算新聞之篇幅數量
YNews.groupby('day_of_week').size().plot(rot=45)
#畫出長條圖
YNews.groupby('day_of_week').size().plot.bar()


#計算文章來自各網站的次數
YahooNews['From'].value_counts()

#尋找各文章和標題包含關鍵字的次數
#YahooNews[YahooNews['Context'].str.contains('貨櫃')].From.count()   #計算總次數
YahooNews[YahooNews['Context'].str.contains('貨櫃')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('貨櫃')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('運價')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('運價')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('TW')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('TW')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('紅海')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('紅海')].From.value_counts()

YahooNews[YahooNews['Context'].str.contains('航線')].From.value_counts()
YahooNews[YahooNews['Title'].str.contains('航線')].From.value_counts()

#YahooNews 的敘述性統計
YahooNews.describe()
YahooNews['length'].var()
YahooNews['length'].plot(kind='hist').mean()

#畫各網站的平均文章長度的箱型圖
YahooNews['length'].plot(kind='box')
YNews =YahooNews[YahooNews['From'].isin(['經濟日報','yahoo新聞','中時新聞網','鉅亨網'])]
YNews.boxplot(column='length',by='From', vert=False)



