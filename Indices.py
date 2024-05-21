# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

list1 = ['2323', '2374', '2409', '3059', '3149', '3481', '3504', '4934', '6164', '6176']
list2 = ['流動比率', '速動比率', '利息保障倍數', '應收帳款週轉率', '平均收現日數', \
         '存貨週轉率','平均銷貨日數']
list3 = ['總資產週轉率', '毛利率', '營業利益率', '稅後純利率', '資產報酬率', \
         '權益報酬率','營業收入年增率']
list4 = ['營業毛利年增率','營業利益年增率','稅後純益年增率','每股盈餘年增率', \
         '營業現金對流動負債比','營業現金對負債比','營業現金對稅後純益比']

url1 = "https://raw.githubusercontent.com/"

for i in list1 :
    url2 = url1 + "jenny172/stockprice/main/yahoo_" + i + "TW.csv"
    df = pd.read_csv(url2, index_col=0)
    #df.isnull().sum()
    #df.fillna(df='ffill')
    df.to_csv("yahoo_" + i +"TW.csv")
    
for j in list2 :
    url3 = url1 + "jenny172/stockprice/main/" + j + ".csv"
    df = pd.read_csv(url3, index_col=0)
    df.to_csv(j + ".csv")
    
for k in list3 :
    url4 = url1 + "carolkao2258/trade/main/" + k + ".csv"
    df = pd.read_csv(url4, index_col=0)
    df.to_csv(k + ".csv")
    
for m in list4 : 
    url5 = url1 + "jacky2223/hihi/main/" + m + ".csv"
    df = pd.read_csv(url5, index_col=0)
    df.to_csv(m + ".csv")
    
    


    