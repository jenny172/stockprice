# -*- coding: utf-8 -*-
#pip install yfinance
import yfinance as yf
stock = ("2323", "2374", "2409", "3059", "3149", "3481", "3504", "4934", "6164", "6176")
for i in stock :
    df = yf.download(i + ".TW", "2014-01-01", "2023-12-31")
    df.to_csv("yahoo_" + i + "TW.csv")

