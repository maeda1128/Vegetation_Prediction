# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:38:42 2023

@author: maeda_naoya
"""


import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', 50)

list1=[]
list2=[]

def logistics4(b1,b2,b3,b4,b5,b6,b7):
    print(b7+ "_"+b5)
    mod_glm=smf.glm(formula="植生正解 ~ "+b1+" + "+b2+" + "+b3+" ",data=data1,family=sm.families.Binomial()).fit()
    print(mod_glm.summary())
    AIC_A=-2*(mod_glm.llf-(mod_glm.df_model +1))
    print("AIC:",AIC_A.round(3))#AICの導出

    df1=1/(1+(np.exp(-(mod_glm.params["Intercept"]+df[b1]*(mod_glm.params[b1])+df[b2]*(mod_glm.params[b2])+df[b3]*(mod_glm.params[b3])))))
    #print(df1)

    df["予測値"]=df1
    print( "ロジスティック回帰モデル AUC:", roc_auc_score(df["植生正解"],df["予測値"]))
    #df.to_csv("D:/鬼怒川/2017_鬼怒川/予測結果/12_15/"+b4+".csv")
    df['予測判定値'] =np.nan   
    fpr, tpr, thresholds = roc_curve(df["植生正解"],df["予測値"])#真値と予測値でrocカーブを描きそこから求められる値を定義
    #print( "ロジスティック回帰モデル AUC:", roc_auc_score(df["植生正解"],df["予測値"]))
    #print(df)

    list_pred1=df["予測値"].values.tolist()

    df_try=pd.DataFrame(index=[],columns=["car","fp","tp","fn","tn","all","thresholds"])
    list100=[i / 100 for i in range(1, 101, 1)]#0.01~1.00 listにする
    df_try["thresholds"]=pd.Series(list100)#list１００を閾値としてdfにする
          
    def car(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(0)
            else:list_pred.append(1)
        #print(list_pred)
        df["予測判定値"]=pd.DataFrame(list_pred)
        return accuracy＿score(df["植生正解"],df["予測判定値"])
    def tp(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df["予測判定値"]=pd.DataFrame(list_pred)
        df["odd"]=df["予測判定値"]+df["植生正解"]
        #print(df["odd"])
        return sum(df["odd"]==2)
    def fp(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df["予測判定値"]=pd.DataFrame(list_pred)
        df["odd"]=df["予測判定値"]+df["植生正解"]
        #print(df_new["odd"])
        return sum(df["odd"]==1)
    def fn(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df["予測判定値"]=pd.DataFrame(list_pred)
        df["odd"]=df["予測判定値"]+df["植生正解"]
        #print(df_new["odd"])
        return sum(df["odd"]==11)
    def tn(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df["予測判定値"]=pd.DataFrame(list_pred)
        df["odd"]=df["予測判定値"]+df["植生正解"]
        #print(df_new["odd"])
        return sum(df["odd"]==10)

    list_car=[]
    for i in list100:list_car.append(car(i))
    list_tp=[]
    for i in list100:list_tp.append(tp(i))
    list_fp=[]
    for i in list100:list_fp.append(fp(i))       
    list_fn=[]
    for i in list100:list_fn.append(fn(i))    
    list_tn=[]
    for i in list100:list_tn.append(tn(i)) 

    df_try["car"]=pd.Series(list_car)
    df_try["tp"]=pd.Series(list_tp)
    df_try["fp"]=pd.Series(list_fp)
    df_try["fn"]=pd.Series(list_fn)
    df_try["tn"]=pd.Series(list_tn)
    df_try["all"]=df_try["tp"]+df_try["fp"]+df_try["fn"]+df_try["tn"]
    df_try["thresholds"]=pd.Series(list100)
    df_try["tpr"]=df_try["tp"]/(df_try["tp"]+df_try["fn"])
    df_try["fpr"]=df_try["fp"]/(df_try["tn"]+df_try["fp"])
    #print(df_try)

    #rocカーブから(FPR,TPR)=(0,1)に最も近い閾値を求める
    pd.set_option('display.max_rows', 10)
    df_try["dis"]=np.sqrt(((df_try["fpr"])**2)+((1-df_try["tpr"])**2))
    pd.set_option('display.max_columns', 15)
    cutoff_D=df_try.iloc[df_try["dis"].idxmin(),6]

    print("距離が最も近い閾値",cutoff_D)
    print("")

    def heat(a):
        list_pred=[]
        for i in list_pred1:
            if i<=  a:
                list_pred.append(0)
            else:
                list_pred.append(1)
        #print(list_pred)

        df["予測判定値"]=pd.DataFrame(list_pred)
        #print(df_new)
        

        Pred=confusion_matrix(df["植生正解"],df["予測判定値"])
        #print(Pred)

        sns.heatmap(Pred,annot=True, cmap='Blues',fmt='d',vmin=0, vmax=b4)
        plt.title(b5)
        plt.show()
        #print(df)
        print("閾値:",a)
        print("正答率:",accuracy_score(df["植生正解"],df["予測判定値"]))
        print("適合率:",precision_score(df["植生正解"],df["予測判定値"]))
        print("再現率:",recall_score(df["植生正解"],df["予測判定値"]))
        print("F値:",f1_score(df["植生正解"],df["予測判定値"]))
        print("")
        
    heat(cutoff_D)
    
    df3=1/(1+(np.exp(-(mod_glm.params["Intercept"]+df2[b1]*(mod_glm.params[b1])+df2[b2]*(mod_glm.params[b2])+df2[b3]*(mod_glm.params[b3])))))
    #print(df1)

    df2["予測値"]=df3
    print( "ロジスティック回帰モデル AUC:", roc_auc_score(df2["植生正解"],df2["予測値"]))
    #df.to_csv("D:/鬼怒川/2017_鬼怒川/予測結果/12_15/"+b4+".csv")
    df2['予測判定値'] =np.nan   
    fpr, tpr, thresholds = roc_curve(df2["植生正解"],df2["予測値"])#真値と予測値でrocカーブを描きそこから求められる値を定義
    #print( "ロジスティック回帰モデル AUC:", roc_auc_score(df["植生正解"],df["予測値"]))
    #print(df)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title("ROC curve_"+b5)
    plt.grid()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.show()

    list_pred1=df2["予測値"].values.tolist()

    df_try2=pd.DataFrame(index=[],columns=["car","fp","tp","fn","tn","all","thresholds"])
    list100=[i / 100 for i in range(1, 101, 1)]#0.01~1.00 listにする
    df_try2["thresholds"]=pd.Series(list100)#list１００を閾値としてdfにする
          
    def car(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(0)
            else:list_pred.append(1)
        #print(list_pred)
        df2["予測判定値"]=pd.DataFrame(list_pred)
        return accuracy＿score(df2["植生正解"],df2["予測判定値"])
    def tp(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df2["予測判定値"]=pd.DataFrame(list_pred)
        df2["odd"]=df2["予測判定値"]+df2["植生正解"]
        #print(df["odd"])
        return sum(df2["odd"]==2)
    def fp(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df2["予測判定値"]=pd.DataFrame(list_pred)
        df2["odd"]=df2["予測判定値"]+df2["植生正解"]
        #print(df_new["odd"])
        return sum(df2["odd"]==1)
    def fn(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df2["予測判定値"]=pd.DataFrame(list_pred)
        df2["odd"]=df2["予測判定値"]+df2["植生正解"]
        #print(df_new["odd"])
        return sum(df2["odd"]==11)
    def tn(a):
        list_pred=[]
        for i in list_pred1:
            if i<= a:list_pred.append(10)
            else:list_pred.append(1)
        df2["予測判定値"]=pd.DataFrame(list_pred)
        df2["odd"]=df2["予測判定値"]+df2["植生正解"]
        #print(df_new["odd"])
        return sum(df2["odd"]==10)

    list_car=[]
    for i in list100:list_car.append(car(i))
    list_tp=[]
    for i in list100:list_tp.append(tp(i))
    list_fp=[]
    for i in list100:list_fp.append(fp(i))       
    list_fn=[]
    for i in list100:list_fn.append(fn(i))    
    list_tn=[]
    for i in list100:list_tn.append(tn(i)) 

    df_try2["car"]=pd.Series(list_car)
    df_try2["tp"]=pd.Series(list_tp)
    df_try2["fp"]=pd.Series(list_fp)
    df_try2["fn"]=pd.Series(list_fn)
    df_try2["tn"]=pd.Series(list_tn)
    df_try2["all"]=df_try2["tp"]+df_try2["fp"]+df_try2["fn"]+df_try2["tn"]
    df_try2["thresholds"]=pd.Series(list100)
    df_try2["tpr"]=df_try2["tp"]/(df_try2["tp"]+df_try2["fn"])
    df_try2["fpr"]=df_try2["fp"]/(df_try2["tn"]+df_try2["fp"])
    #print(df_try)

    def heat(a):
        list_pred=[]
        for i in list_pred1:
            if i<=  a:
                list_pred.append(0)
            else:
                list_pred.append(1)
        #print(list_pred)

        df2["予測判定値"]=pd.DataFrame(list_pred)
        #print(df_new)
        

        Pred=confusion_matrix(df2["植生正解"],df2["予測判定値"])
        #print(Pred)

        sns.heatmap(Pred,annot=True, cmap='Blues',fmt='d',vmin=0, vmax=b6)
        plt.title(b5)
        plt.show()
        #print(df)
        print("閾値:",a)
        print("正答率:",accuracy_score(df2["植生正解"],df2["予測判定値"]))
        print("適合率:",precision_score(df2["植生正解"],df2["予測判定値"]))
        print("再現率:",recall_score(df2["植生正解"],df2["予測判定値"]))
        print("F値:",f1_score(df2["植生正解"],df2["予測判定値"]))
        print("")
        
    heat(cutoff_D)
    list2.append([b7,b5,"AIC:",AIC_A.round(3),"閾値",cutoff_D,"テストデータ","AUC", roc_auc_score(df["植生正解"],df["予測値"]),"正答率:",accuracy_score(df["植生正解"],df["予測判定値"]),"適合率",precision_score(df["植生正解"],df["予測判定値"]),"再現率",recall_score(df["植生正解"],df["予測判定値"]),"F値",f1_score(df["植生正解"],df["予測判定値"]),"テストデータ","AUC", roc_auc_score(df2["植生正解"],df2["予測値"]),"正答率:",accuracy_score(df2["植生正解"],df2["予測判定値"]),"適合率",precision_score(df2["植生正解"],df2["予測判定値"]),"再現率",recall_score(df2["植生正解"],df2["予測判定値"]),"F値",f1_score(df2["植生正解"],df2["予測判定値"])])
    list1.append([b5,"切片",mod_glm.params["Intercept"],b1,mod_glm.params[b1],b2,mod_glm.params[b2],b3,mod_glm.params[b3]])

#F_seg1
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg1_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg1_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg1_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg1_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg1_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg1")



#F_seg2_1
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_1_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_1_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_1_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_1_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_1_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_1")



#F_seg2_2
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_2_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_2_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_2_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_2_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_2_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_2_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_2_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_2_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/F_seg2_2_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg2_2_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"F_seg2_2")

#N_seg1
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg1_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg1_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/F_seg1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg1_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg1_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg1_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg1")



#N_seg2_1
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_1_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_1_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_1_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_1_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_1")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_1_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_1")



#N_seg2_2
data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_2_1.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_2_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_2_2.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_2_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_2_3.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_2_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_2_4.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_2_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_2")

data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/train/std/N_seg2_2_5.csv", encoding="shift-jis")
df2=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/test/std/N_seg2_2_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist',"R_height","植生",110000,"modelC",47000,"N_seg2_2")

listp=[]
def logisticsp(b1,b2,b3,b4):
    print(b4)
    data1=pd.read_csv("D:/鬼怒川/データセット/データセットy/k_split/all/std/"+b4+".csv", encoding="shift-jis")
    data1 = data1.dropna(how='all').dropna(how='all', axis=1)
    mod_glm=smf.glm(formula="植生正解 ~ "+b1+" + "+b2+" + "+b3+"",data=data1,family=sm.families.Binomial()).fit()
    print(mod_glm.summary())
    listp.append([b4,"切片",mod_glm.params["Intercept"],b1,mod_glm.params[b1],b2,mod_glm.params[b2],b3,mod_glm.params[b3]])

logisticsp('Hubdist',"R_height","植生","F_seg1")
logisticsp('Hubdist',"R_height","植生","F_seg2_1")
logisticsp('Hubdist',"R_height","植生","F_seg2_2")
logisticsp('Hubdist',"R_height","植生","N_seg1")
logisticsp('Hubdist',"R_height","植生","N_seg2_1")
logisticsp('Hubdist',"R_height","植生","N_seg2_2")

df_p=pd.DataFrame(list1)
df_F=pd.DataFrame(list2)
df_p=pd.DataFrame(listp)
#df_p=df_p.T
#df_F=df_F.T
df_p.to_csv("D:/鬼怒川/データセット/データセットy/k_split/result/c_para_all.csv",encoding="shift-jis",index=False,header=False)
df_F.to_csv("D:/鬼怒川/データセット/データセットy/k_split/result/c_f1_all_c.csv",encoding="shift-jis",index=False,header=False)
df_p.to_csv("D:/鬼怒川/データセット/データセットy/k_split/result/c_para_model.csv",encoding="shift-jis",index=False,header=False)