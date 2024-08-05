# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:11:15 2023

@author: maeda_naoya
"""

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 15)

list1=[]
list2=[]
summary_list=[]

def logistics(use_columns,model_name,Segment,Situation):
    print(model_name)
    # Fitting the logistic regression model
    mod_glm=smf.glm(formula = "answer ~ " + " + ".join(use_columns),data=df,family=sm.families.Binomial()).fit()
    # Getting summary table as DataFrame
    summary_df = pd.DataFrame(mod_glm.summary().tables[1].data, columns=mod_glm.summary().tables[1].data[0])
    summary_df = summary_df.drop(0)
    summary_list.append(summary_df)
    print(mod_glm.summary())
    
    AIC_A=-2*(mod_glm.llf-(mod_glm.df_model +1))
    print("AIC:",AIC_A.round(3))#AIC
    #train data predict
    df1=mod_glm.predict(df)
    df["Pred"]=df1
    print( "Logistic Regression Model AUC:", roc_auc_score(df["answer"],df["Pred"]))
    df['JPred'] =np.nan   
    fpr, tpr, thresholds = roc_curve(df["answer"],df["Pred"])

    # Setting new thresholds
    thresholds_new = np.arange(0, 1.01, 0.01)
    new_fpr = []
    new_tpr = []
    
    # Calculating new FPR and TPR for new thresholds
    for threshold in thresholds_new:
        y_pred = [1 if prob >= threshold else 0 for prob in df["Pred"]]
        tn, fp, fn, tp = confusion_matrix(df["answer"], y_pred).ravel()
        new_fpr.append(fp / (fp + tn))
        new_tpr.append(tp / (tp + fn))

    # Creating DataFrame for new FPR and TPR
    roc_df = pd.DataFrame({'Threshold': thresholds_new, 'FPR': new_fpr, 'TPR': new_tpr})

    # ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    #rocカーブから(FPR,TPR)=(0,1)に最も近い閾値を求める
    roc_df["dis"]=np.sqrt(((roc_df["FPR"])**2)+((1-roc_df["TPR"])**2))

    cutoff_D = roc_df.loc[roc_df["dis"].idxmin(), "Threshold"]
    print("")
    print("train")
    print("Cutoff:",cutoff_D)
    
    # Defining function to plot heat map
    def heat(a,b):
        for i in b.index:
            b.loc[i, "JPred"] = 0 if b.loc[i, "Pred"] <= a else 1
            
        Pred = confusion_matrix(b["answer"], b["JPred"])
        sns.heatmap(Pred, annot=True, cmap='Blues', fmt='d', vmin=0)
        plt.show()
        print("F-measure", f1_score(b["answer"], b["JPred"]))
        print("")
    
    # Plotting heat map for train data
    heat(cutoff_D,df)
    
    print("test")
    #test data predict
    df3=mod_glm.predict(df2)
    df2["Pred"]=df3
    print( "Logistic Regression Model AUC:", roc_auc_score(df2["answer"],df2["Pred"]))
    df2['JPred'] =np.nan   
    fpr, tpr, thresholds = roc_curve(df2["answer"],df2["Pred"])
    
    # ROC curve plot
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title("ROC curve_"+model_name)
    plt.grid()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.show()
    
    # Plotting heat map for test data
    heat(cutoff_D,df2)

#k_split1
data1=pd.read_csv("./5_split/train/F_seg2_1_1.csv", encoding="shift-jis")
data2=pd.read_csv("./5_split/test/F_seg2_1_1.csv", encoding="shift-jis")

print("Nomber of Training data:",len(data1))
print("Training data rate:",len(data1)/(len(data1)+len(data2)))
print("Nomber of Test data:",len(data2))
print("Test data rate:",len(data2)/(len(data1)+len(data2)))
data1 = data1.dropna(how='all').dropna(how='all', axis=1)
df=data1
df2=data2
logistics(['DisWater',"RHeight"],"modelA","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP"],"modelB","Segment2_1","Flood")
logistics(['DisWater',"RHeight","Vegetation"],"modelC","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP","Vegetation"],"modelD","Segment2_1","Flood")

#k_split2
data1=pd.read_csv("./5_split/train/F_seg2_1_2.csv", encoding="shift-jis")
data2=pd.read_csv("./5_split/test/F_seg2_1_2.csv", encoding="shift-jis")

print("Nomber of Training data:",len(data1))
print("Training data rate:",len(data1)/(len(data1)+len(data2)))
print("Nomber of Test data:",len(data2))
print("Test data rate:",len(data2)/(len(data1)+len(data2)))
data1 = data1.dropna(how='all').dropna(how='all', axis=1)
df=data1
df2=data2
logistics(['DisWater',"RHeight"],"modelA","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP"],"modelB","Segment2_1","Flood")
logistics(['DisWater',"RHeight","Vegetation"],"modelC","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP","Vegetation"],"modelD","Segment2_1","Flood")

#k_split3
data1=pd.read_csv("./5_split/train/F_seg2_1_3.csv", encoding="shift-jis")
data2=pd.read_csv("./5_split/test/F_seg2_1_3.csv", encoding="shift-jis")

print("Nomber of Training data:",len(data1))
print("Training data rate:",len(data1)/(len(data1)+len(data2)))
print("Nomber of Test data:",len(data2))
print("Test data rate:",len(data2)/(len(data1)+len(data2)))
data1 = data1.dropna(how='all').dropna(how='all', axis=1)
df=data1
df2=data2
logistics(['DisWater',"RHeight"],"modelA","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP"],"modelB","Segment2_1","Flood")
logistics(['DisWater',"RHeight","Vegetation"],"modelC","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP","Vegetation"],"modelD","Segment2_1","Flood")

#k_split4
data1=pd.read_csv("./5_split/train/F_seg2_1_4.csv", encoding="shift-jis")
data2=pd.read_csv("./5_split/test/F_seg2_1_4.csv", encoding="shift-jis")

print("Nomber of Training data:",len(data1))
print("Training data rate:",len(data1)/(len(data1)+len(data2)))
print("Nomber of Test data:",len(data2))
print("Test data rate:",len(data2)/(len(data1)+len(data2)))
data1 = data1.dropna(how='all').dropna(how='all', axis=1)
df=data1
df2=data2
logistics(['DisWater',"RHeight"],"modelA","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP"],"modelB","Segment2_1","Flood")
logistics(['DisWater',"RHeight","Vegetation"],"modelC","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP","Vegetation"],"modelD","Segment2_1","Flood")

#k_split5
data1=pd.read_csv("./5_split/train/F_seg2_1_5.csv", encoding="shift-jis")
data2=pd.read_csv("./5_split/test/F_seg2_1_5.csv", encoding="shift-jis")

print("Nomber of Training data:",len(data1))
print("Training data rate:",len(data1)/(len(data1)+len(data2)))
print("Nomber of Test data:",len(data2))
print("Test data rate:",len(data2)/(len(data1)+len(data2)))
data1 = data1.dropna(how='all').dropna(how='all', axis=1)
df=data1
df2=data2
logistics(['DisWater',"RHeight"],"modelA","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP"],"modelB","Segment2_1","Flood")
logistics(['DisWater',"RHeight","Vegetation"],"modelC","Segment2_1","Flood")
logistics(['DisWater',"ShieldsP","Vegetation"],"modelD","Segment2_1","Flood")

df_p=pd.DataFrame(list1)
df_F=pd.DataFrame(list2)
#df_p=df_p.T
#df_F=df_F.T
df_p.to_csv("./5_split/result/F_para_seg2_1.csv",encoding="shift-jis",index=False,header=False)
df_F.to_csv("./5_split/result/F_f1_seg2_1.csv",encoding="shift-jis",index=False,header=False)
