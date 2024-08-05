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

list1 = []
list2 = []

def logistics4(b1, b2, b3, b4, b5, b6, b7):
    print(b7 + "_" + b5)
    mod_glm = smf.glm(formula="answer ~ " + b1 + " + " + b2 + " + " + b3, data=data1, family=sm.families.Binomial()).fit()
    print(mod_glm.summary())
    AIC_A = -2 * (mod_glm.llf - (mod_glm.df_model + 1))
    print("AIC:", AIC_A.round(3))

    df1 = 1 / (1 + (np.exp(-(mod_glm.params["Intercept"] + df[b1] * (mod_glm.params[b1]) + df[b2] * (mod_glm.params[b2]) + df[b3] * (mod_glm.params[b3])))))
    
    df["P"] = df1
    print("Logistic Regression Model AUC:", roc_auc_score(df["answer"], df["P"]))
    
    df['binary_predicted'] = np.nan
    fpr, tpr, thresholds = roc_curve(df["answer"], df["P"])
    
    list_pred1 = df["P"].values.tolist()

    df_try = pd.DataFrame(index=[], columns=["accuracy", "fp", "tp", "fn", "tn", "all", "thresholds"])
    list100 = [i / 100 for i in range(1, 101, 1)]
    df_try["thresholds"] = pd.Series(list100)

    def accuracy(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(0)
            else: list_pred.append(1)
        df["binary_predicted"] = pd.DataFrame(list_pred)
        return accuracy_score(df["answer"], df["binary_predicted"])

    def tp(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df["binary_predicted"] = pd.DataFrame(list_pred)
        df["odd"] = df["binary_predicted"] + df["answer"]
        return sum(df["odd"] == 2)

    def fp(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df["binary_predicted"] = pd.DataFrame(list_pred)
        df["odd"] = df["binary_predicted"] + df["answer"]
        return sum(df["odd"] == 1)

    def fn(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df["binary_predicted"] = pd.DataFrame(list_pred)
        df["odd"] = df["binary_predicted"] + df["answer"]
        return sum(df["odd"] == 11)

    def tn(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df["binary_predicted"] = pd.DataFrame(list_pred)
        df["odd"] = df["binary_predicted"] + df["answer"]
        return sum(df["odd"] == 10)

    list_accuracy = []
    for i in list100: list_accuracy.append(accuracy(i))
    list_tp = []
    for i in list100: list_tp.append(tp(i))
    list_fp = []
    for i in list100: list_fp.append(fp(i))
    list_fn = []
    for i in list100: list_fn.append(fn(i))
    list_tn = []
    for i in list100: list_tn.append(tn(i))

    df_try["accuracy"] = pd.Series(list_accuracy)
    df_try["tp"] = pd.Series(list_tp)
    df_try["fp"] = pd.Series(list_fp)
    df_try["fn"] = pd.Series(list_fn)
    df_try["tn"] = pd.Series(list_tn)
    df_try["all"] = df_try["tp"] + df_try["fp"] + df_try["fn"] + df_try["tn"]
    df_try["thresholds"] = pd.Series(list100)
    df_try["tpr"] = df_try["tp"] / (df_try["tp"] + df_try["fn"])
    df_try["fpr"] = df_try["fp"] / (df_try["tn"] + df_try["fp"])

    pd.set_option('display.max_rows', 10)
    df_try["distance"] = np.sqrt(((df_try["fpr"]) ** 2) + ((1 - df_try["tpr"]) ** 2))
    pd.set_option('display.max_columns', 15)
    cutoff_D = df_try.iloc[df_try["distance"].idxmin(), 6]

    print("Closest Threshold", cutoff_D)
    print("")

    def heat(a):
        list_pred = []
        for i in list_pred1:
            if i <= a:
                list_pred.append(0)
            else:
                list_pred.append(1)
        
        df["binary_predicted"] = pd.DataFrame(list_pred)
        Pred = confusion_matrix(df["answer"], df["binary_predicted"])

        sns.heatmap(Pred, annot=True, cmap='Blues', fmt='d', vmin=0, vmax=b4)
        plt.title(b5)
        plt.show()
        
        print("Threshold:", a)
        print("Accuracy:", accuracy_score(df["answer"], df["binary_predicted"]))
        print("Precision:", precision_score(df["answer"], df["binary_predicted"]))
        print("Recall:", recall_score(df["answer"], df["binary_predicted"]))
        print("F1 Score:", f1_score(df["answer"], df["binary_predicted"]))
        print("")
        
    heat(cutoff_D)
    
    df3 = 1 / (1 + (np.exp(-(mod_glm.params["Intercept"] + df2[b1] * (mod_glm.params[b1]) + df2[b2] * (mod_glm.params[b2]) + df2[b3] * (mod_glm.params[b3])))))
    df2["P"] = df3
    print("Logistic Regression Model AUC:", roc_auc_score(df2["answer"], df2["P"]))
    
    df2['binary_predicted'] = np.nan
    fpr, tpr, thresholds = roc_curve(df2["answer"], df2["P"])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title("ROC curve_" + b5)
    plt.grid()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.show()

    list_pred1 = df2["P"].values.tolist()

    df_try2 = pd.DataFrame(index=[], columns=["accuracy", "fp", "tp", "fn", "tn", "all", "thresholds"])
    list100 = [i / 100 for i in range(1, 101, 1)]
    df_try2["thresholds"] = pd.Series(list100)

    def accuracy(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(0)
            else: list_pred.append(1)
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        return accuracy_score(df2["answer"], df2["binary_predicted"])

    def tp(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        df2["odd"] = df2["binary_predicted"] + df2["answer"]
        return sum(df2["odd"] == 2)

    def fp(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        df2["odd"] = df2["binary_predicted"] + df2["answer"]
        return sum(df2["odd"] == 1)

    def fn(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        df2["odd"] = df2["binary_predicted"] + df2["answer"]
        return sum(df2["odd"] == 11)

    def tn(a):
        list_pred = []
        for i in list_pred1:
            if i <= a: list_pred.append(10)
            else: list_pred.append(1)
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        df2["odd"] = df2["binary_predicted"] + df2["answer"]
        return sum(df2["odd"] == 10)

    list_accuracy = []
    for i in list100: list_accuracy.append(accuracy(i))
    list_tp = []
    for i in list100: list_tp.append(tp(i))
    list_fp = []
    for i in list100: list_fp.append(fp(i))
    list_fn = []
    for i in list100: list_fn.append(fn(i))
    list_tn = []
    for i in list100: list_tn.append(tn(i))

    df_try2["accuracy"] = pd.Series(list_accuracy)
    df_try2["tp"] = pd.Series(list_tp)
    df_try2["fp"] = pd.Series(list_fp)
    df_try2["fn"] = pd.Series(list_fn)
    df_try2["tn"] = pd.Series(list_tn)
    df_try2["all"] = df_try2["tp"] + df_try2["fp"] + df_try2["fn"] + df_try2["tn"]
    df_try2["thresholds"] = pd.Series(list100)
    df_try2["tpr"] = df_try2["tp"] / (df_try2["tp"] + df_try2["fn"])
    df_try2["fpr"] = df_try2["fp"] / (df_try2["tn"] + df_try2["fp"])

    pd.set_option('display.max_rows', 10)
    df_try2["distance"] = np.sqrt(((df_try2["fpr"]) ** 2) + ((1 - df_try2["tpr"]) ** 2))
    pd.set_option('display.max_columns', 15)
    cutoff_D = df_try2.iloc[df_try2["distance"].idxmin(), 6]

    print("Closest Threshold", cutoff_D)
    print("")

    def heat(a):
        list_pred = []
        for i in list_pred1:
            if i <= a:
                list_pred.append(0)
            else:
                list_pred.append(1)
        
        df2["binary_predicted"] = pd.DataFrame(list_pred)
        Pred = confusion_matrix(df2["answer"], df2["binary_predicted"])

        sns.heatmap(Pred, annot=True, cmap='Blues', fmt='d', vmin=0, vmax=b4)
        plt.title(b6)
        plt.show()
        
        print("Threshold:", a)
        print("Accuracy:", accuracy_score(df2["answer"], df2["binary_predicted"]))
        print("Precision:", precision_score(df2["answer"], df2["binary_predicted"]))
        print("Recall:", recall_score(df2["answer"], df2["binary_predicted"]))
        print("F1 Score:", f1_score(df2["answer"], df2["binary_predicted"]))
        print("")
        
    heat(cutoff_D)
    list2.append([
    b7, b5, "AIC:", AIC_A.round(3), "threshold", cutoff_D, "Test Data", "AUC", 
    roc_auc_score(df["answer"], df["P"]), "Accuracy:", accuracy_score(df["answer"], df["binary_predicted"]), 
    "Precision", precision_score(df["answer"], df["binary_predicted"]), "Recall", recall_score(df["answer"], df["binary_predicted"]), 
    "F1 Score", f1_score(df["answer"], df["binary_predicted"]), "Test Data", "AUC", 
    roc_auc_score(df2["answer"], df2["P"]), "Accuracy:", accuracy_score(df2["answer"], df2["binary_predicted"]), 
    "Precision", precision_score(df2["answer"], df2["binary_predicted"]), "Recall", recall_score(df2["answer"], df2["binary_predicted"]), 
    "F1 Score", f1_score(df2["answer"], df2["binary_predicted"])
    ])

    list1.append([b5, "Intercept", mod_glm.params["Intercept"], b1, mod_glm.params[b1], b2, mod_glm.params[b2], b3, mod_glm.params[b3]])

#data1 is training data .csv
#df2 is test data .csv
#logistics4(variable, variable, variable, maximum heatmap (training), model name, maximum heatmap (test), which segment is F during flood or N during normal)



# F_seg1
data1 = pd.read_csv("./5_split/train/std/F_seg1_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg1")

data1 = pd.read_csv("./5_split/train/std/F_seg1_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg1")

data1 = pd.read_csv("./5_split/train/std/F_seg1_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg1")

data1 = pd.read_csv("./5_split/train/std/F_seg1_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg1")

data1 = pd.read_csv("./5_split/train/std/F_seg1_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg1")

# F_seg2_1
data1 = pd.read_csv("./5_split/train/std/F_seg2_1_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_1")

data1 = pd.read_csv("./5_split/train/std/F_seg2_1_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_1")

data1 = pd.read_csv("./5_split/train/std/F_seg2_1_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_1")

data1 = pd.read_csv("./5_split/train/std/F_seg2_1_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_1")

data1 = pd.read_csv("./5_split/train/std/F_seg2_1_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_1")

# F_seg2_2
data1 = pd.read_csv("./5_split/train/std/F_seg2_2_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_2_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_2")

data1 = pd.read_csv("./5_split/train/std/F_seg2_2_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_2_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_2")

data1 = pd.read_csv("./5_split/train/std/F_seg2_2_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_2_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_2")

data1 = pd.read_csv("./5_split/train/std/F_seg2_2_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_2_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_2")

data1 = pd.read_csv("./5_split/train/std/F_seg2_2_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/F_seg2_2_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('DisWater', "RHeight", "Vegetation", 110000, "modelC", 47000, "F_seg2_2")

# N_seg1
data1 = pd.read_csv("./5_split/train/std/N_seg1_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg1")

data1 = pd.read_csv("./5_split/train/std/N_seg1_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg1")

data1 = pd.read_csv("./5_split/train/std/N_seg1_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg1")

data1 = pd.read_csv("./5_split/train/std/N_seg1_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg1")

data1 = pd.read_csv("./5_split/train/std/N_seg1_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg1")

# N_seg2_1
data1 = pd.read_csv("./5_split/train/std/N_seg2_1_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_1_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_1")

data1 = pd.read_csv("./5_split/train/std/N_seg2_1_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_1_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_1")

data1 = pd.read_csv("./5_split/train/std/N_seg2_1_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_1_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_1")

data1 = pd.read_csv("./5_split/train/std/N_seg2_1_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_1_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_1")

data1 = pd.read_csv("./5_split/train/std/N_seg2_1_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_1_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_1")

# N_seg2_2
data1 = pd.read_csv("./5_split/train/std/N_seg2_2_1.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_2_1.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_2")

data1 = pd.read_csv("./5_split/train/std/N_seg2_2_2.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_2_2.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_2")

data1 = pd.read_csv("./5_split/train/std/N_seg2_2_3.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_2_3.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_2")

data1 = pd.read_csv("./5_split/train/std/N_seg2_2_4.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_2_4.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_2")

data1 = pd.read_csv("./5_split/train/std/N_seg2_2_5.csv", encoding="shift-jis")
df2 = pd.read_csv("./5_split/test/std/N_seg2_2_5.csv", encoding="shift-jis")
df = data1.dropna(how='all').dropna(how='all', axis=1)
logistics4('Hubdist', "R_height", "Vegetation", 110000, "modelC", 47000, "N_seg2_2")


listp=[]
def logisticsp(b1,b2,b3,b4):
    print(b4)
    data1=pd.read_csv("./5_split/all/std/"+b4+".csv", encoding="shift-jis")
    data1 = data1.dropna(how='all').dropna(how='all', axis=1)
    mod_glm=smf.glm(formula="answer ~ "+b1+" + "+b2+" + "+b3+"",data=data1,family=sm.families.Binomial()).fit()
    print(mod_glm.summary())
    listp.append([b4,"Intercept",mod_glm.params["Intercept"],b1,mod_glm.params[b1],b2,mod_glm.params[b2],b3,mod_glm.params[b3]])

logisticsp('Hubdist', "R_height", "Vegetation", "F_seg1")
logisticsp('Hubdist', "R_height", "Vegetation", "F_seg2_1")
logisticsp('Hubdist', "R_height", "Vegetation", "F_seg2_2")
logisticsp('Hubdist', "R_height", "Vegetation", "N_seg1")
logisticsp('Hubdist', "R_height", "Vegetation", "N_seg2_1")
logisticsp('Hubdist', "R_height", "Vegetation", "N_seg2_2")

df_p = pd.DataFrame(list1)
df_F = pd.DataFrame(list2)
df_p = pd.DataFrame(listp)

# Save the dataframes to CSV files
df_p.to_csv("./5_split/result/c_para_all.csv", encoding="shift-jis", index=False, header=False)
df_F.to_csv("./5_split/result/c_f1_all_c.csv", encoding="shift-jis", index=False, header=False)
df_p.to_csv("./5_split/result/c_para_model.csv", encoding="shift-jis", index=False, header=False)
