import pandas as pd 
import numpy as np
import random as rd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier ### import for neural network
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier  ### import for random forest 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier ### import for GBDT
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier ### import for XGBDT
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error ### measure regressor
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc ### measure classification
from sklearn.metrics import accuracy_score

def read_data(t):
    abalone = pd.read_csv('data/abalone.data' , sep=',' , header = None)
    abalone = abalone.values
    ### combine female and male feature as adult
    abalone[:,0] = np.where(abalone[:,0]=='I',0,1)
    ### clean 0 data in this dataset
    for i in range(1,abalone.shape[1]):
        abalone = np.delete(abalone, abalone[:,i] == 0,axis = 0)
    ### 0 for regression and other for classifier 
    if t == 0: 
        abalone = abalone.astype("float")
        return(abalone)
    ### treat the output part by label code which means 0-7 years as 1, 8-10 years as 2, and so on
    else:
        for j in range(abalone.shape[0]):
            mid = abalone[j,-1]
            if mid < 8:
                abalone[j,-1] = 1
            elif mid < 11:
                abalone[j,-1] = 2
            elif mid < 16:
                abalone[j,-1] = 3
            else:
                abalone[j,-1] = 4
        ### change to float data type
        abalone = abalone.astype("float")
        return(abalone)

def heatmap(X_input, t):
    if t == 0:
        string = "Regression"
    else:
        string = "Classifier"
    corrmatrix = np.corrcoef(X_input.T)
    plt.figure(figsize=(9,9))
    heat_map = sns.heatmap( corrmatrix, linewidth = 1 , annot = True)
    heat_map.set_xticklabels(['Adult','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
    heat_map.set_yticklabels(['Adult','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
    plt.title( "The heatmap for correlation coefficient" )
    plt.savefig('%s_corr_heatmap.png'%string)
    plt.clf()
### this is for pure tree model ###
def tree_model_class(train_x, train_y, test_x, test_y, depth = 1 , msl = 1, ccp=0, treetype = 'Pure'):
    if treetype == 0:
        model = DecisionTreeClassifier(max_depth=depth, criterion = 'gini')
    if treetype == 1:
        model = DecisionTreeClassifier(max_depth=depth, criterion = 'gini',min_samples_leaf = msl )
    else:
        model = DecisionTreeClassifier(max_depth=depth, criterion = 'gini',min_samples_leaf = msl, ccp_alpha = ccp)
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    return(acc_tr,acc_te)
### This is for gradient boosting ###
def gradboost(depth, times,train_x,test_x,train_y,test_y):
    classifier = list()
    list_y = list([train_y])
    for i in range(times):
        classifier.append(DecisionTreeClassifier(max_depth = depth))
    for i in range(times):
        classifier[i].fit(train_x, list_y[i])
        ynew = list_y[i] - classifier[i].predict(train_x)
        list_y.append(ynew)
    ytrain_pre = sum(tree.predict(train_x) for tree in classifier )
    ytest_pre = sum(tree.predict(test_x) for tree in classifier )
    acc_tr = accuracy_score(y_true = train_y, y_pred = ytrain_pre )
    acc_te = accuracy_score(y_true = test_y, y_pred = ytest_pre)
    return(acc_tr,acc_te)
def boost_depth(depth,train_x,test_x,train_y,test_y):
    all_tr = np.zeros(15)
    all_te = np.zeros(15)
    for i in range(1,16):
        each_tr = np.zeros(3)
        each_te = np.zeros(3)
        for j in range(3):
            train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
            (acc_tr,acc_te) = gradboost(depth, i,train_x,test_x,train_y,test_y)
            each_tr[j] = acc_tr
            each_te[j] = acc_te
        all_tr[i-1] = each_tr.mean()
        all_te[i-1] = each_te.mean()
    return (all_tr,all_te)


def roc_curve_f(train_x,test_x,train_y,test_y,model,name):
    detree = model.fit(train_x,train_y)
    fig = plt.figure()
    for i in range(4):
        predict_y = detree.predict_proba(test_x)
        a = np.where(test_y==i+1,1,0)
        fpr, tpr, _ = roc_curve(a, predict_y[:,i])
        plt.subplot(2, 2, i+1)
        plt.tight_layout()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1])
        plt.title("Roc Curve of Class %s in %s"%(i+1,name))
    plt.savefig('%s_roc_each.png'%name)
    plt.clf()

    fig = plt.figure()
    for i in range(4):
        predict_y = detree.predict_proba(test_x)
        a = np.where(test_y==i+1,1,0)
        fpr, tpr, _ = roc_curve(a, predict_y[:,i])
        plt.plot(fpr, tpr,label = "Class %s"%(i+1))
        print('AUC for Class %s in %s is '%(i+1,name),auc(fpr, tpr))
    plt.legend(['Class1', 'Class2', 'Class3', 'Class4'], loc='lower right')
    plt.title("%s"%(name))
    plt.savefig('%s_roc_all.png'%name)
    plt.clf()

def tree_model_regress(train_x, train_y, test_x, test_y, depth = 1 , msl = 1, ccp=0, treetype = 'Pure'):
    if treetype == 0:
        model = DecisionTreeRegressor(max_depth=depth, criterion = 'squared_error')
    if treetype == 1:
        model = DecisionTreeRegressor(max_depth=depth, criterion = 'squared_error',min_samples_leaf = msl )
    else:
        model = DecisionTreeRegressor(max_depth=depth, criterion = 'squared_error',min_samples_leaf = msl, ccp_alpha = ccp)
    detree = model.fit(train_x,train_y)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    return(mse_tr,mse_te)

def mse_graph(train_x,test_x,train_y,test_y,model,name):
    plt.scatter(test_y,detree.predict(test_x))
    plt.title('Real-Predict')
    plt.xlabel("Real value")
    plt.ylabel('Predict Value')
    plt.savefig('%s_roc_each.png'%name)
    plt.clf()
    plt.scatter(test_y,test_y - detree.predict(test_x))
    plt.title('Predict-Residual')
    plt.xlabel("Predict value")
    plt.ylabel('Residual')
    plt.savefig('%s_roc_each.png'%name)
    plt.clf()

abalone = abalone = read_data(1)
x = abalone[:,:-1]
y = abalone[:,-1]

### visualisation of multi ###
heatmap(abalone,1)

k = ['0-7','8-10','11-15','16+']
a = [sum(abalone[:,-1]==1),sum(abalone[:,-1]==2),sum(abalone[:,-1]==3),sum(abalone[:,-1]==4)]
plt.bar(k,a)
plt.title('Ring age frequency')
plt.xlabel("Ring age")
plt.ylabel('Frequency')
plt.savefig('Frequency_Classification.png')
plt.clf()


fig = plt.figure()
plt.subplot(2, 2, 1)
plt.hist(abalone[:,1])
plt.title("Length")
plt.ylabel('Frequency')
plt.subplot(2, 2, 2)
plt.hist(abalone[:,2])
plt.title("Diameter")
plt.subplot(2, 2, 3)
plt.hist(abalone[:,3])
plt.xlabel("Height")
plt.ylabel('Frequency')
plt.subplot(2, 2, 4)
plt.hist(abalone[:,4])
plt.xlabel("Whole weight")
plt.savefig('Classification Part')
plt.clf()

### decision tree ###
all_tr = np.zeros(15)
all_te = np.zeros(15)
for i in range(1,16):
    each_tr = np.zeros(5)
    each_te = np.zeros(5)
    for j in range(5):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (acc_tr,acc_te) = tree_model_class(train_x,train_y,test_x,test_y,depth = i)
        each_tr[j] = acc_tr
        each_te[j] = acc_te
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()
plt.plot([i for i in range(1,16)],all_tr,c='orange')
plt.title('Decision Tree for Train Data')
plt.xlabel('Max_Depth')
plt.ylabel('Train Accuracy')
plt.show()
plt.savefig('Acc_Train_Pure.png')
plt.clf()
max_id = np.where(all_te == max(all_te))[0][0]
plt.plot([i for i in range(1,16)],all_te,c='orange')
plt.title('Decision Tree for Test Data')
plt.xlabel('Max_Depth')
plt.ylabel('Test Accuracy')
plt.vlines(max_id+1,min(all_te),max(all_te),linestyles='dashed', colors='red')
plt.savefig('Acc_Test_Pure.png')
plt.clf()
plt.plot([i for i in range(1,16)],all_tr-all_te,c='orange')
plt.title('Accuracy Difference for train and test')
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy Difference')
plt.savefig('Acc_Difference_Pure.png')
plt.clf()
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = DecisionTreeClassifier(max_depth=5, criterion = 'gini')
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*",end='',flush=True)
print("*")
print("The mean of accuracy for train in DT: ",actr.mean())
print("The standard deviation of accuracy for train in DT: ",actr.std())
print("The mean of accuracy for test in DT: ",acte.mean())
print("The standard deviation of accuracy for test in DT: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeClassifier(max_depth=5, criterion = 'gini')
detree = model.fit(train_x,train_y)
roc_curve_f(train_x,test_x,train_y,test_y,model,"Pure DT")

### pre posting ###
all_tr = np.zeros(15)
all_te = np.zeros(15)
for i in range(1,16):
    each_tr = np.zeros(5)
    each_te = np.zeros(5)
    for j in range(5):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (acc_tr,acc_te) = tree_model_class(train_x,train_y,test_x,test_y,depth = 5, msl = 5*(i+1))
        each_tr[j] = acc_tr
        each_te[j] = acc_te
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()
plt.plot([i*5 for i in range(1,16)],all_tr,c='orange')
plt.title('Pre-pruning Decision Tree for Train Data')
plt.xlabel('min_samples_leaf')
plt.ylabel('Train Accuracy')
plt.savefig('Acc_Train_Pre_Pruning.png')
plt.clf()

max_id = np.where(all_te == max(all_te))[0][0]
plt.plot([i*5 for i in range(1,16)],all_te,c='orange')
plt.title('Decision Tree for Test Data')
plt.xlabel('min_samples_leaf')
plt.ylabel('Test Accuracy')
plt.vlines(max_id*5+5,min(all_te),max(all_te),linestyles='dashed', colors='red')
plt.savefig('Acc_Test_Pre_Pruning.png')
plt.clf()
plt.plot([i*5 for i in range(1,16)],all_tr-all_te,c='orange')
plt.title('Accuracy Difference for train and test')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy Difference')
plt.savefig('Acc_Difference_Pre_Pruning.png')
plt.clf()
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = DecisionTreeClassifier(max_depth=5, criterion = 'gini',min_samples_leaf = 50)
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*",end='',flush=True)
print("*")
print("The mean of accuracy for train in pre_pruning: ",actr.mean())
print("The standard deviation of accuracy for train in pre_pruning: ",actr.std())
print("The mean of accuracy for test in pre_pruning: ",acte.mean())
print("The standard deviation of accuracy for test in pre_pruning: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeClassifier(max_depth=5, criterion = 'gini',min_samples_leaf = 50)
detree = model.fit(train_x,train_y)
roc_curve_f(train_x,test_x,train_y,test_y,model,"Pre_Pruning")

### post_pruning ###

train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeClassifier(max_depth=5, criterion = 'gini',min_samples_leaf = 50)
path = model.cost_complexity_pruning_path(train_x,train_y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(max_depth = 5,random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(train_x,train_y)
    clfs.append(clf)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
train_scores = [clf.score(train_x, train_y) for clf in clfs]
test_scores = [clf.score(test_x, test_y) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig('CCP_ALPHA classification.png')
plt.clf()
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = DecisionTreeClassifier(max_depth=5, criterion = 'gini',min_samples_leaf = 50,ccp_alpha = 0.0035)
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*",end='',flush=True)
print("*")
print("The mean of accuracy for train in post_pruning: ",actr.mean())
print("The standard deviation of accuracy for train in post_pruning: ",actr.std())
print("The mean of accuracy for test in post_pruning: ",acte.mean())
print("The standard deviation of accuracy for test in post_pruning: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeClassifier(max_depth=5, criterion = 'gini',min_samples_leaf = 50,ccp_alpha = 0.0035)
detree = model.fit(train_x,train_y)
roc_curve_f(train_x,test_x,train_y,test_y,model,"Post_Pruning")

### Random Forest ###
def forest_model(train_x,train_y,test_x,test_y,i):
    model = RandomForestClassifier(n_estimators = i,criterion = 'gini',max_depth = 2)
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    return(acc_tr,acc_te)
all_tr = np.zeros(40)
all_te = np.zeros(40)
for i in range(10,410,10):
    each_tr = np.zeros(1)
    each_te = np.zeros(1)
    for j in range(1):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (acc_tr,acc_te) = forest_model(train_x,train_y,test_x,test_y,i)
        each_tr[j] = acc_tr
        each_te[j] = acc_te
    k = int(i/10 - 1)
    all_tr[k] = each_tr.mean()
    all_te[k] = each_te.mean()
plt.plot([i for i in range(10,410,10)],all_tr,c='orange')
plt.title('Random Forest for Train Data')
plt.xlabel('Number of Tree')
plt.ylabel('Train Accuracy')
plt.savefig('Acc_Train_RandomForest.png')
plt.clf()
max_id = np.where(all_te == max(all_te))[0][0]
plt.plot([i for i in range(10,410,10)],all_te,c='orange')
plt.title('Random Forest for Test Data')
plt.xlabel('Number of Tree')
plt.ylabel('Test Accuracy')
plt.vlines((max_id+1)*10,min(all_te),max(all_te),linestyles='dashed', colors='red')
plt.savefig('Acc_TestRandomForest.png')
plt.clf()
actr = np.zeros(10)
acte = np.zeros(10)
for i in range(10):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = RandomForestClassifier(n_estimators = 180,criterion = 'gini',max_depth = 2)
    detree = model.fit(train_x,train_y)
    acc_tr = accuracy_score(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y, y_pred=detree.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("***",end="", flush= True)
print('*')
print("The mean of accuracy for train in Random Forest: ",actr.mean())
print("The standard deviation of accuracy for train in Random Forest: ",actr.std())
print("The mean of accuracy for test in Random Forest: ",acte.mean())
print("The standard deviation of accuracy for test in Random Forest: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = RandomForestClassifier(n_estimators = 180,criterion = 'gini',max_depth = 2)
detree = model.fit(train_x,train_y)
roc_curve_f(train_x,test_x,train_y,test_y,model,"RF") 

### GBDT ###
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
te = list()
tr = list()
for i in range(4):
    all_tr,all_te = boost_depth(i+1,train_x,test_x,train_y,test_y)
    te.append(all_te)
    tr.append(all_tr)

fig = plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.tight_layout()
    plt.plot([i for i in range(1,16)],tr[i])
    plt.xlabel("Number of iteration")
    plt.title("Depth for weak leaner: %s in train"%(i+1) )
plt.savefig("Number of iterations in train")
plt.clf()

for i in range(4):
    max_id = np.where(te[i]== max(te[i]))[0][0]
    plt.subplot(2, 2, i+1)
    plt.tight_layout()
    plt.plot([i for i in range(1,16)],te[i])
    plt.vlines(max_id+1,min(te[i]),max(te[i]),linestyles='dashed', colors='red')
    plt.xlabel("Number of iteration")
    plt.title("Depth for weak leaner: %s in test"%(i+1) )
plt.savefig("Number of iterations in train")
plt.clf()
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    (acc_tr,acc_tr) = gradboost(2, 8,train_x,test_x,train_y,test_y)
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*", end = "", flush = True)
print("*")
print("The mean of accuracy for train in GBDT: ",actr.mean())
print("The standard deviation of accuracy for train in GBDT: ",actr.std())
print("The mean of accuracy for test in GBDT: ",acte.mean())
print("The standard deviation of accuracy for test in GBDT: ",acte.std())

### XGboost ###
def xgradboost(depth, times,train_x,test_x,train_y,test_y):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    detree = XGBClassifier(max_depth=depth, learning_rate=0.3, n_estimators=times,reg_alpha=0.5)
    detree.fit(train_x,train_y-1)
    detree.predict(train_x)
    detree.predict(test_x)
    acc_tr = accuracy_score(y_true=train_y-1, y_pred=detree.predict(train_x))
    acc_te = accuracy_score(y_true=test_y-1, y_pred=detree.predict(test_x))
    return(acc_tr,acc_te)
def boost_depth(depth,train_x,test_x,train_y,test_y):
    all_tr = np.zeros(30)
    all_te = np.zeros(30)
    for i in range(1,31):
        each_tr = np.zeros(5)
        each_te = np.zeros(5)
        for j in range(5):
            train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
            (acc_tr,acc_te) = xgradboost(depth, i,train_x,test_x,train_y-1,test_y-1)
            each_tr[j] = acc_tr
            each_te[j] = acc_te
        all_tr[i-1] = each_tr.mean()
        all_te[i-1] = each_te.mean()
    return (all_tr,all_te)

te = list()
tr = list()
for i in range(4):
    all_tr,all_te = boost_depth(i+1,train_x,test_x,train_y,test_y)
    te.append(all_te)
    tr.append(all_tr)

fig = plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.tight_layout()
    plt.plot([i for i in range(1,31)],tr[i])
    plt.xlabel("Number of iteration")
    plt.title("Depth  in XGBoost: %s in train"%(i+1) )
plt.show()

for i in range(4):
    max_id = np.where(te[i]== max(te[i]))[0][0]
    plt.subplot(2, 2, i+1)
    plt.tight_layout()
    plt.plot([i for i in range(1,31)],te[i])
    plt.vlines(max_id+1,min(te[i]),max(te[i]),linestyles='dashed', colors='red')
    plt.xlabel("Number of iteration")
    plt.title("Depth in XGBoost: %s in test"%(i+1) )
plt.show()

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.tight_layout()
    plt.plot([i for i in range(1,31)],tr[i]-te[i])
    plt.xlabel("Number of iteration")
    plt.title("Depth in XGBoost: %s for difference"%(i+1) )
plt.show()

actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    (acc_tr,acc_tr) = xgradboost(2, 27,train_x,test_x,train_y-1,test_y-1)
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*", end = "", flush = True)
print("*")
print("The mean of accuracy for train in XGboost: ",actr.mean())
print("The standard deviation of accuracy for train in XGboost: ",actr.std())
print("The mean of accuracy for test in XGboost: ",acte.mean())
print("The standard deviation of accuracy for test in XGboost: ",acte.std())

train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model= XGBClassifier(max_depth=2, learning_rate=0.3, n_estimators=27,reg_alpha=0.5)
roc_curve_f(train_x,test_x,train_y-1,test_y,model,"XGBoost")

### snn ###
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='sgd', max_iter=1000, learning_rate_init=0.4)
    model.fit(train_x, train_y)
    acc_tr = accuracy_score(y_true = train_y, y_pred = model.predict(train_x) )
    acc_te = accuracy_score(y_true = test_y, y_pred = model.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*", end = "", flush = True)
print("*")
print("The mean of accuracy for train in sgd snn: ",actr.mean())
print("The standard deviation of accuracy for train in sgd snn: ",actr.std())
print("The mean of accuracy for test in sgd snn: ",acte.mean())
print("The standard deviation of accuracy for test in sgd snn: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model= MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='sgd', max_iter=1000, learning_rate_init=0.4)
roc_curve_f(train_x,test_x,train_y,test_y,model,"sgd snn")

### snn adam ###
actr = np.zeros(30)
acte = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.1)
    model.fit(train_x, train_y)
    acc_tr = accuracy_score(y_true = train_y, y_pred = model.predict(train_x) )
    acc_te = accuracy_score(y_true = test_y, y_pred = model.predict(test_x))
    each_tr[j] = acc_tr
    actr[i] = acc_tr
    acte[i] = acc_te
    print("*", end = "", flush = True)
print("*")
print("The mean of accuracy for train in adam snn: ",actr.mean())
print("The standard deviation of accuracy for train in adam snn: ",actr.std())
print("The mean of accuracy for test in adam snn: ",acte.mean())
print("The standard deviation of accuracy for test in adam snn: ",acte.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model= MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', solver='adam', max_iter=1000, learning_rate_init=0.1)
roc_curve_f(train_x,test_x,train_y,test_y,model,"adam snn")

### Regression Problem ###
## Visulization ##
abalone = abalone = read_data(0)
heatmap(abalone,0)
x = abalone[:,:-1]
y = abalone[:,-1]
plt.hist(abalone[:,-1])
plt.title('Ring age frequency')
plt.xlabel("Ring age")
plt.ylabel('Frequency')
plt.savefig("Frequency_Regression")
plt.clf()
plt.close()

### Decision Tree ###
all_tr = np.zeros(15)
all_te = np.zeros(15)
for i in range(1,16):
    each_tr = np.zeros(5)
    each_te = np.zeros(5)
    for j in range(5):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (mse_tr,mse_te) = tree_model_regress(train_x,train_y,test_x,test_y,depth = i)
        each_tr[j] = np.sqrt(mse_tr)
        each_te[j] = np.sqrt(mse_te)
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()
plt.plot([i for i in range(1,16)],all_tr,c='orange')
plt.title('Decision Tree for Train Data')
plt.xlabel('Max Depth')
plt.ylabel('Train Rmse')
plt.savefig('Rmse Train Pure.png')
plt.clf()

min_id = np.where(all_te == min(all_te))[0][0]
plt.plot([i for i in range(1,16)],all_te,c='orange')
plt.title('Decision Tree for Test Data')
plt.xlabel('Max Depth')
plt.ylabel('Test Rmse')
plt.vlines(min_id+1,2.2,min(all_te),linestyles='dashed', colors='red')
plt.savefig('Rmse Test Pure.png')
plt.clf()

plt.plot([i for i in range(1,16)],all_tr-all_te,c='orange')
plt.title('Rmse Difference for train and test')
plt.xlabel('Max Depth')
plt.ylabel('Rmse Difference')
plt.savefig('Rmse Difference Pure.png')
plt.clf()

actr = np.zeros(30)
acte = np.zeros(30)
r2tr = np.zeros(30)
r2te = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = DecisionTreeRegressor(max_depth=5, criterion = 'squared_error')
    detree = model.fit(train_x,train_y)
    acc_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    actr[i] = np.sqrt(acc_tr)
    acte[i] = np.sqrt(acc_te)
    r2_tr = r2_score(y_true=train_y, y_pred=detree.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred=detree.predict(test_x))
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("*",end='',flush=True)
print("*")
print("The mean of Rmse for train in DT: ",actr.mean())
print("The standard deviation of Rmse for train in DT: ",actr.std())
print("The mean of Rmse for test in DT: ",acte.mean())
print("The standard deviation of Rmse for test in DT: ",acte.std())
print("The mean of R2 for train in DT: ",r2tr.mean())
print("The standard deviation of R2 of accuracy for train in DT: ",r2tr.std())
print("The mean of R2 for test in DT: ",r2te.mean())
print("The standard deviation of R2 for test in DT: ",r2te.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeRegressor(max_depth=5, criterion = 'squared_error')
detree = model.fit(train_x,train_y)
mse_graph(train_x,test_x,train_y,test_y,model,"Pure DT")

### pre-prununing ###
all_tr = np.zeros(15)
all_te = np.zeros(15)
for i in range(1,16):
    each_tr = np.zeros(20)
    each_te = np.zeros(20)
    for j in range(20):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        model =  DecisionTreeRegressor(max_depth=5, criterion = 'squared_error',min_samples_leaf = 5 *(i))
        detree = model.fit(train_x,train_y)
        mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
        mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
        each_tr[j] = np.sqrt(mse_tr)
        each_te[j] = np.sqrt(mse_te)
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()
plt.plot([i*5 for i in range(1,16)],all_tr,c='orange')
plt.title('Decision Tree for Train Data')
plt.xlabel('Minimum Samples Leaf')
plt.ylabel('Train RMSE')
plt.savefig('Rmse_pre_pruning_train.png')
plt.clf()
max_id = np.where(all_te == min(all_te))[0][0]
plt.plot([i*5 for i in range(1,16)],all_te,c='orange')
plt.title('Decision Tree for Test Data')
plt.xlabel('Minimum Samples Leaf')
plt.ylabel('Test Rmse')
plt.vlines((max_id+1)*5,2.3,min(all_te),linestyles='dashed', colors='red')
plt.savefig('Rmse_pre_pruning_test.png')
plt.clf()

msetr = np.zeros(30)
msete = np.zeros(30)
r2tr = np.zeros(30)
r2te = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = DecisionTreeRegressor(max_depth=5, criterion = 'squared_error',min_samples_leaf = 35) 
    detree = model.fit(train_x,train_y)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    r2_tr = r2_score(y_true=train_y, y_pred=detree.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred=detree.predict(test_x))
    msetr[i] = np.sqrt(mse_tr)
    msete[i] = np.sqrt(mse_te)
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("*",end='',flush=True)
print("*")
print("The mean of Rmse for train: ",msetr.mean())
print("The standard deviation of Rmse for train in Pre Pruning: ",msetr.std())
print("The mean of Rmse for test in Pre Pruning: ",msete.mean())
print("The standard deviation of Rmse for test in Pre Pruning: ",msete.std())
print("The mean of R2 for train in Pre Pruning: ",r2tr.mean())
print("The standard deviation of R2 of accuracy for train inPre Pruning: ",r2tr.std())
print("The mean of R2 for test in Pre Pruning: ",r2te.mean())
print("The standard deviation of R2 for test in Pre Pruning: ",r2te.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeRegressor(max_depth=5, criterion = 'squared_error',min_samples_leaf=35)
detree = model.fit(train_x,train_y)
mse_graph(train_x,test_x,train_y,test_y,model,"Pre Pruning")

### post-pruned ###
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = DecisionTreeRegressor(max_depth=5, criterion = 'squared_error',min_samples_leaf = 35) 
path = model.cost_complexity_pruning_path(train_x,train_y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas: 
    clf =  DecisionTreeRegressor(max_depth=5, criterion = 'squared_error',ccp_alpha=ccp_alpha,random_state=0)
    clf.fit(train_x,train_y)
    clfs.append(clf)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
train_scores = [mean_squared_error(y_true=train_y, y_pred=clf.predict(train_x)) for clf in clfs]
test_scores = [mean_squared_error(y_true=test_y, y_pred=clf.predict(test_x)) for clf in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Rmse")
ax.set_title("Train vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig("alpha")
plt.clf()

### random forest ###
def forest_model_r(train_x,train_y,test_x,test_y,i):
    model = RandomForestRegressor(n_estimators = i,criterion = 'squared_error',max_depth = 2)
    detree = model.fit(train_x,train_y)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    return(np.sqrt(mse_tr),np.sqrt(mse_te))
all_tr = np.zeros(40)
all_te = np.zeros(40)
for i in range(10,410,10):
    each_tr = np.zeros(2)
    each_te = np.zeros(2)
    for j in range(2):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (mse_tr,mse_te) = forest_model_r(train_x,train_y,test_x,test_y,i)
        each_tr[j] = mse_tr
        each_te[j] = mse_te
    k = int(i/10 - 1)
    all_tr[k] = each_tr.mean()
    all_te[k] = each_te.mean()
plt.plot([i for i in range(10,410,10)],all_tr,c='orange')
plt.title('Random Forest for Train Data')
plt.xlabel('Number of Tree')
plt.ylabel('MSE')
plt.show()
plt.savefig('MSE_Train_RF.png')
plt.clf()
max_id = np.where(all_te == min(all_te))[0][0]
plt.plot([i for i in range(10,410,10)],all_te,c='orange')
plt.title('Random Forest for Test Data')
plt.xlabel('Number of Tree')
plt.ylabel('RMSE')
plt.vlines((max_id+1)*10,2.2,min(all_te),linestyles='dashed', colors='red')
plt.show()
plt.savefig('MSE_Test_RF.png')
plt.clf()

msetr = np.zeros(10)
msete = np.zeros(10)
r2tr = np.zeros(10)
r2te = np.zeros(10)
for i in range(10):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = RandomForestRegressor(n_estimators = 210,criterion = 'squared_error',max_depth = 2)
    detree = model.fit(train_x,train_y)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    r2_tr = r2_score(y_true=train_y, y_pred=detree.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred=detree.predict(test_x))
    msetr[i] = np.sqrt(mse_tr)
    msete[i] = np.sqrt(mse_te)
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("***",end="",flush=True)
print("*")
print("The mean of Rmse for train in Random Forest: ",msetr.mean())
print("The standard deviation of Rmse for train in Random Forest: ",msetr.std())
print("The mean of Rmse for test in Random Forest: ",msete.mean())
print("The standard deviation of Rmse for test in Random Forest: ",msete.std())
print("The mean of R2 for train in Random Forest: ",r2tr.mean())
print("The standard deviation of R2 for train in Random Forest: ",r2tr.std())
print("The mean of R2 for test in Random Forest: ",r2te.mean())
print("The standard deviation of R2 for test in Random Forest: ",r2te.std())
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = RandomForestRegressor(n_estimators = 210,criterion = 'squared_error',max_depth = 2)
mse_graph(train_x,test_x,train_y,test_y,model,"Random Forest")

### gradient boosting ###

def gradient_boost_r(train_x,train_y,test_x,test_y,i):
    model = GradientBoostingRegressor(n_estimators = i,loss = 'squared_error',max_depth = 2)
    detree = model.fit(train_x,train_y)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    return(mse_tr,mse_te)
all_tr = np.zeros(50)
all_te = np.zeros(50)
for i in range(1,51):
    each_tr = np.zeros(5)
    each_te = np.zeros(5)
    for j in range(5):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (mse_tr,mse_te) = gradient_boost_r(train_x,train_y,test_x,test_y,i)
        each_tr[j] = np.sqrt(mse_tr)
        each_te[j] = np.sqrt(mse_te)
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()
plt.plot([i for i in range(1,51)],all_tr,c='orange')
plt.title('Gradient boosting for Train Data')
plt.xlabel('Number of iteration')
plt.ylabel('RMSE')
plt.show()
plt.savefig('RMSE_Train_GBDT.png')
plt.clf()
max_id = np.where(all_te == min(all_te))[0][0]
plt.plot([i for i in range(1,51)],all_te,c='orange')
plt.title('Gradient boosting for Test Data')
plt.xlabel('Number of iteration')
plt.ylabel('RMSE')
plt.vlines((max_id+1),2.3,min(all_te),linestyles='dashed', colors='red')
plt.savefig('RMSE_Test_GBDT.png')
plt.clf()
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = GradientBoostingRegressor(n_estimators = 43,loss = 'squared_error',max_depth = 2)
mse_graph(train_x,test_x,train_y,test_y,model,"GBDT")

actr = np.zeros(30)
acte = np.zeros(30)
r2tr = np.zeros(30)
r2te = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = GradientBoostingRegressor(n_estimators = 43,loss = 'squared_error',max_depth = 2)
    detree = model.fit(train_x,train_y)
    acc_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    actr[i] = np.sqrt(acc_tr)
    acte[i] = np.sqrt(acc_te)
    r2_tr = r2_score(y_true=train_y, y_pred=detree.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred=detree.predict(test_x))
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("*",end='',flush=True)
print("*")
print("The mean of Rmse for train in GBRT: ",actr.mean())
print("The standard deviation of Rmse for train in GBRT: ",actr.std())
print("The mean of Rmse for test in GBRT: ",acte.mean())
print("The standard deviation of Rmse for test in GBRT: ",acte.std())
print("The mean of R2 for train in GBRT: ",r2tr.mean())
print("The standard deviation of R2 of accuracy for train in GBRT: ",r2tr.std())
print("The mean of R2 for test in GBRT: ",r2te.mean())
print("The standard deviation of R2 for test in GBRT: ",r2te.std())
### XGBDT ###
def xgradboost(depth, times,train_x,test_x,train_y,test_y):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    detree = XGBRegressor(max_depth=depth, learning_rate=0.3, n_estimators=times,reg_alpha=0.5)
    detree.fit(train_x,train_y)
    detree.predict(train_x)
    detree.predict(test_x)
    mse_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    mse_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    return(mse_tr,mse_te)
all_tr = np.zeros(40)
all_te = np.zeros(40)
for i in range(1,41):
    each_tr = np.zeros(5)
    each_te = np.zeros(5)
    for j in range(5):
        train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
        (mse_tr,mse_te) = xgradboost(2,i,train_x,test_x,train_y,test_y)
        each_tr[j] = np.sqrt(mse_tr)
        each_te[j] = np.sqrt(mse_te)
    all_tr[i-1] = each_tr.mean()
    all_te[i-1] = each_te.mean()

plt.plot([i for i in range(1,41)],all_tr,c='orange')
plt.title('XGBDT  for Train Data')
plt.xlabel('Number of Iterations')
plt.ylabel('Rmse')
plt.show()
plt.savefig('Rmse_Train_XGBDT.png')
plt.clf()
max_id = np.where(all_te == min(all_te))[0][0]
plt.plot([i for i in range(1,41)],all_te,c='orange')
plt.title('XGBDT for Test Data')
plt.xlabel('Number of Iterations')
plt.ylabel('TMSE')
plt.vlines((max_id+1),min(all_te)-0.5,min(all_te),linestyles='dashed', colors='red')
plt.savefig('MSE_Test_XGBDT.png')
plt.clf()
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = XGBRegressor(max_depth=depth, learning_rate=0.3, n_estimators=21,reg_alpha=0.5)
mse_graph(train_x,test_x,train_y,test_y,model,"XGBRT")
actr = np.zeros(30)
acte = np.zeros(30)
r2tr = np.zeros(30)
r2te = np.zeros(30)
for i in range(30):
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
    model = GradientBoostingRegressor(n_estimators = 43,loss = 'squared_error',max_depth = 2)
    detree = model.fit(train_x,train_y)
    acc_tr = mean_squared_error(y_true=train_y, y_pred=detree.predict(train_x))
    acc_te = mean_squared_error(y_true=test_y, y_pred=detree.predict(test_x))
    actr[i] = np.sqrt(acc_tr)
    acte[i] = np.sqrt(acc_te)
    r2_tr = r2_score(y_true=train_y, y_pred=detree.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred=detree.predict(test_x))
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("*",end='',flush=True)
print("*")
print("The mean of Rmse for train in XGBRT: ",actr.mean())
print("The standard deviation of Rmse for train in XGBRT: ",actr.std())
print("The mean of Rmse for test in XGBRT: ",acte.mean())
print("The standard deviation of Rmse for test in XGBRT: ",acte.std())
print("The mean of R2 for train in GBRT: ",r2tr.mean())
print("The standard deviation of R2 of accuracy for train in XGBRT: ",r2tr.std())
print("The mean of R2 for test in XGBRT: ",r2te.mean())
print("The standard deviation of R2 for test in XGBRT: ",r2te.std())

### snn ###
msetr = np.zeros(10)
msete = np.zeros(10)
r2tr = np.zeros(10)
r2te = np.zeros(10)
train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
for i in range(10):
    model = MLPRegressor(hidden_layer_sizes=(20,14), activation='logistic', solver='adam', max_iter=1000, learning_rate_init=0.01)
    snn =  model.fit(train_x, train_y)
    mse_tr = mean_squared_error(y_true = train_y, y_pred = snn.predict(train_x) )
    mse_te = mean_squared_error(y_true = test_y, y_pred = snn.predict(test_x))
    msetr[i] = np.sqrt(mse_tr)
    msete[i] = np.sqrt(mse_te)
    r2_tr = r2_score(y_true=train_y, y_pred = snn.predict(train_x))
    r2_te = r2_score(y_true=test_y, y_pred = snn.predict(test_x))
    r2tr[i] = r2_tr
    r2te[i] = r2_te
    print("***",end="",flush=True)
print("*")
print("The mean of mse for train in sgd snn: ",msetr.mean())
print("The standard deviation of mse for train in sgd snn: ",msetr.std())
print("The mean of mse for test in sgd snn: ",msete.mean())
print("The standard deviation of mse for test in sgd snn: ",msete.std())
print("The mean of R2 for train in sgd snn: ",r2tr.mean())
print("The standard deviation of R2 of accuracy for train in sgd snn: ",r2tr.std())
print("The mean of R2 for test in sgd snn: ",r2te.mean())
print("The standard deviation of R2 for test in sgd snn: ",r2te.std())

train_x,test_x,train_y,test_y = train_test_split(x, y, test_size = 0.4)
model = MLPRegressor(hidden_layer_sizes=(20,14), activation='logistic', solver='adam', max_iter=1000, learning_rate_init=0.01)
mse_graph(train_x,test_x,train_y,test_y,model,"SNN sgd")

species = ("Class1", "Class2", "Class3","Class4")
penguin_means = {
    'DT': (0.935,0.714,0.754,0.807),
    'Pre': (0.934, 0.715, 0.761, 0.847),
    'Post': (0.923, 0.722, 0.729,0.834),
    "RF": (0.9304,0.6732,0.741,0.791),
    "XGBDT": (0.945,0.779,0.782,0.889),
    "SNN":(0.946,0.789,0.805,0.921)
}
x = np.arange(len(species)) 
width = 0.1  
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1
ax.set_ylabel('AUC')
ax.set_title('AUC for different class and method')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=5)
ax.set_ylim(0.6, 1.1)
plt.savefig('AUC_Plot')
plt.clf()

###export_graphviz(detree, out_file='tree.dot',feature_names= ['Adult','Length','Diameter','Height','Whole','Shucked','Viscera','Shell'],class_names = ['Class1','Class2','Class3','Class4'],rounded = True, proportion = False,  filled = True)
###call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
###plt.figure(figsize = (15, 15))
###plt.imshow(plt.imread('tree.png'))
###plt.axis('off');
