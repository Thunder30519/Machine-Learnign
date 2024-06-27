import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
from sklearn.neural_network import MLPClassifier

rd.seed(20231010)
### extract data from abalone data set
def data_clean(j):
    abalone = pd.read_csv('data/abalone.data',sep = ',',header=None)
    abalone = abalone.values
    ### convert srt type to int type for sex data and 0 for male, 1 for female ,2 for infant
    if j == 0:
        abalone[:,0] = np.where(abalone[:,0]=='M',0,np.where(abalone[:,0]=='F',1,2))
        ### delete the infant part
        abalone = np.delete(abalone,abalone[:,0]==2,axis=0)
    else:
        ### combine the male and female as adult
        abalone[:,0] = np.where(abalone[:,0]=='I',0,1)
    ### there are two ways to process data, delete infant or combine male and female as adult.
    ### This function can retrun two different data structure.
    
    ### cleaning data due to there is some 0 value in the dataset 
    for i in range(1,abalone.shape[1]):
        abalone = np.delete(abalone, abalone[:,i] == 0,axis = 0) 
    
    return(abalone)



### construct a correlation map and heatmap
def heatmap(X_input,i):
    corrmatrix = np.corrcoef(X_input.T)
    ### draw heatmap for different condition, one is sex term another is adult term.
    if i ==1:
        plt.figure(figsize=(9,9))
        heat_map = sns.heatmap( corrmatrix, linewidth = 1 , annot = True)
        heat_map.set_xticklabels(['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
        heat_map.set_yticklabels(['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
        plt.title( "The heatmap for correlation coefficient" )
        plt.savefig(f'{i}_corr_heatmap.png')
        plt.clf()
    else:
        ### else part is for adult and infant 
        plt.figure(figsize=(9,9))
        heat_map = sns.heatmap( corrmatrix, linewidth = 1 , annot = True)
        heat_map.set_xticklabels(['Adult','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
        heat_map.set_yticklabels(['Adult','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings age'])
        plt.title( "The heatmap for correlation coefficient" )
        plt.savefig(f'{i}_corr_heatmap.png')
        plt.clf()
    
### find most correlated features ###
def find_corr(X_input):
    corrmatrix = np.corrcoef(X_input.T)
    ### take absolute value to find most correlated two features
    corr_new = np.abs(corrmatrix)[:-1,-1]
    nu_feature = len(corr_new)
    ### find correlations which are last one and last two small
    first_max = np.where(np.argsort(corr_new) == nu_feature-1 )[0][0]
    second_max = np.where(np.argsort(corr_new) == nu_feature-2)[0][0]
    return(first_max,second_max)


### print scatter plot and histogram plot simutaneously###
def scatter_plot(X_input):
    most_corr = find_corr(X_input)### function to find most 2 correlated features
    
    print('Below 2 features are most correlated with ring ages')
    text = ['Sex','Length','Diameter','Height','Whole Weight','Shucked weight','Viscera weight','Shell weight','Rings age']
    print(text[most_corr[0]],'  ',text[most_corr[1]])
    
    first_fea = X_input[:,most_corr[0]]
    second_fea = X_input[:,most_corr[1]]
    depe = X_input[:,-1]
    
    ###draw scatter plot and histogram plot
    
    plt.scatter(first_fea,depe,s=5,c='orange')
    plt.title('First Most Correlated Features')
    plt.xlabel(text[most_corr[0]])
    plt.ylabel('Ring age')
    plt.savefig('3_first_feature.png')
    plt.clf()
    
    plt.scatter(second_fea,depe,s=5,c='green')
    plt.title('Second Most Correlated Features')
    plt.xlabel(text[most_corr[1]])
    plt.ylabel('Ring age')
    plt.savefig('3_second_feature.png')
    plt.clf()
   
    plt.hist(first_fea)
    plt.title('First Most Correlated Features Histogram')
    plt.xlabel(text[most_corr[0]])
    plt.ylabel('Frequency')
    plt.savefig('4_histogram_first.png')
    plt.clf()
    
    plt.hist(second_fea)
    plt.title('Second Most Correlated Features Histogram')
    plt.xlabel(text[most_corr[1]])
    plt.ylabel('Frequency')
    plt.savefig('4_histogram_second.png')
    plt.clf()
    
    plt.hist(depe)
    plt.title('Second Most Correlated Features Histogram')
    plt.xlabel('Ring age')
    plt.ylabel('Frequency')
    plt.savefig('4_histogram_years.png')
    plt.clf()

###separate to 60/40 train/test###
def sepa_dataset(X_input,normalizing):
    ### randomly permutate the order of input data 
    if normalizing == 0:
        X_new = np.random.permutation(X_input) 
        ### get the line where we can separate it to train and set.
        ### as 60/40 split, we set before 60% data is train set and rest 40% is test
        killline = int(len(X_new)*0.6)
        train = X_new[:killline,:]
        test = X_new[killline:,:]
        print('The number of train data is  ',len(train))
        print('The number of test data is  ',len(test))
        return(train, test)  
    ### this is for normalizing part
    if normalizing == 1:
        X_input[:,:8]  = (X_input[:,:8] - X_input[:,:8].min(axis = 0))/(X_input[:,:8].max(axis=0)-X_input[:,:8].min(axis=0))
        X_new = np.random.permutation(X_input) 
        killline = int(len(X_new)*0.6)
        train = X_new[:killline,:]
        test = X_new[killline:,:]
        return(train, test) 

### A class for linear regression
class linear_regression:
    def __init__(self,tr_set,te_set,nu_input,learn_rate,nu_epoch):
        self.tr_set = tr_set  ### size = (nu_train, nu_input)
        self.te_set = te_set  ### size = (number of test, number of input)
        self.nu_input = nu_input ### number of input, or the number of features contributing to dependent value
        self.nu_train = self.tr_set.shape[0] ###number of data to train the model
        self.w = np.random.uniform(-0.5,0.5,self.nu_input).reshape(-1,1)  ### w with size (nu_input, 1)
        self.b = np.random.uniform(-0.5,0.5,1) ### b with size (1, )
        self.learn_rate = learn_rate
        self.epoch = nu_epoch
        self.rmse_vec = np.zeros(self.epoch) #initial the rmse vector as all 0 for importing the new rmse data
        self.rsquare_vec = np.zeros(self.epoch)


    ### function to calculate the rmse and rsquared
    def error_term(self,indep_vec,dep_vec):
        predicted = (indep_vec.dot(self.w) + self.b).reshape(-1,1) ###to change the shape of (nu_train,) to (nu_train,1)
        error = predicted - dep_vec  ### size of error term is (nu_train,1)      
        rmse = np.sqrt(np.sum(error**2)/self.nu_train) ### calculate RMSE
        r_square = 1 - np.sum(error**2)/np.sum((dep_vec-np.mean(dep_vec))**2) ###calculate R-square
        return(rmse,r_square)
   
    ### function for gradient descent
    def GD(self,indep_vec,dep_vec):
        predicted = (indep_vec.dot(self.w) + self.b).reshape(-1,1) ### with shape of (nu_train,1)
        error = predicted - dep_vec
        w_gradient = -(2/self.nu_train) * (indep_vec.T).dot(error) ### sum by columns then we get gradient for each feature parameters
        b_gradient = -(2/self.nu_train) * np.sum(error)
        self.w += w_gradient*self.learn_rate
        self.b += b_gradient*self.learn_rate
    
    # function for training model and return tain rmse rsquare
    def run(self):
        indep_vec = self.tr_set[:,0:self.nu_input]
        dep_vec = self.tr_set[:,self.nu_input:]
        for i in range(self.epoch):
            rmse_rsquare = self.error_term(indep_vec,dep_vec)
            self.rmse_vec[i] = rmse_rsquare[0]
            self.rsquare_vec[i] = rmse_rsquare[1]
            self.GD(indep_vec,dep_vec)
        return(rmse_rsquare)
    
    # draw some visulization plot
    def plot_r(self,number_ques):
        x_value = [i for i in range(self.epoch)]
        plt.plot(x_value,self.rmse_vec)
        plt.title('Rmse Line')
        plt.xlabel('Number of iteration')
        plt.ylabel('Rmse')
        plt.savefig(f'{number_ques}_rmse_graph')
        plt.clf()
   
    # draw plot for real value of y and predicted value of y 
    def plot_y_y(self,number_ques):
        indep_vec = self.te_set[:,0:self.nu_input]
        dep_vec = self.te_set[:,self.nu_input:]
        
        predicted = (indep_vec.dot(self.w) + self.b).reshape(-1,1)
        test_y = self.te_set[:,self.nu_input:]
        plt.scatter(test_y,predicted)
        plt.title('Real-Predicted')
        plt.xlabel('y_real')
        plt.ylabel('y_predicted')
        plt.savefig(f'{number_ques}_yreal_ypredicted')
        plt.clf()

    # check the test set's rmse and rsquare
    def test_fun(self):
        indep_vec = self.te_set[:,0:self.nu_input]
        dep_vec = self.te_set[:,self.nu_input:]
        predicted = (indep_vec.dot(self.w) + self.b).reshape(-1,1) ###to change the shape of (nu_train,) to (nu_train,1)
        error = predicted - dep_vec  ### size of error term is (nu_train,1)      
        rmse = (np.sum(error**2)/self.nu_train)**(1/2) ### calculate RMSE
        r_square = 1 - np.sum(error**2)/np.sum((dep_vec-np.mean(dep_vec))**2) ###calculate R-square
        return(rmse,r_square)

### neutral network with stiachastic gredient descent in scikilt
def error_term(predicted,truevalue):
    error = predicted - truevalue ### size of error term is (nu_train,1)      
    rmse = np.sqrt(np.sum(error**2)/len(error)) ### calculate RMSE
    return(rmse)

### this is for find most suitable learn rate under a speicific hidden layer structure
### and lr is from 0.1 to 1 for each step 0.1
def neural_changing_lr(train_x,train_y,test_x,test_y,hidden_layer):
    rmse_ma = []
    for i in range(1,11,2):
        lr = i/10
        model = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='logistic', solver='sgd', max_iter=1000, learning_rate_init=lr,random_state=2023)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        rmse_ma.append(error_term(y_pred,test_y))
    rmse_label = np.where(np.argsort(rmse_ma)==0)[0][0]
    rmse = rmse_ma[rmse_label]
    return(rmse,rmse_label)

### this is for finding the most suitable hidden layer with max layer number 2
def find_suitable(train_x,train_y,test_x,test_y):
    rmse_heat = np.zeros((6,6))
    lr_he = np.zeros((6,6))
    for i in range(0,6):
        for j in range(0,6):
            m = 1+4*i
            k = 4*j
            if k == 0:
                middle = neural_changing_lr(train_x,train_y,test_x,test_y,(m))
                rmse_heat[i,j] = middle[0]
                lr_he[i,j] = middle[1]
            else:
                middle = neural_changing_lr(train_x,train_y,test_x,test_y,(m,k))
                rmse_heat[i,j] = middle[0]
                lr_he[i,j] = middle[1]
    place = np.where(rmse_heat == rmse_heat.min())
    x = place[0][0]*4+1
    y = place[1][0]*4
    lr = lr_he[place]
    print(lr)
    print(f'The minimun Rmse existing when layer structure is ({x},{y})and learn rate is {lr}.')
    plt.figure(figsize=(6,6))
    heat_map = sns.heatmap( rmse_heat, linewidth = 1 , annot = True)
    heat_map.set_xticklabels(['0','4','8','12','16','20'])
    heat_map.set_yticklabels(['1','5','9','13','17','21'])
    plt.title( "Heatmap for rmse for different combination of neurons" )
    plt.savefig('9_rmse_heatmap.png')
    plt.clf()


###########  Question 1 ##############
abalone = data_clean(0).astype('float')


print('The cleaned data is showing below')
print(abalone)

###########  Question 2 ##############
heatmap(abalone,1)

###########  Question 3 & 4 ##############
scatter_plot(abalone)

########## Question 5 #############
(trainset,testset) = sepa_dataset(abalone,0)
(normal_train,normal_test) = sepa_dataset(abalone,1)

######### Question 6 ##############
q6_rmse = np.array([0 for i in range(30)]).astype('float')
q6_rsquare = np.array([0 for i in range(30)]).astype('float')
q6_rmse_test = np.array([0 for i in range(30)]).astype('float')
q6_rsquare_test = np.array([0 for i in range(30)]).astype('float')
for i in range(30):
    q6 = linear_regression(trainset,testset,8,0.1,2000)
    a = q6.run()
    b = q6.test_fun()
    
    q6_rmse[i] = a[0]
    q6_rsquare[i] = a[1]
    q6_rmse_test[i] = b[0]
    q6_rsquare_test[i] = b[1]
    
    print('*',end='',flush=True)
print('q6')
print('Full features linear model')
print('Mean of Rmse in train: ', q6_rmse.mean())
print('Standard deviation of Rmse in train: ', q6_rmse.std())
print('Mean of Rsquared in train: ', q6_rsquare.mean())
print('Standard deviation of Rsquared in train: ', q6_rsquare.std())

print('Mean of Rmse in test: ', q6_rmse_test.mean())
print('Standard deviation of Rmse in test: ', q6_rmse_test.std())
print('Mean of Rsquared in test: ', q6_rsquare_test.mean())
print('Standard deviation in test: ', q6_rsquare_test.std())


q6.plot_r(6)
q6.plot_y_y(6)
######### Question 7 ##############

q7_rmse = np.array([i for i in range(30)]).astype('float')
q7_rsquare = np.array([i for i in range(30)]).astype('float')
q7_rmse_test = np.array([i for i in range(30)]).astype('float')
q7_rsquare_test = np.array([i for i in range(30)]).astype('float')
for i in range(30):
    q7 = linear_regression(normal_train,normal_test,8,0.1,2000)
    a = q7.run()
    b = q7.test_fun()
    q7_rmse[i] = a[0]
    q7_rsquare[i] = a[1]
    q7_rmse_test[i] = b[0]
    q7_rsquare_test[i] = b[1]
    print('*',end='',flush=True)
print('q7')
print('Full normalizing features linear model')
print('Mean of Rmse for in train: ', q7_rmse.mean())
print('Standard deviation of Rmse in train: ', q7_rmse.std())
print('Mean of Rsquared in train: ', q7_rsquare.mean())
print('Standard deviation of Rsquaredin train: ', q7_rsquare.std())

print('Mean of Rmse in test: ', q7_rmse_test.mean())
print('Standard deviation of Rmse in test: ', q7_rmse_test.std())
print('Mean of Rsquared in test: ', q7_rsquare_test.mean())
print('Standard deviation of Rsquared in test: ', q7_rsquare_test.std())
q7.plot_r(7)
q7.plot_y_y(7)

######### Question 8 ##############
two_cor = find_corr(abalone)
two_train = trainset.take([two_cor[0],two_cor[1],-1], axis=1)
two_test = testset.take([two_cor[0],two_cor[1],-1],axis=1)

q8_rmse = np.array([i for i in range(30)]).astype('float')
q8_rsquare = np.array([i for i in range(30)]).astype('float')
q8_rmse_test = np.array([i for i in range(30)]).astype('float')
q8_rsquare_test = np.array([i for i in range(30)]).astype('float')

for i in range(30):
    q8 = linear_regression(two_train,two_test,2,0.1,2000)
    a = q8.run()
    b = q8.test_fun()
    q8_rmse[i] = a[0]
    q8_rsquare[i] = a[1]
    q8_rmse_test[i] = b[0]
    q8_rsquare_test[i] = b[1]
    print('*',end='',flush=True)
print('q8')

print('Two most correlated features linear model')
print('Mean of Rmse in train: ', q8_rmse.mean())
print('Standard deviation of Rmse in train: ', q8_rmse.std())
print('Mean of Rsquared in train: ', q8_rsquare.mean())
print('Standard deviation of Rsquared in train: ', q8_rsquare.std())

print('Mean of Rmse in test: ', q8_rmse_test.mean())
print('Standard deviation of Rmse in test: ', q8_rmse_test.std())
print('Mean of Rsquared in test: ', q8_rsquare_test.mean())
print('Standard deviation of Rsquared in test: ', q8_rsquare_test.std())


q8.plot_r(8)
q8.plot_y_y(8)

######### Question 9 ##############
train_x = trainset[:,:8]
train_y = trainset[:,-1]
test_x = testset[:,:8]
test_y = testset[:,-1]


### find_suitable(train_x,train_y,test_x,test_y)

test_rmse =np.zeros(30)
train_rmse =np.zeros(30)
test_acc =np.zeros(30)
train_acc=np.zeros(30)
for i in range(30):
    model = MLPClassifier(hidden_layer_sizes=(13,8), activation='logistic', solver='sgd', max_iter=2000, learning_rate_init=0.5)
    model.fit(train_x, train_y)
    y_pred_1 = model.predict(test_x)
    test_rmse[i] = (error_term(y_pred_1,test_y))
    y_pred_2 = model.predict(train_x)
    train_rmse[i] = (error_term(y_pred_2,train_y))
    acc_1 = sum(y_pred_1==test_y)/test_x.shape[0]
    acc_2 = sum(y_pred_2==train_y)/train_x.shape[0]
    test_acc[i] = (acc_1)
    train_acc[i] = (acc_2)
    print('*',end='',flush=True)
print('q9')

print('Mean of Rmse in test: ', test_rmse.mean())
print('Standard deviation of Rmse in test: ', test_rmse.std())
print('Mean of Rmse in train: ', train_rmse.mean())
print('Standard deviation of Rmse in train: ', train_rmse.std())
print('Mean of accuracy in test: ', test_acc.mean())
print('Mean of accuracy in train: ', train_acc.mean())

######### Question 10 #############

abalone = data_clean(1).astype('float')
heatmap(abalone,10)
