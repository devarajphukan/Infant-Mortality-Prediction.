import pandas as pd 
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import *

train_data_df = pd.read_csv('train.csv',delimiter=',',header = 0)
test_data_df = pd.read_csv('test.csv',header = 0 ,delimiter=",")

col_names = train_data_df.columns[:-1].tolist()

for i in col_names :
	train_data_df[i] =  pd.Series(train_data_df[i],dtype = "category")
	test_data_df[i] = pd.Series(test_data_df[i],dtype = "category")

train_data_df['Mothers Single Year of Age'] = pd.Series(train_data_df['Mothers Single Year of Age'],dtype = 'float64')
train_data_df['Cigarettes Before Pregnancy '] = pd.Series(train_data_df['Cigarettes Before Pregnancy '],dtype = 'float64')
train_data_df['Cigarettes 1st trimester'] = pd.Series(train_data_df['Cigarettes 1st trimester'],dtype = 'float64')
train_data_df['Cigarettes 2nd trimester'] = pd.Series(train_data_df['Cigarettes 2nd trimester'],dtype = 'float64')
train_data_df['Cigarettes 3rd trimester'] = pd.Series(train_data_df['Cigarettes 3rd trimester'],dtype = 'float64')
train_data_df['Number of previous Cesarean Deliveries'] = pd.Series(train_data_df['Number of previous Cesarean Deliveries'],dtype = 'float64')
train_data_df["Mother's Height in Inches"] = pd.Series(train_data_df["Mother's Height in Inches"],dtype = 'float64')

test_data_df['Mothers Single Year of Age'] = pd.Series(test_data_df['Mothers Single Year of Age'],dtype = 'float64')
test_data_df['Cigarettes Before Pregnancy '] = pd.Series(test_data_df['Cigarettes Before Pregnancy '],dtype = 'float64')
test_data_df['Cigarettes 1st trimester'] = pd.Series(test_data_df['Cigarettes 1st trimester'],dtype = 'float64')
test_data_df['Cigarettes 2nd trimester'] = pd.Series(test_data_df['Cigarettes 2nd trimester'],dtype = 'float64')
test_data_df['Cigarettes 3rd trimester'] = pd.Series(test_data_df['Cigarettes 3rd trimester'],dtype = 'float64')
test_data_df['Number of previous Cesarean Deliveries'] = pd.Series(test_data_df['Number of previous Cesarean Deliveries'],dtype = 'float64')
test_data_df["Mother's Height in Inches"] = pd.Series(test_data_df["Mother's Height in Inches"],dtype = 'float64')

labels_numeric = pd.Series(train_data_df['Infant Living at Time of Report'],dtype = "category")
 
train_data_df = train_data_df.drop('Infant Living at Time of Report',1)

train_data_df = np.array(train_data_df)
test_data_df = np.array(test_data_df)

# X = train_data_df
# pca = PCA(n_components=29).fit(X)
# print(pca.explained_variance_ratio_)
# data2D = pca.transform(X)
# print len(data2D[:,0])
# print len(data2D[:,1])
# plt.scatter(data2D[:,0], data2D[:,1], c=np.array(labels_numeric))
# plt.show()

# my_model = MultinomialNB(alpha = 0.8 ,fit_prior = False)
# my_model = LinearSVC(penalty = 'l2',dual = True, C = 0.7, loss = 'hinge')
my_model = LogisticRegression(penalty = 'l1',C = 0.6)


my_model = my_model.fit(X=train_data_df, y=labels_numeric)
test_pred = my_model.predict(test_data_df)
test_pred =  list(test_pred)
op = []
f = open('results.csv')
for i in f :
	op.append(int(i.strip()))

print(classification_report(op,test_pred))
print(accuracy_score(op,test_pred))