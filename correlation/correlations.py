import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt

data = pd.read_csv("pre.csv",delimiter = ",", header = 0)
col_names = data.columns[:-1].tolist()
data_array = np.array(data)

corr_li = []
pval_li = []
for col in range(0,29):

	li = []
	for row in range(0,40892) :
	# for row in range(0,9):
		li.append(data_array[row][col])

	if (col == 1 or col == 4 or col == 5 or col == 6 or col == 7 or col == 16 or col == 28 ) :
		
		indep = pd.Series(li,dtype = "float")
	
	else :
		indep = pd.Series(li,dtype = "category")

	dep = pd.Series(data['Infant Living at Time of Report'],dtype = "category")

	
	corr_li.append(kendalltau(indep.tolist(),dep.tolist()).correlation)
	pval_li.append(kendalltau(indep.tolist(),dep.tolist()).pvalue)

fw = open('correlation.txt','w')
for i in range(len(col_names)) :
	fw.write(col_names[i] + "\t" + str(corr_li[i]) + "\t" + str(pval_li[i]) + "\n")
