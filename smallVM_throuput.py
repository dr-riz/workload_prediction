header=['workload','Q1_c','Q6_c','Q12_c','Q21_c','new_order_c','payment_c','trade_order_c','trade_update_c','Q1_t','Q6_t','Q12_t','Q21_t','new_order_t','payment_t','trade_order_t','trade_update_t']
names=header
index=header
url="./data_tp.csv"
data=pandas.read_csv(url, names=header)

df = data.values
del df['workload']
del df['Q1(tps)']
del df['Q6(tps)']
del df['Q12(tps)']
del df['Q21(tps)']

array = df.values
array[0]
x = array[:,0:8]
x
y = array[:,9]
y

array = data.values
x = array[:,1:8]
y = array[:,13:]

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import gaussian_process
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

num_samples = 10
test_size = 0.33
num_instances = len(x)
seed = 7
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
model = gaussian_process.GaussianProcessRegressor(kernel='rbf')
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("MAE: %.3f (%.3f)") % (results.mean(), results.std())





# import 
import pandas as pd

scatter_matrix(data)
plt.show()


from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import gaussian_process
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import GridSearchCV

kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = gaussian_process.GaussianProcessRegressor(ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)))
metric='neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=metric)
print(metric + ": %.3f (%.3f)") % (results.mean(), results.std())

df = data
df = df.drop(df[df.trade_order_t<=0].index)
array = df.values
X = array[:,1:9]
PM = array[:,14] #payment
TO = array[:,15] #trade order
Y=TO
num_instances = len(X)
seed = 7

models = []
models.append(('LR', LinearRegression()))
models.append(('svm(linear)', svm.SVR(kernel='linear')))
models.append(('svm(rbf)', svm.SVR(kernel='rbf', C=8192.0, gamma=2**-15)))
models.append(('gp(rbf)', gaussian_process.GaussianProcessRegressor()))
models.append(('mlp', MLPRegressor()))

# evaluate each model in turn
results = []
names = []
#scoring='r2'
scoring='neg_mean_absolute_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s:%s %f (%f)" % (name, scoring, cv_results.mean(), cv_results.std())
	print(msg)


C = numpy.array([2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15])
gamma = numpy.array([2**-15, 2**-11, 2**-7, 2**-3, 2**1, 2**5])
param_grid = dict(C=C, gamma=gamma)
model = svm.SVR(kernel='rbf')
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.C)
print(grid.best_estimator_.gamma)
	

# Fit the model on 33%
LR=LinearRegression()
LR.fit(X, Y)	
