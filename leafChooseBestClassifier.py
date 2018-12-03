import sys
import os
import numpy as np
import sklearn.neighbors as neighbors
import sklearn.feature_selection as feature_selection
import sklearn.model_selection as model_selection
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.neural_network as neural
import sklearn.externals as externals
import warnings
warnings.filterwarnings('ignore')

def read_from_file(input_):
	data = np.genfromtxt(input_, delimiter='\t',skip_header = 1 , usecols = range(1,20))
	dic,i,target = ({},0,[])
	target = np.concatenate(([0]*40,[1]*40,[2]*40,[3]*40,[4]*40,[5]*40,[6]*40,[7]*40,[8]*40,[9]*40,[10]*40,[11]*40,[12]*40,[13]*40,[14]*40,[15]*40,[16]*40,[17]*40,[18]*40,[19]*40))
	return data, target

def best_classifier(data, target, n_columns, classifier, classifier_parameters, score_func,file_name):
	best_data = feature_selection.SelectKBest(score_func=score_func, k=n_columns).fit_transform(data, target)
	clf = model_selection.GridSearchCV(classifier, classifier_parameters, cv=5)
	clf.fit(best_data, target)
	with open(file_name, 'a') as myfile:
		myfile.write('-----------------'+str(n_columns)+'-----------------'+'\n'+str(clf.best_estimator_)+'\n'+str(clf.best_params_)+'\n'+str(clf.best_score_)+'\n\n')

if __name__ == "__main__":
	if len(sys.argv)<2:
		sys.argv.append('output.csv')
	data, target = read_from_file(sys.argv[1])
	#KNeighborsClassifier
	knn = neighbors.KNeighborsClassifier()
	knn_parameters = {'algorithm': ['auto','ball_tree','kd_tree','brute'],'n_neighbors': list(range(1,26)), 'p':[1,2,3,4],'leaf_size':[10,20,30,40,50],'weights':['uniform','distance']}
	#RandomForestClassifier
	rfc = ensemble.RandomForestClassifier()
	rfc_parameters = [{'n_estimators': list(range(1,32,2)), 'max_features':['sqrt','log2',None], 'min_samples_split': list(range(2,6)), 'min_samples_leaf': list(range(1,7)), 'bootstrap': [True,False],'warm_start': [True,False]},
			{'n_estimators': list(range(1, 32, 2)), 'max_features': ['sqrt', 'log2', None],'min_samples_split': list(range(2, 6)), 'min_samples_leaf': list(range(1, 7)), 'bootstrap': [True], 'oob_score': [True, False], 'warm_start': [True, False]}]
	#DecisionTreeClassifier
	dtc= tree.DecisionTreeClassifier()
	dtc_parameters = {'criterion': ['gini','entropy'], 'splitter': ['best','random'],'max_features':['sqrt','log2',None],'min_samples_split': list(range(2,6)), 'min_samples_leaf': list(range(1,7)), 'presort':[True,False]}
	#NeuralNetwork
	nn = neural.MLPClassifier()
	nn_parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'],'solver': ['lbfgs', 'sgd', 'adam'],'learning_rate': ['constant', 'invscaling', 'adaptive'], 'shuffle':[True,False],'early_stopping':[True,False]}
	for n_columns in range(1,20):
		best_classifier(data,target,n_columns,knn,knn_parameters,feature_selection.chi2,'knn_chi2.txt')
		best_classifier(data,target,n_columns,knn,knn_parameters,feature_selection.f_classif,'knn_f-classif.txt')
		best_classifier(data,target,n_columns,rfc,rfc_parameters,feature_selection.chi2,'rfc_chi2.txt')
		best_classifier(data,target,n_columns,rfc,rfc_parameters,feature_selection.f_classif,'rfc_f-classif.txt')
		best_classifier(data,target,n_columns,dtc,dtc_parameters,feature_selection.chi2,'dtc_chi2.txt')
		best_classifier(data,target,n_columns,dtc,dtc_parameters,feature_selection.f_classif,'dtc_f-classif.txt')
		best_classifier(data,target,n_columns,nn,nn_parameters,feature_selection.chi2,'nn_chi2.txt')
		best_classifier(data,target,n_columns,nn,nn_parameters,feature_selection.f_classif,'nn_f-classif.txt')
	#zapis do pliku najlepszego klasyfikatora
	best_data = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=19).fit_transform(data, target)
	parameters = {'n_estimators': [27], 'max_features':['log2'], 'min_samples_split': [2], 'min_samples_leaf': [1], 'bootstrap': [False],'warm_start': [False]},
	clf = model_selection.GridSearchCV(rfc, parameters, cv=5)
	clf.fit(best_data, target)
	externals.joblib.dump(clf, 'best_classifier.pkl')
