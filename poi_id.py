
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import matplotlib.pyplot as plt

POI_label = ['poi']

features_list = ['poi','salary', 'expenses', 'total_stock_value', 'bonus', 'from_poi_to_this_person', 'shared_receipt_with_poi'] # You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

total_features = POI_label + financial_features + email_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Number of data points = ', len(data_dict)

#(POI/non-POI)
poi_count = 0
for a in data_dict:
    if data_dict[a]['poi'] == True:
        poi_count += 1
print 'Number of POI = ', poi_count
print 'Number of non-POI = ', len(data_dict) - poi_count


# Number and list of features used
print 'Number of available features = ', len(total_features), 
print 'List of features: ', total_features
print 'Number of features used = ', len(features_list)
print 'List of features used: ', features_list


#Features with missing values
missing_values = {}
for b in total_features:
    missing_values[b] = 0

for c in data_dict:
    for d in data_dict[c]:
        if data_dict[c][d] == 'NaN':
            missing_values[d] += 1
print missing_values

### Task 2: Remove outliers
def outliners_plot(dataset, f1, f2):
    data = featureFormat(dataset, [f1, f2])
    for e in data:
        x = e[0]
        y = e[1]
        plt.scatter(x, y)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()
      

#Identifying outliers
outliners_plot(data_dict, "salary", "bonus")
#Removing outliers
data_dict.pop("TOTAL", 0 )
data_dict.pop("TRAVEL AGENCY IN THE PARK", 0 )
data_dict.pop("LOCKHART EUGENE E", 0 )

outliners_plot(data_dict, "salary", "bonus")


### Store to my_dataset for easy export below.
my_dataset = data_dict

#Creating new feature
def fraction(poi_messages, all_messages):
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)
    return fraction

for f in my_dataset:
    from_poi_to_this_person = my_dataset[f]['from_poi_to_this_person']
    to_messages = my_dataset[f]['to_messages']
    fraction_poi = fraction(from_poi_to_this_person, to_messages)
    my_dataset[f]['fraction_poi'] = fraction_poi
    
new_features_list =  total_features + ['fraction_poi']
new_features_list.remove('email_address')
print new_features_list


data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Univariate feature selection
def getKey(g):
    return g[1]

#Removes all features with variance below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 8)

selector.fit(features, labels)
score = zip(new_features_list[1:], selector.scores_)
sorted_score = sorted(score, key = getKey, reverse = True)
print sorted_score
kBest = POI_label + [(i[0]) for i in sorted_score[0:8]]
print kBest





for ff in data_dict:
    for jj in data_dict[ff]:
        if data_dict[ff][jj] == 'NaN':
            data_dict[ff][jj] = 0


from sklearn import preprocessing
data = featureFormat(my_dataset, kBest, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


data = featureFormat(my_dataset, kBest + \
                     ['fraction_poi'], \
                     sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)


from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def tune_your_classifier(grid_search, features, labels, parameters, toiter = 100):
    accuracy = []
    precision = []
    recall = []
    
    for j in range(toiter):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = j)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        accuracy = accuracy + [accuracy_score(labels_test, predicts)] 
        precision = precision + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
        
    print "accuracy: {}".format(np.mean(accuracy))
    print "precision: {}".format(np.mean(precision))
    print "recall: {}".format(np.mean(recall))

    best_parameters = grid_search.best_estimator_.get_params()
    for parameters_name in parameters.keys():
        print("%s = %r, " % (parameters_name, best_parameters[parameters_name]))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_clf = GaussianNB()
NB_parameters = {}
NB_grid_search = GridSearchCV(estimator = NB_clf, param_grid = NB_parameters)
print("Naive Bayes model evaluation")
tune_your_classifier(NB_grid_search, features, labels, NB_parameters)
print("Naive Bayes model evaluation with new features")
tune_your_classifier(NB_grid_search, new_f_features, new_f_labels, NB_parameters)


#Support Vector Machines
from sklearn import svm
SVM_clf = svm.SVC()
SVM_parameters = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
SVM_grid_search = GridSearchCV(estimator = SVM_clf, param_grid = SVM_parameters)
print("SVM model evaluation")
tune_your_classifier(SVM_grid_search, features, labels, SVM_parameters)


#Decision Tree
from sklearn import tree
DT_clf = tree.DecisionTreeClassifier()
DT_parameters = {'criterion':('gini', 'entropy'),
	'splitter':('best','random')}
DT_grid_search = GridSearchCV(estimator = DT_clf, param_grid = DT_parameters)
print("Decision Tree model evaluation")
tune_your_classifier(DT_grid_search, features, labels, DT_parameters)


clf = NB_clf
features_list = kBest



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

