from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import jieba
import csv
import pandas as pd
from pandas import read_csv
#from pandas.plotting import scatter_matrix
from matplotlib import pyplot
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path_train="train.csv"
path_test="test.csv"
Train_df = pd.read_csv(path_train)
Test_df=pd.read_csv(path_test)
Data_labels=Train_df['title1_zh']+Train_df['title2_zh']
text_list=[]
for x in Data_labels:
    text_list.append(' '.join(jieba.cut(x, HMM=False)))
#print(text_list)
output=Train_df['label']
#print(output.values)0

Y_test=Test_df['title1_zh']+Test_df['title2_zh']
Y_list=[]
for x in Y_test:
    Y_list.append(' '.join(str(jieba.cut(x, HMM=False))))
#print(Y_list)


#string to vector for machine learning transform
count_vect = CountVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=None)
X_train_counts = count_vect.fit_transform(text_list)
TF_train_object=TfidfTransformer()
X_tf_train=TF_train_object.fit_transform(X_train_counts)
#print(X_tf_train)

#clf=MultinomialNB().fit(X_tf_train,output.values.ravel())
clf=LogisticRegression().fit(X_tf_train,output.values.ravel())
#clf=DecisionTreeClassifier().fit(X_tf_train,output.values.ravel())
#clf=KNeighborsClassifier().fit(X_tf_train,output.values.ravel())

Y_count_vector=count_vect.transform(Y_list)
#print (Y_test)
Y_predict=TF_train_object.transform(Y_count_vector)

prediction_result=clf.predict(Y_predict)

Y_Id=Test_df['id']

with open('result_KN.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Y_Id, prediction_result))
