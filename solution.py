import pandas as pd
docs = pd.read_csv('~/Downloads/movie_review_train.csv', header=0, names=['Class', 'text'])
docs_test = pd.read_csv('~/Downloads/movie_review_test.csv', header=0, names=['Class', 'text'])
docs.head()

#df.column_name.value_counts() - gives no. of unique inputs in that columns
docs.Class.value_counts()

neg_res=docs.Class.value_counts()
neg_res
print("Neg resp % is ",(neg_res[1]/float(neg_res[0]+neg_res[1]))*100)

# mapping labels to 1 and 0
docs['label'] = docs.Class.map({'Neg':0, 'Pos':1})
docs_test['label'] = docs_test.Class.map({'Neg':0, 'Pos':1})
docs.head()

X=docs.text
y=docs.label
X_test = docs_test.text
y_test = docs_test.label
print(X.shape)
print(y.shape)

# splitting into test and train

# from sklearn.model_selection  import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# X_train.head()

from sklearn.feature_extraction.text import CountVectorizer

# vectorising the text
vect = CountVectorizer(stop_words='english',min_df=.03,max_df=.8)

vect.fit(X)
vect.vocabulary_
len(vect.get_feature_names())

# transform
X_train_transformed = vect.transform(X)
X_test_transformed =vect.transform(X_test)
X_test_tranformed

# training the NB model and making predictions
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed,y)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba =mnb.predict_proba(X_test_transformed)

# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# note that alpha=1 is used by default for smoothing
mnb

# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)

confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
#[row, column]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)

specificity = TN / float(TN + FP)
print("specificity",specificity)

precision = TP / float(TP + FP)

print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))

print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))

y_pred_class
y_pred_proba

# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)


# area under the curve
print (roc_auc)

# matrix of thresholds, tpr, fpr
pd.DataFrame({'Threshold': thresholds, 
              'TPR': true_positive_rate, 
              'FPR':false_positive_rate
             })
             
# plotting the ROC curve

%matplotlib inline  
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)


## Extra
#count of total words in negative reviews
word_count=mnb.feature_count_.sum(axis=1)
nwc=word_count[0]
#total words in positive reviews which are part of dictionary
pwc=word_count[1]

