#function to convert arrays to Tuples
def listOfTuples(l1, l2, l3, l4, l5, l6, l7): 
    return list(map(lambda x, y, a, b, c, d, e:(x,y,a,b,c,d,e), l1, l2, l3, l4, l5, l6, l7)) 

G1 = [1.0,2.0,1.5,2.2,3.9,5.1,1.8,2.3,4.2,3.6]
G2 = [2.3,3.6,1.5,1.9,2.4,3.6,4.2,1.5,2.4,5.6]
G3 = [5.2,1.8,4.1,9.5,5.3,2.7,3.6,7.2,6.2,1.9]
G4 = [1.2,2.3,1.3,1.5,1.7,2.6,3.5,4.1,2.9,3.2]
G5 = [5.3,1.6,1.2,1.5,1.6,1.7,1.6,7.1,2.5,2.6]
G6 = [2.6,2.1,3.1,4.2,2.5,2.8,3.4,3.1,3.3,5.2]
G7 = [2.3,1.5,1.6,1.4,2.9,3.4,1.3,1.8,2.5,2.7]
Output = ['yes','no','yes','no','yes','yes','no','no','yes','no']

from sklearn import preprocessing

#encode the labels to 1's 0's
label_encoder = preprocessing.LabelEncoder()

output_encoded = label_encoder.fit_transform(Output)

#print(output_encoded)

features = listOfTuples(G1,G2,G3,G4,G5,G6,G7)

#print(features)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_curve
import matplotlib.pyplot as plt

model = KNeighborsClassifier(n_neighbors=3)

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(features,output_encoded, test_size = 0.50, random_state = 10)

model.fit(data_train, target_train)

pred = model.predict(data_test)

print("KNN algorithm accuracy : ",accuracy_score(target_test, pred, normalize = True))

#get the report for precission, recall, f1-score, support
report = classification_report(target_test, pred)
print(report)

#get the true negative, true positive, false negative and false positive
tn,fp ,fn ,tp = confusion_matrix(target_test,pred).ravel();

print(tn,fp,fn,tp)

#sensitivity of the classifier
sensitivity = tp /(tp+fn)
print('Sensitivity = '+ str(sensitivity))

#the specificity of the classifier
specificity = tn / (tn+fp)
print('Specificity = ' + str(specificity))

f1_score(target_test, pred, average='macro') 

y_score = model.fit(data_train, target_train).predict_proba(data_test)

false_pos_rate,true_pos_rate,_ = roc_curve(target_test,pred)

plt.plot(false_pos_rate,true_pos_rate,c = 'green');
plt.title = 'ROC'
plt.plot([0,1],[0,1], color = 'red', linestyle='--')
# plt.xlim([0.0,1.0])
# plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()