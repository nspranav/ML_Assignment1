
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_curve
import matplotlib.pyplot as plt

#function to convert arrays to Tuples
def listOfTuples(l1, l2, l3, l4): 
    return list(map(lambda x, y, a, b:(x,y,a,b), l1, l2, l3, l4)) 

outlook =['sunny','sunny','rain','rain','rain',
          'rain','overcast','sunny'
          ,'sunny','rain']
#print(len(outlook))
temperature = ['hot','hot','hot','mild','cool',
               'mild','cool','cool','cool','mild']
humidity = ['high','high','high','normal','high','normal','normal'
            ,'high','high','high']
wind = ['weak','strong','weak','strong','weak','strong',
       'strong','weak','strong','strong']
play = ['no','yes','yes','no','no','no',
        'yes','no','yes','yes']

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

outlook_encoded = label_encoder.fit_transform(outlook)

#print(outlook_encoded)

temperature_encoded = label_encoder.fit_transform(temperature)

#print(temperature_encoded)

humidity_encoded = label_encoder.fit_transform(humidity)

#print(humidity_encoded)

wind_encoded = label_encoder.fit_transform(wind)
#print(wind_encoded)

play_encoded = label_encoder.fit_transform(play)
#print(play_encoded)

features = listOfTuples(outlook_encoded, temperature_encoded,humidity_encoded,wind_encoded)
#print(len(wind_encoded))

model = tree.DecisionTreeClassifier()

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(features,play_encoded, test_size = 0.30, random_state = 100)

model.fit(data_train,target_train)

pred = model.predict(data_test)
print(pred)
print("Decision tree accuracy : ",accuracy_score(target_test, pred))

#get the report for precission, recall, f1-score, support
report = classification_report(target_test, pred)
print(report)

#get the true negative, true positive, false negative and false positive
tn,fp ,fn ,tp = confusion_matrix(target_test,pred).ravel();

#print(tn,fp,fn,tp)

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