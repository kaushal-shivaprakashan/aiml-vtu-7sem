from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
iris=datasets.load_iris() 
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
print("Results of Classification using K-nn") 
for r in range(0,len(x_test)):
    print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r])," Predicted-label:", str(y_pred[r]))

    print("Classification Accuracy :" , classifier.score(x_test,y_test));
    
    
    
    ########################################################################
    Results of Classification using K-nn
 Sample: [5.9 3.2 4.8 1.8]  Actual-label: 1  Predicted-label: 2
Classification Accuracy : 0.9333333333333333
 Sample: [5.8 2.7 5.1 1.9]  Actual-label: 2  Predicted-label: 2
Classification Accuracy : 0.9333333333333333
 Sample: [5.4 3.  4.5 1.5]  Actual-label: 1  Predicted-label: 1
Classification Accuracy : 0.9333333333333333
 Sample: [6.  2.2 4.  1. ]  Actual-label: 1  Predicted-label: 1
Classification Accuracy : 0.9333333333333333
 Sample: [4.3 3.  1.1 0.1]  Actual-label: 0  Predicted-label: 0
Classification Accuracy : 0.9333333333333333
 Sample: [4.8 3.4 1.6 0.2]  Actual-label: 0  Predicted-label: 0
Classification Accuracy : 0.9333333333333333
 Sample: [5.7 2.8 4.5 1.3]  Actual-label: 1  Predicted-label: 1
Classification Accuracy : 0.9333333333333333
 Sample: [4.9 2.4 3.3 1. ]  Actual-label: 1  Predicted-label: 1
Classification Accuracy : 0.9333333333333333
 Sample: [6.1 3.  4.9 1.8]  Actual-label: 2  Predicted-label: 2
Classification Accuracy : 0.9333333333333333
 Sample: [6.4 3.2 5.3 2.3]  Actual-label: 2  Predicted-label: 2
Classification Accuracy : 0.9333333333333333
 Sample: [5.6 3.  4.5 1.5]  Actual-label: 1  Predicted-label: 1
Classification Accuracy : 0.9333333333333333
 Sample: [5.7 4.4 1.5 0.4]  Actual-label: 0  Predicted-label: 0
Classification Accuracy : 0.9333333333333333
 Sample: [4.9 3.  1.4 0.2]  Actual-label: 0  Predicted-label: 0
Classification Accuracy : 0.9333333333333333
 Sample: [7.7 3.8 6.7 2.2]  Actual-label: 2  Predicted-label: 2
Classification Accuracy : 0.9333333333333333
 Sample: [5.  3.2 1.2 0.2]  Actual-label: 0  Predicted-label: 0
Classification Accuracy : 0.9333333333333333
