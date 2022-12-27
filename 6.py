import pandas as pd


from sklearn.naive_bayes import GaussianNB

# Load Data from CSV
data = pd.read_csv('123.csv')
print("The first 5 Values of data is :\n", data.head())

# obtain train data and train output
X = data.iloc[:, :-1]
print("\nThe First 5 values of the train data is\n", X.head())

y = data.iloc[:, -1]
print("\nThe First 5 values of train output is\n", y.head())



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))




###################################################################################################
The first 5 Values of data is :
    1  1.1  1.2  1.3   5
0  1    1    1    2   5
1  2    1    1    2  10
2  3    2    1    1  10
3  3    3    2    1  10
4  3    3    2    2   5

The First 5 values of the train data is
    1  1.1  1.2  1.3
0  1    1    1    2
1  2    1    1    2
2  3    2    1    1
3  3    3    2    1
4  3    3    2    2

The First 5 values of train output is
 0     5
1    10
2    10
3    10
4     5
Name: 5, dtype: int64
Accuracy is: 0.8
  ###########################################################csv#########################################
  1,1,1,1,5
1,1,1,2,5
2,1,1,2,10
3,2,1,1,10
3,3,2,1,10
3,3,2,2,5
2,3,2,2,10
1,2,1,1,5
1,3,2,1,10
3,2,2,2,10
1,2,2,2,10
2,2,1,2,10
2,1,2,1,10
3,2,1,2,5
1,2,1,2,10
1,2,1,2,5
1,1,1,1,5
1,1,1,2,5
2,1,1,2,10
3,2,1,1,10
3,3,2,1,10
3,3,2,2,5
2,3,2,2,10
1,2,1,1,5
1,3,2,1,10
3,2,2,2,10
1,2,2,2,10
2,2,1,2,10
2,1,2,1,10
3,2,1,2,5
1,2,1,2,10
1,2,1,2,5
1,1,1,1,5
1,1,1,2,5
2,1,1,2,10
3,2,1,1,10
3,3,2,1,10
3,3,2,2,5
2,3,2,2,10
1,2,1,1,5
1,3,2,1,10
3,2,2,2,10
1,2,2,2,10
2,2,1,2,10
2,1,2,1,10
3,2,1,2,5
1,2,1,2,10
1,2,1,2,5
