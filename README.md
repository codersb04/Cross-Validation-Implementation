# Cross-Validation-Implementation
## Task
Build a Machine Learning model to predict the Heart disease using the K-Fold Cross Validation method.</br>
Tool Used: Jupyter Notebook, Python
## K- Fold Cross Validation
In K-Fold Cross Validation the dataset is split into "K" number of folds(Subsets). One chunk of data is used as test data for evaluation and remaining part of data is used for training. Each time, different chunk is used for test data.
#### Benifits of K-Fold Cross Validation
- Better alternative for Train test and split method for small dataset
- Better for Multiclass Classification problems
- More Reliable
- Useful for model selection
## Working
- Import all the necessary libraries
- Import 4 different model for comparing performance: Logistic Regression, Support Vector Machine Classifier, K nearest neighbor Classifier and Random Forest Classifer
- Load the dataset using pandas function read_csv
- Analyse the dataset
- Split the feature and target columns
- Split into Training and test data
- Compare the performance of all the 4 model using train test split methode
  - Accuracy score of LogisticRegression(max_iter=1000) :  78.68852459016394
  - Accuracy score of SVC(kernel='linear') :  77.04918032786885
  - Accuracy score of KNeighborsClassifier() :  65.57377049180327
  - Accuracy score of RandomForestClassifier() :  78.68852459016394
- Implement the "Cross validation" with cv = 5 with X and Y values
- Compare the performance of all the 4 model using the K-Fold Cross Validation method
  - Cross Validation score for  LogisticRegression(max_iter=1000) :  [0.81967213 0.8852459  0.80327869 0.86666667 0.76666667]</br>
    Mean accuracy score for  LogisticRegression(max_iter=1000) :  82.83
  - Cross Validation score for  SVC(kernel='linear') :  [0.81967213 0.8852459  0.80327869 0.86666667 0.76666667]</br>
    Mean accuracy score for  SVC(kernel='linear') :  82.83
  - Cross Validation score for  KNeighborsClassifier() :  [0.81967213 0.8852459  0.80327869 0.86666667 0.76666667]</br>
    Mean accuracy score for  KNeighborsClassifier() :  82.83
  - Cross Validation score for  RandomForestClassifier() :  [0.81967213 0.8852459  0.80327869 0.86666667 0.76666667]</br>
    Mean accuracy score for  RandomForestClassifier() :  82.83
- Inference: All the model performe better when implemented using K-Fold Cross Validation</br></br></br>
Reference: 8.2. Cross Validation - Python implementation | cross_val_score | Cross Validation in Sklearn, Siddhardhan, https://www.youtube.com/watch?v=GVZ_zPxr8D4&list=PLfFghEzKVmjsNtIRwErklMAN8nJmebB0I&index=119
