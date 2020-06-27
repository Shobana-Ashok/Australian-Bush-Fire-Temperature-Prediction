# Australian Bushfire Temperature Prediction

### Summary
The purpose of this repository is to predict the brightness-temperature of bushfire in Australia using Machine Learning algorithms with sklearn. The predicted accuracy with the Random Forest Classifier is 87%.
### Dataset
Fires from Space: Australia 

NASA Satellite Data MODISC6 and VIIRS 375m from 2019-08-01 to 2020-01-11
### Environment
* Jupyter notebooks on Azure
* Programming Language: Python 3.6
### Requirements
* pandas for reading the dataset in the form of *.csv
* sklearn for machine learning
* numpy for performing mathematical operations
* reverse_geocoder for converting raw coordinates to insightful location data
* seaborn for statistical data visualization
### Preprocessing
The dataset consists of 36011 rows and 15 columns. The dataset has has 5 feature columns (latitude, longitude, daynight, frp, confidence) and one target variable (brightness).Since the target variable is known, this classification comes under supervised learning. Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.The target variable 'brightness' is continuous. The classifier only accepts categorical values. So we have to convert the target variable to categorical data.The target variable is classified as Low, Medium & Extreme based on the brightness value and mapped to numerical values i.e, Low-0, High-1, Extreme-2. Also the input variable 'daynight' is mapped to the numerical value.
The input variables are stored as data frames in X and the output variable is stored in y.
```python
X.shape, y.shape
```
```
((36011, 6), (36011,))
```
### Training & Testing the model
After preprocessing, the dataset is splitted into train and test.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,stratify=y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
```
```
((28808, 6), (7203, 6), (28808,), (7203,))
```
80% of the dataset is used for training, whereas 20% of the dataset is used for testing.
### ML Algorithms `sklearn`
* KNeighborsClassifier
* DecisionTreeClassifier
* RandomForestClassifier
### Model Performance
| Model         	| Accuracy % 	| Absolute Mean Error 	|
|---------------	|------------	|---------------------	|
| KNN           	| 82.6       	| 0.192               	|
| Decision Tree 	| 83.6       	| 0.167               	|
| Random Forest 	| 87.3       	| 0.129               	|

The RandomForest Classifier is pretty good compared with other classifiers in terms of performance, with the score of 0.87, as well as good precision and recall score.