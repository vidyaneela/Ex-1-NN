<H3>ENTER YOUR NAME : M VIdya Neela</H3>  
<H3>ENTER YOUR REGISTER NO : 212221230120</H3> 
<H3>EX. NO.1</H3>
<H3>DATE : 29-02-1024</H3> 
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

*Kaggle :*
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

*Data Preprocessing:*

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

*Need of Data Preprocessing :*

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('data.csv')
print(df)
df.head()
X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.duplicated()

print(df['CustomerId'].describe())
print(df['Surname'].describe())
print(df['CreditScore'].describe())
print(df['Geography'].describe())
print(df['Gender'].describe())
print(df['Age'].describe())
print(df['Tenure'].describe())
print(df['Balance'].describe())
print(df['NumOfProducts'].describe())
print(df['HasCrCard'].describe())
print(df['IsActiveMember'].describe())
print(df['EstimatedSalary'].describe())
print(df['Exited'].describe())

data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))



## OUTPUT:
### Initial DataFrame : 
![op](./nn1.png)
![op](./nn2.png)
### X values : 
![op](./nn3.png)
### y values :
![op](./nn4.png)
### Null count :
![op](./nn5.png)
### Duplicate :
![op](./nn6.png)
### Description of the Column "CustomerId" : 
![op](./nn7.png)
### Description of the Column "Surname" : 
![op](./nn8.png)
### Description of the Column "CreditScore" : 
![op](./nn9.png)
### Description of the Column "Geography" : 
![op](./nn10.png)
### Description of the Column "Gender" : 
![op](./nn11.png)
### Description of the Column "Age" : 
![op](./nn12.png)
### Description of the Column "Tenure" : 
![op](./nn13.png)
### Description of the Column "Balance" : 
![op](./nn14.png)
### Description of the Column "NumOfProducts" : 
![op](./nn15.png)
### Description of the Column "HasCrCard" : 
![op](./nn16.png)
### Description of the Column "IsActiveMember" : 
![op](./nn17.png)
### Description of the Column "EstimatedSalary" : 
![op](./nn18.png)
### Description of the Column "Exited" : 
![op](./nn19.png)
### Dropping the non numerical columns :
![op](./nn20.png)
### Data after applying Min Max Scaling: 
![op](./nn21.png)
### Values of X_train and X_test :
![op](./n2n22.png)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
