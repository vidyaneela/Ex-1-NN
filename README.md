)<H3>ENTER YOUR NAME : M Vidya Neela</H3>  
<H3>ENTER YOUR REGISTER NO : 212221230120</H3> 
<H3>EX. NO.1</H3>

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
```
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
```


## OUTPUT:
### Initial DataFrame : 
![nn1](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/ca1b57e3-f44b-4f28-9918-8f8b5c4c9319)
![nn2](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/3c727a18-a2b8-4265-8cdf-3a0b171b960e)


### X values : 
![nn3](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/c95503f8-c7bc-4ae5-b39f-44b86f3e8356)

### y values :
![nn4](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/b0f15bf3-1377-4ccd-ba62-a9a06764d558)

### Null count :
![nn5](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/0eebd1af-ea1a-4ac1-bfdb-cbd688e96fdc)

### Duplicate :
![nn6](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/6e983ae3-c56a-485a-abde-67f2a0664853)

### Description of the Column "CustomerId" : 
![nn7](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/3f07a664-6796-4dc8-83a6-7d6cc02e58be)

### Description of the Column "Surname" : 
![nn8](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/b9fa82ce-8ae4-4efb-a660-2e53cf2b8d75)

### Description of the Column "CreditScore" : 
![nn9](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/0f0f44cc-365d-4cf6-9b7c-93148713abf7)

### Description of the Column "Geography" : 
![nn10](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/063bbc55-baf0-49ed-b0a7-8647dcd14ef1)

### Description of the Column "Gender" : 
![nn11](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/5661b643-ccb7-4c94-a5aa-d27e9e81d497)


### Description of the Column "Age" : 
![nn12](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/9745ba09-567b-4e9f-a184-1e69b9dee580)

### Description of the Column "Tenure" : 
![nn13](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/150cef86-3a7c-429f-a3b6-f69af8aa9172)

### Description of the Column "Balance" : 
![nn14](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/a984d0b6-8eed-4fba-9234-9dcbf0d027d7)

### Description of the Column "NumOfProducts" : 
![nn15](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/37e456ca-9248-4c49-a6f7-963b5dfd3059)

### Description of the Column "HasCrCard" : 
![nn16](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/0a058378-00af-4ec4-83cb-7debd802f398)

### Description of the Column "IsActiveMember" : 
![nn17](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/dda3193e-380c-4efc-8520-1f19871136c1)

### Description of the Column "EstimatedSalary" : 
![nn18](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/4bde5b63-4c97-4925-9447-bd68dc0f988e)

### Description of the Column "Exited" : 
![nn19](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/e0724338-304f-48cf-8cdf-139b63a8cd60)

### Dropping the non numerical columns :
![nn20](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/f0448d6d-1047-41de-bd84-052ca2be343f)

### Data after applying Min Max Scaling: 
![nn21](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/34245c6c-587d-4565-bc6c-dadf5da5576b)

### Values of X_train and X_test :

![n2n22](https://github.com/Haridharshini21500176/Ex-1-NN/assets/94168395/d68f53cb-6f08-4904-8e4c-8c46a81af39a)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
