import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from IPython.display import display

dataset = pd.read_csv('bank.csv', delimiter=';')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2,3,4,6,7,8,10,15])], remainder='passthrough')
X=np.array(ct.fit_transform(X))

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

pred_table = np.zeros((2,2))

t = len(y_pred)
u = 1/t
for i in range(0,t):
	pred_table[int(y_pred[i]>0.5),int(y_test[i]>0.5)] += u

df = pd.DataFrame(pred_table, columns=['Resposta Esperada \'não\'','Resposta Esperada \'sim\''], index=['Resposta Predita \'não\'','Resposta Predita \'sim\''])
display(df)
