import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('Iris.csv')

x = df.iloc[:, 0:-1]
le = LabelEncoder()

y = le.fit_transform(df.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=11)

# LR = LinearRegression()
RD = RandomForestClassifier(n_estimators=200,random_state=42)

RD.fit(x_train,y_train)

y_pred = RD.predict(x_test)

# print("RMSE: ",root_mean_squared_error(y_test,y_pred))
# print("MSE: ",mean_squared_error(y_test,y_pred))
# print("R2: ",r2_score(y_test,y_pred))

# with open("RD.pkl", 'wb') as f:
#     pickle.dump(RD,f)

# with open("LE.pkl", 'wb') as f:
#     pickle.dump(le,f)

print(x)