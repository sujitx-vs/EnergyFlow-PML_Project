import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle as pk

from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

from xgboost import XGBRegressor



data = pd.read_csv('Data/cleaned_data.csv')

Y = data['Global_active_power']

X = data.drop('Global_active_power',axis=1)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=42)


# Scaling is not required for XGBoost like tree models
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)



# params = {
#     'n_estimators' : [100,800,1000],
#     'max_depth' : [3,5,7,9],
#     'learning_rate' : [0.01,0.01,0.02,0.1,1]
# }

# grid = GridSearchCV(XGBRegressor(),params,cv=5,scoring="neg_mean_squared_error")
# grid.fit(X_train,Y_train)
# print(grid.best_params_)

# we found it the best parameters
# {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 800}

model_1 = XGBRegressor(n_estimators = 800,max_depth =3,learning_rate=0.1)

model_1.fit(X_train,Y_train)

pred1 = model_1.predict(X_train)


mse = mean_squared_error(Y_train,pred1)
rmse = root_mean_squared_error(Y_train,pred1)
r2 = r2_score(Y_train,pred1)


print("Train Score -")
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score :",r2)


pred = model_1.predict(X_test)


mse = mean_squared_error(Y_test,pred)
rmse = root_mean_squared_error(Y_test,pred)
r2 = r2_score(Y_test,pred)


print("Test Score -")
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score :",r2)

# Saving the Model


with open('models/XGBmodel.pkl','wb') as file:
    pk.dump(model_1,file)





