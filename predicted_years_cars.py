from math import floor
import pandas as pd
import random
import matplotlib.pyplot as plt


auto_mpg = pd.read_csv("C:/Users/Oluwapelumi/Desktop/auto-mpg2.csv")
auto_mpg.describe()
auto_mpg.dropna

Price =[]

for n in range(234):
     Price.append(random.randint(10000,30000))
Price    



auto_mpg_features=["hwy","cty","cyl"]
X= auto_mpg[auto_mpg_features]

print(X.head())
print(X.describe())

from sklearn.tree import DecisionTreeRegressor

auto_mpg_model = DecisionTreeRegressor(random_state=1)
auto_mpg_model.fit(X,Price)


print(auto_mpg_model.predict(X.head()))

from sklearn.metrics import mean_absolute_error

predicted_car_prices = auto_mpg_model.predict(X)
print(mean_absolute_error(Price, predicted_car_prices))

from sklearn.model_selection import train_test_split

val_X, train_X, val_Price, train_Price = train_test_split(X,Price,random_state=1)
auto_mpg_model = DecisionTreeRegressor()
auto_mpg_model.fit(train_X,train_Price)

predicted_val= auto_mpg_model.predict(val_X)
print(mean_absolute_error(predicted_val,val_Price))

def get_mae(max_leaf_nodes, train_X, val_X, train_Price, val_Price):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_Price)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_Price, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_Price, val_Price)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_Price)
auto_mpg_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_Price, auto_mpg_preds))

plt.scatter(train_Price,train_Price)
plt.show()
