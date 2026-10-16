from src.data_prep import get_splits
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


X_train, y_train, X_val, y_val, X_test, y_test = get_splits()

X13_train = X_train.drop(columns=["land_area_m2"])
X13_val = X_val.drop(columns=["land_area_m2"])

print("13 Features:", X13_train.shape, X13_val.shape)
print("14 Features:", X_train.shape, X_val.shape)

model13 = LinearRegression()
model13.fit(X13_train, y_train)

pred13 = model13.predict(X13_val)

print("13 Feature MAE:", mean_absolute_error(y_val, pred13))
print("13 Feature RMSE:", np.sqrt(mean_squared_error(y_val, pred13)))


model14 = LinearRegression()
model14.fit(X_train, y_train)

pred14 = model14.predict(X_val)

print("14 Feature MAE:", mean_absolute_error(y_val, pred14))
print("14 Feature RMSE:", np.sqrt(mean_squared_error(y_val, pred14)))


house_mask = X_val["property_type_Həyət evi/Bağ evi"] == 1

print("House count:", house_mask.sum())
print("13 Feature House MAE:",
      mean_absolute_error(y_val[house_mask], pred13[house_mask]))
print("14 Feature House MAE:",
      mean_absolute_error(y_val[house_mask], pred14[house_mask]))



keep_train = y_train <= y_train.quantile(0.99)
model14_filtered = LinearRegression()

model14_filtered.fit(
    X_train.loc[keep_train],
    y_train.loc[keep_train]
)
pred14_filtered = model14_filtered.predict(X_val)

print("14 Feature Filtered MAE:",
      mean_absolute_error(y_val, pred14_filtered))

print("14 Feature Filtered RMSE:",
      np.sqrt(mean_squared_error(y_val, pred14_filtered)))



high_val = y_val > y_train.quantile(0.99)

print("Expensive listings:", high_val.sum())

print("Original Expensive MAE:",
      mean_absolute_error(y_val[high_val], pred14[high_val]))

print("Filtered Expensive MAE:",
      mean_absolute_error(y_val[high_val], pred14_filtered[high_val]))


normal_val = ~high_val

print("Original Normal MAE:",
      mean_absolute_error(y_val[normal_val], pred14[normal_val]))

print("Filtered Normal MAE:",
      mean_absolute_error(y_val[normal_val], pred14_filtered[normal_val]))