import matplotlib.pyplot as plt
from src.data_prep import get_splits

X_train, y_train, *_ = get_splits()

plt.scatter(X_train["area_m2"], y_train, alpha=0.2, s=5)

plt.xlabel("Area (m²)")
plt.ylabel("Sale Price (AZN)")
plt.title("Sale Price vs Area")
plt.xlim(0,X_train["area_m2"].quantile(0.99))
plt.ylim(0,y_train.quantile(0.99))
print("Pearson Correlation:", X_train["area_m2"].corr(y_train))
mask = (X_train["area_m2"] <= X_train["area_m2"].quantile(0.99)) & (y_train <= y_train.quantile(0.99))
print("Filtered Pearson:", X_train.loc[mask, "area_m2"].corr(y_train.loc[mask]))
is_new = X_train["property_type_Yeni tikili"] == 1

print("Yeni tikili Pearson:",
      X_train.loc[is_new, "area_m2"].corr(y_train.loc[is_new]))

new_area = X_train.loc[is_new, "area_m2"]
new_price = y_train.loc[is_new]
new_mask = (
    (new_area <= new_area.quantile(0.99)) &
    (new_price <= new_price.quantile(0.99))
)
print(
    "Filtered Yeni tikili Pearson:",
    new_area.loc[new_mask].corr(new_price.loc[new_mask])
)
plt.show()