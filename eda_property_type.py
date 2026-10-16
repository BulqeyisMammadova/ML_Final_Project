import matplotlib.pyplot as plt
from src.data_prep import load_raw, clean, get_splits

df = clean(load_raw())
_, y_train, *_ = get_splits()

df_train = df.loc[y_train.index]
median_by_type = df_train.groupby("Kateqoriya")["price"].median().sort_values()

plt.barh(median_by_type.index, median_by_type.values)
plt.xlabel("Median Sale Price (AZN)")
plt.ylabel("Property Type")
plt.title("Median Sale Price by Property Type")
plt.tight_layout()
plt.show()