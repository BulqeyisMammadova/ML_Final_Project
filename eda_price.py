import matplotlib.pyplot as plt
import numpy as np
from src.data_prep import get_splits

X_train, y_train, *_ = get_splits()

plt.hist(y_train, bins=np.logspace(np.log10(y_train.min()), np.log10(y_train.max()), 51))
plt.xscale("log")
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price (AZN, Log Scale)")
plt.ylabel("Number of Listings")
plt.show()

plt.figure()
plt.boxplot(y_train, vert=False)
plt.xscale("log")
plt.title("Boxplot of Property Sale Prices")
plt.show()