import pandas as pd
from src.data_prep import train_val_test_split

X=pd.DataFrame(
    {"area_m2":list(range(20))}
)

y=pd.Series(
    list(range(20))
)

netice=train_val_test_split(X,y)
netice_tekrar=train_val_test_split(X, y)
print(netice_tekrar[0].equals(netice[0]))
index_0=netice[0].index.intersection(netice[2].index)
index_00=netice[0].index.intersection(netice[4].index)
index_2=netice[2].index.intersection(netice[4].index)

print(index_0)
print(index_00)
print(index_2)

print(netice[0])
print(netice[1])
print(len(netice[2]))
print(len(netice[4]))