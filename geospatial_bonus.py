import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from src.data_prep import load_raw, clean

from src.data_prep import get_splits


from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score




df = clean(load_raw())

baku = df[df["city"] == "bakı"].copy()

valid = (
    baku["lat"].between(39.5, 41.0) &
    baku["lng"].between(48.5, 50.5)
)

baku = baku.loc[valid].copy()

print("Baku listings:", len(baku))
print("Coordinates:", baku[["lat", "lng"]].shape)




X_train, y_train, X_val, y_val, X_test, y_test = get_splits()

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Coordinate columns:", X_train[["lat", "lng"]].head())




train_geo = baku.loc[baku.index.intersection(X_train.index), ["lat", "lng"]]
val_geo = baku.loc[baku.index.intersection(X_val.index), ["lat", "lng"]]
test_geo = baku.loc[baku.index.intersection(X_test.index), ["lat", "lng"]]

print("Baku Train:", train_geo.shape)
print("Baku Validation:", val_geo.shape)
print("Baku Test:", test_geo.shape)

print("Total:", len(train_geo) + len(val_geo) + len(test_geo))

print(
    "Coordinates match:",
    train_geo.equals(X_train.loc[train_geo.index, ["lat", "lng"]])
)






R = 6371.0

lat0 = np.deg2rad(train_geo["lat"].median())
lng0 = np.deg2rad(train_geo["lng"].median())

def to_xy_km(coords):
    lat = np.deg2rad(coords["lat"].to_numpy())
    lng = np.deg2rad(coords["lng"].to_numpy())

    x = R * np.cos(lat0) * (lng - lng0)
    y = R * (lat - lat0)

    return np.column_stack((x, y))

X_geo_train = to_xy_km(train_geo)
X_geo_val = to_xy_km(val_geo)
X_geo_test = to_xy_km(test_geo)

print("Train km:", X_geo_train.shape)
print("Validation km:", X_geo_val.shape)
print("Test km:", X_geo_test.shape)
print("First 3 coordinates (km):")
print(X_geo_train[:3])




kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10 
)

train_labels = kmeans.fit_predict(X_geo_train)
val_labels = kmeans.predict(X_geo_val)
test_labels = kmeans.predict(X_geo_test)

print("\nK-Means Results")
print("Train cluster counts:", np.bincount(train_labels))
print("Validation cluster counts:", np.bincount(val_labels))
print("Test cluster counts:", np.bincount(test_labels))

print("Cluster centers (km):")
print(kmeans.cluster_centers_.round(2))

print("Inertia:", round(kmeans.inertia_, 2))








print("\nK Selection Results")

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_geo_train)

    labels_val = model.predict(X_geo_val)

    score = silhouette_score(
        X_geo_val,
        labels_val,
        sample_size=1500,
        random_state=42
    )

    print(
        f"K={k} | "
        f"Inertia={model.inertia_:.2f} | "
        f"Validation Silhouette={score:.4f}"
    )










plt.figure(figsize=(11, 8))

points = plt.scatter(
    X_geo_train[:, 0],
    X_geo_train[:, 1],
    c=train_labels,
    cmap="tab10",
    s=3,
    alpha=0.35,
    rasterized=True
)

centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    marker="X",
    s=220,
    edgecolors="white",
    linewidths=1.5,
    label="Cluster centers"
)

for i, (x, y) in enumerate(centers):
    plt.annotate(
        str(i),
        (x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold"
    )

plt.title("Baku Housing — K-Means Geographic Clusters (K=5)")
plt.xlabel("East-West Distance (km)")
plt.ylabel("North-South Distance (km)")
plt.axis("equal")
plt.colorbar(points, ticks=range(5), label="Cluster ID")
plt.legend()
plt.grid(alpha=0.2)

plt.savefig(
    "geospatial_k5_clusters.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("Cluster visualization saved!")





y_train_baku = y_train.loc[train_geo.index]

cluster_price_summary = pd.DataFrame({
    "cluster": train_labels,
    "price": y_train_baku.to_numpy()
}).groupby("cluster")["price"].agg(
    ["count", "median", "mean"]
).sort_values("median", ascending=False)

print("\nCluster Price Summary")
print(cluster_price_summary.round(2))






fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    cluster_price_summary.index.astype(str),
    cluster_price_summary["median"]
)

ax.bar_label(
    bars,
    labels=[
        f"{v:,.0f}"
        for v in cluster_price_summary["median"]
    ],
    padding=4
)

ax.set_title("Median House Price by Geographic Cluster")
ax.set_xlabel("Geographic Cluster")
ax.set_ylabel("Median Price (AZN)")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("geospatial_cluster_prices.png", dpi=180)
plt.close()

print("Cluster price chart saved!")






cluster_properties = pd.DataFrame({
    "cluster": train_labels,
    "property_type": baku.loc[
        train_geo.index, "property_type"
    ].to_numpy()
})

property_mix = pd.crosstab(
    cluster_properties["cluster"],
    cluster_properties["property_type"],
    normalize="index"
) * 100

print("\nProperty Type Distribution by Cluster (%)")
print(property_mix.round(1).to_string())






houses = baku.loc[
    train_geo.index,
    ["property_type", "area_m2"]
].copy()

houses["cluster"] = train_labels
houses["price"] = y_train_baku.to_numpy()

houses = houses[
    (houses["property_type"] == "Həyət evi/Bağ evi") &
    (houses["area_m2"] > 0) &
    (houses["price"] > 0)
].copy()

houses["price_per_m2"] = (
    houses["price"] / houses["area_m2"]
)

house_summary = houses.groupby("cluster").agg(
    count=("price", "size"),
    median_price=("price", "median"),
    median_area=("area_m2", "median"),
    median_price_m2=("price_per_m2", "median")
)

print("\nHouse Price per m2 by Cluster")
print(house_summary.round(2).to_string())





print("\nHouse Area Units by Cluster")

print(
    pd.crosstab(
        houses["cluster"],
        baku.loc[houses.index, "area_unit"]
    )
)

print("\nOriginal House Area Examples")

print(
    baku.loc[
        houses.index,
        ["Sahə", "Torpaq sahəsi", "area_m2", "land_area_m2"]
    ].head(10).to_string()
)







price_m2_sorted = house_summary.sort_values(
    "median_price_m2",
    ascending=False
)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    price_m2_sorted.index.astype(str),
    price_m2_sorted["median_price_m2"]
)

ax.bar_label(
    bars,
    labels=[
        f"{v:,.0f}"
        for v in price_m2_sorted["median_price_m2"]
    ],
    padding=4
)

ax.set_title("Median House Price per m² by Geographic Cluster")
ax.set_xlabel("Geographic Cluster")
ax.set_ylabel("Median Price per m² (AZN)")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("geospatial_house_price_m2.png", dpi=180)
plt.close()

print("House price per m² chart saved!")