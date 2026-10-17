# Exploratory geospatial bonus

`geospatial_bonus.py` is a separate, descriptive analysis of Baku listings.
It does not change the main tree/SVM pipeline or its held-out test metrics.

The script keeps cleaned Baku rows with coordinates inside its stated bounds,
intersects them with the existing train/validation/test indices, and converts
latitude and longitude to approximate east-west and north-south kilometres
around the training-coordinate median. It fits scikit-learn K-Means with
`k=5` on training coordinates only, assigns validation and test coordinates
to those centres, and prints validation silhouette scores for `k=2..8`.
In the 17 October rerun, `k=2` scored 0.6668 and `k=5` scored 0.5813.
The five-cluster view offers more geographic detail but is exploratory;
it is not the best choice by this silhouette comparison and does not
form a final predictive model. This helper uses the separate
`get_splits()` split with 9,668 validation and 9,668 test rows, rather
than the integrated runner's stratified 9,667/9,669 holdouts.

Using training rows only, it compares price and property-type distributions
across clusters and separately describes house price per square metre.
Outputs are `geospatial_k5_clusters.png`, `geospatial_cluster_prices.png`,
and `geospatial_house_price_m2.png`. The charts describe spatial patterns;
they do not establish a causal location effect or estimate performance on
unseen sale prices. Cluster boundaries depend on coordinate bounds, the
five-cluster choice, and the current listing sample.

Run from the repository root after installing `requirements.txt` and placing
the dataset under `data/`:

```bash
python geospatial_bonus.py
```
