# Data Quality and Outlier Analysis
## 1. Dataset Overview
Dataset: house_sale.csv
Original observations: 100,775
Observations after deduplication: 64,454
Train Set: 45,118 (70%)
Validation Set: 9,668 (15%)
Test Set: 9,668 (15%)
Final feature count: 14
Random seed: 42
## 2. Data Cleaning Strategy
Converted area measurements from sot to m².
Extracted floor numbers and total floors.
Removed duplicate property listings by keeping the latest scraped record.
Removed target leakage columns: unit_price and total_price.
Applied median imputation using Train statistics for remaining missing numeric values.
Filled missing land_area_m2 values with 0 as an unknown-value indicator.
Applied One-Hot Encoding to property types.
Used the same feature columns across Train, Validation and Test.
Generated classification labels using the Train price median as the threshold.
## 3. Data Quality Findings
### 3.1 Suspicious Low Prices
During EDA, several unusually low property prices were identified.
Listings with price below 1,000 AZN:
Train: 6
Validation: 2
Test: 0
Total: 8
Manual inspection of Train observations revealed the following:
700 AZN: Confirmed rental listing.
999 AZN: Confirmed rental listing.
400 AZN: Confirmed rental listing.
250 AZN: Likely rental listing.
127 AZN: Sale listing with a suspicious price.
11 AZN: Price described as negotiable.
1,600 AZN: Suspicious sale price for a 155 m² apartment.
Manual inspection of Validation observations revealed:
800 AZN: Agricultural land listing describing a 25-hectare property.
73 AZN: Suspicious property price and potentially inconsistent property category.
These findings indicate possible target-value errors and rental listings mixed with sale listings.
No automatic removal was performed during the investigation.
### 3.2 Rental Listing Investigation
Descriptions containing the word "kiray": 2,053 Train observations.
Descriptions matching "kirayə verilir" or "kiraye verilir": 180 Train observations.
Among these 180 observations, 3 had prices below 5,000 AZN.
Manual inspection demonstrated that some sale listings mention existing tenants or rental income.
Therefore, keyword matching alone is insufficient to distinguish rental listings from sale listings.
Automatically removing every listing containing rental-related words would incorrectly remove valid sale observations.
### 3.3 Extremely High Prices
Listings priced above 2,500,000 AZN:
Train: 427
Validation: 84
Test: 84
Total: 595
Maximum observed price: 40,000,000 AZN.
One extreme observation represented a land listing with an area of 200,000 sot.
Converted area: 20,000,000 m².
Calculated price per square meter: 2 AZN/m².
Although the values are unusual, their correctness could not be conclusively verified.
High prices do not necessarily indicate data errors.
Luxury properties and large land plots may have legitimately high sale prices.
Therefore, high-priced observations were not automatically removed.
## 4. Price Distribution Analysis
Mean sale price in Train: 330,394 AZN.
Median sale price in Train: 210,000 AZN.
Minimum price: 11 AZN.
Maximum price: 40,000,000 AZN.
First Quartile (Q1): 139,000 AZN.
Third Quartile (Q3): 329,000 AZN.
95th Percentile: 765,000 AZN.
99th Percentile: 2,500,000 AZN.
The price distribution is strongly right-skewed.
The mean is considerably higher than the median because of expensive properties.
A histogram with logarithmically spaced bins was used to visualize the distribution.
A boxplot was also created to examine quartiles and potential outliers.
## 5. IQR Outlier Analysis
Interquartile Range (IQR) = Q3 - Q1.
IQR = 329,000 - 139,000 = 190,000 AZN.
Lower Fence = Q1 - 1.5 × IQR.
Lower Fence = -146,000 AZN.
Upper Fence = Q3 + 1.5 × IQR.
Upper Fence = 614,000 AZN.
According to the standard IQR rule, prices above 614,000 AZN are potential upper outliers.
However, this threshold does not imply that all such observations are incorrect.
The negative lower fence also demonstrates that the standard IQR rule does not detect suspiciously low positive prices such as 11 AZN.
Property categories must be considered before deciding whether an observation is an actual data error.
## 6. Property Type Analysis
Median sale prices by property category:
Obyekt: 560,000 AZN (1,768 listings).
Ofis: 494,999 AZN (73 listings).
Yeni tikili: 240,000 AZN (25,668 listings).
Həyət evi/Bağ evi: 205,000 AZN (6,772 listings).
Köhnə tikili: 145,000 AZN (8,514 listings).
Torpaq: 135,000 AZN (2,282 listings).
Qaraj: 30,000 AZN (41 listings).
Property types have substantially different median sale prices.
This supports including property_type as a categorical model feature.
However, differences in location, area and other characteristics may also contribute to these price differences.
Categories with few observations should be interpreted cautiously.
## 7. Area and Price Correlation Analysis
Pearson correlation between area_m2 and price across all Train observations: 0.2902.
Pearson correlation after filtering observations above the 99th percentile of area and price: 0.3166.
Pearson correlation for Yeni tikili observations: 0.7993.
Filtered Pearson correlation for Yeni tikili observations: 0.8124.
The overall area-price relationship is weakly positive.
Within the Yeni tikili category, the relationship is strongly positive.
These findings suggest that relationships between features and price depend on property type.
Extreme observations also affect the measured correlation.
Filtering was performed for exploratory analysis only, without modifying the original dataset.
## 8. Feature Selection Experiment
Two versions of the regression dataset were compared using Linear Regression.
Version A: 13 features.
Version B: 14 features, including land_area_m2.
Both models used identical Train and Validation observations.
13 Feature MAE: 188,279.20 AZN.
14 Feature MAE: 188,276.51 AZN.
13 Feature RMSE: 675,751.29 AZN.
14 Feature RMSE: 675,746.13 AZN.
The 14-feature version produced slightly lower MAE and RMSE.
However, the improvement was extremely small and may not be practically significant.
An additional analysis was performed for the Həyət evi/Bağ evi category.
Number of house observations in Validation: 1,418.
13 Feature House MAE: 207,340.79 AZN.
14 Feature House MAE: 207,339.28 AZN.
The additional land_area_m2 feature produced only a marginal improvement in this subgroup.
The 14-feature version was retained as the working pipeline because land_area_m2 provides meaningful property information.
## 9. Redundant Feature Detection
The initial 15-feature version included land_area_known.
This feature indicated whether land_area_m2 was originally available.
EDA revealed that land_area_known was identical to property_type_Həyət evi/Bağ evi.
The equality was verified across Train, Validation and Test.
This meant that land_area_known did not provide additional information in the current dataset.
Keeping identical features can create redundant information and linear dependencies.
Therefore, land_area_known was removed from the model feature matrix.
The final working feature count was reduced from 15 to 14.
The land_area_known column remained available in the cleaned DataFrame for exploratory purposes.
## 10. Outlier Impact Experiment
An experiment was conducted to measure the effect of removing extremely expensive properties from model training.
Model: Linear Regression.
Feature count: 14.
Original model: Trained on the complete Train Set.
Filtered model: Trained after removing observations above the Train price 99th percentile.
Both models were evaluated on the same unmodified Validation Set.
### 10.1 Overall Validation Results
Original MAE: 188,276.51 AZN.
Filtered MAE: 170,460.51 AZN.
Original RMSE: 675,746.13 AZN.
Filtered RMSE: 999,537.06 AZN.
Filtering reduced MAE by approximately 9.46%.
However, filtering increased RMSE by approximately 47.92%.
### 10.2 Expensive Property Results
Number of expensive Validation observations: 84.
Original Expensive MAE: 4,620,245.30 AZN.
Filtered Expensive MAE: 5,005,153.48 AZN.
The filtered model performed worse on expensive properties.
Removing expensive observations from training reduced the model's ability to predict high-value properties accurately.
### 10.3 Other Property Results
Number of remaining Validation observations: 9,584.
Original Normal MAE: 149,432.04 AZN.
Filtered Normal MAE: 128,086.33 AZN.
The filtered model achieved approximately 14.3% lower MAE on the remaining observations.
This indicates a trade-off between performance on typical and high-priced properties.
## 11. Final Outlier Handling Decision
The original 14-feature Data Pipeline was retained.
No universal upper-price removal threshold was introduced.
The 99th-percentile filtering experiment improved MAE but substantially worsened RMSE.
High-priced properties were therefore preserved in the working dataset.
Confirmed rental listings and suspicious target values were documented separately.
A reliable general cleaning rule for identifying all rental and incorrectly priced listings was not established.
Consequently, suspicious observations were not automatically deleted from the production data preparation functions.
Train, Validation and Test sets remained unchanged during the outlier impact experiment, except for the temporary filtered Train subset used to fit the experimental model.
Future improvements may include category-specific anomaly detection and more reliable rental-versus-sale classification.
## 12. Final Data Preparation Summary
The dataset was reduced from 100,775 raw observations to 64,454 unique listings.
A reproducible 70/15/15 Train/Validation/Test split was implemented.
Target leakage columns were excluded from model features.
Categorical properties were encoded numerically.
Missing values were handled consistently using Train statistics and a fixed zero placeholder for unknown land area.
A 14-feature design matrix was selected as the working version.
EDA identified substantial price skewness, extreme observations and category-dependent relationships.
Feature comparison showed a small improvement from adding land_area_m2.
Outlier experiments demonstrated that removing extreme prices does not necessarily improve overall regression performance.
The data preparation pipeline is ready for further evaluation using the team's regression, classification and SVM models.