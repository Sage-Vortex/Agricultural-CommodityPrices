import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
print("Running Imports well")

warnings.filterwarnings('ignore')
## Load the data
kamis_df = pd.read_csv('kamis_maize_prices.csv')
agri_df = pd.read_csv('agribora_maize_prices.csv')
# Selecting all entries with White Maize
kdf = kamis_df[kamis_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()
## Selecting all entries in the agriBORA set with White Maize (all the data)
agri_df = agri_df[agri_df['Commodity_Classification'].str.contains("White_Maize", na=False)].copy()
agri_df.shape

target_counties = {"Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"}

kdf = kdf[kdf['County'].isin(target_counties)]
agri_df = agri_df[agri_df['County'].isin(target_counties)]

kdf['price'] = pd.to_numeric(kdf['Wholesale'], errors='coerce')
agri_df['price'] = pd.to_numeric(agri_df['WholeSale'], errors='coerce')
## Convert Date to datetime object
kdf['Date'] = pd.to_datetime(kdf['Date'])
agri_df['Date'] = pd.to_datetime(agri_df['Date'])
# Weekly alignment (use Monday as start of ISO week)
agri_df["week"] = agri_df["Date"].dt.to_period("D").apply(lambda p: p.start_time)
kdf["week"] = kdf["Date"].dt.to_period("D").apply(lambda p: p.start_time)
## Aggregate to Count-Week wholesale mean
kdf_week = (
    kdf.groupby(['County', 'week'], as_index=False)['price']
    .mean()
    .rename(columns={'price': 'kamis_price'})
)

agri_df_week = (
    agri_df.groupby(['County', 'week'], as_index=False)['price']
    .mean()
    .rename(columns={'price': 'agri_price'})
)

coverage_agr = (agri_df_week.groupby("County")["week"]
                .agg(min_date="min", max_date="max", weeks="nunique")
                .reset_index())
coverage_agr["dataset"] = "agribora"

coverage_kam = (kdf_week.groupby("County")["week"]
                .agg(min_date="min", max_date="max", weeks="nunique")
                .reset_index())
coverage_kam["dataset"] = "kamis"
coverage = pd.concat([coverage_agr, coverage_kam], ignore_index=True)
# --- Create pivot table for visualization ---
pivot_cov = (
    coverage
    .pivot_table(
        index="County",
        columns="dataset",
        values="weeks",
    )
    .fillna(0)
)

# ---   Plot heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot_cov,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Weeks with Data"}
)

plt.title("Data Coverage — Weeks of Actual Data per County for White Maize", fontsize=14, pad=15)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("County", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# Is there an overlap between kamis & agribora data ?
overlap = (
    agri_df_week.merge(kdf_week, on=['County', 'week'], how='inner')
    .sort_values(['County', 'week'])
)

overlap['week'] = pd.to_datetime(overlap['week'])

sns.set(style="whitegrid")
counties = overlap['County'].unique()

for county in counties:
    df = overlap[overlap['County'] == county]

    plt.figure(figsize=(10, 4))

    # Plot agr_price and kamis_price
    plt.plot(df['week'], df['agri_price'], marker='o', label='agri_price')
    plt.plot(df['week'], df['kamis_price'], marker='o', label='kamis_price')

    # Shade the difference
    plt.fill_between(df['week'], df['agri_price'], df['kamis_price'],
                     color='gray', alpha=0.2, label='Difference')

    plt.title(f"Weekly Prices Comparison — {county}", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
if not overlap.empty:
    overlap["diff"] = overlap["agri_price"] - overlap["kamis_price"]
    stats = overlap.groupby("County").agg(
        n=("agri_price", "size"),
        corr=(
        "agri_price", lambda x: np.corrcoef(x, overlap.loc[x.index, "kamis_price"])[0, 1] if len(x) > 1 else np.nan),
        mean_bias=("diff", "mean"),
        mae=("diff", lambda x: np.abs(x).mean())
    ).reset_index().round(10)
else:
    stats = pd.DataFrame(columns=["county_norm", "n", "corr", "mean_bias", "mae"])

    agr = pd.read_csv("agribora_maize_prices.csv", parse_dates=["Date"])
kamis = pd.read_csv("kamis_maize_prices.csv", parse_dates=["Date"])

# Filter to white maize
agr = agri_df.copy()
agr = agr[agr["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()
kamis = kamis[kamis["Commodity_Classification"].str.contains("White_Maize", na=False)].copy()


def norm_county(s):
    return s.strip() if isinstance(s, str) else s


agr["county_norm"] = agr["County"].apply(norm_county)
kamis["county_norm"] = kamis["County"].apply(norm_county)

target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
agr = agr[agr["county_norm"].isin(target_counties)].copy()
kamis = kamis[kamis["county_norm"].isin(target_counties)].copy()

# Weekly aggregation
agr["week_start"] = agr["Date"].dt.to_period("W").apply(lambda p: p.start_time)
kamis["week_start"] = kamis["Date"].dt.to_period("W").apply(lambda p: p.start_time)

agr["agr_price"] = pd.to_numeric(agr["WholeSale"], errors="coerce")
kamis["kamis_price"] = pd.to_numeric(kamis["Wholesale"], errors="coerce")

agr_week = agr.groupby(["county_norm", "week_start"], as_index=False)["agr_price"].median()
kamis_week = kamis.groupby(["county_norm", "week_start"], as_index=False)["kamis_price"].median()
# Kenyan county centroid coordinates (approximate)
# Source: Public GIS datasets / government county shapefiles (preloaded here manually)
data = {
    "county_norm": [
        "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu",
        "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho",
        "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui",
        "Kwale", "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera",
        "Marsabit", "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi",
        "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri",
        "Samburu", "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi",
        "Trans Nzoia", "Turkana", "Uasin-Gishu", "Vihiga", "Wajir", "West Pokot"
    ],
    "latitude": [
        0.469, -0.801, 0.569, 0.434, 1.046, -0.531,
        -0.453, -0.495, 0.352, -2.098, 0.307, -0.377,
        -1.030, -3.510, -0.498, -0.681, -0.091, -1.366,
        -4.175, 0.421, -2.162, -1.517, -2.247, 3.937,
        3.544, 0.355, -1.064, -4.043, -0.783, -1.286,
        -0.303, 0.205, -1.145, -0.566, -0.258, -0.419,
        0.993, -0.133, -3.316, -1.845, -0.283, 1.010,
        3.118, 0.539, 0.023, 1.748, 1.532
    ],
    "longitude": [
        35.990, 35.342, 34.564, 34.124, 35.363, 37.456,
        39.654, 34.639, 38.570, 36.789, 34.751, 35.279,
        36.868, 39.800, 37.318, 34.778, 34.761, 38.015,
        39.458, 36.787, 40.902, 37.263, 37.892, 41.847,
        37.998, 37.655, 34.473, 39.668, 36.605, 36.816,
        36.188, 35.117, 35.860, 34.935, 36.574, 36.947,
        37.537, 34.266, 37.757, 39.507, 37.908, 35.021,
        35.587, 35.283, 34.729, 40.060, 35.162
    ]
}

centroids = pd.DataFrame(data)
centroids.head()
present_counties = kamis_week["county_norm"].unique().tolist()
centroids = centroids[centroids["county_norm"].isin(present_counties)].reset_index(drop=True)


# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 63  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

counties = centroids["county_norm"].tolist()
dist_matrix = pd.DataFrame(np.zeros((len(counties), len(counties))),
                           index=counties, columns=counties)

for i, c1 in centroids.iterrows():
    for j, c2 in centroids.iterrows():
        dist_matrix.loc[c1["county_norm"], c2["county_norm"]] = haversine(
            c1["latitude"], c1["longitude"],
            c2["latitude"], c2["longitude"]
        )

nearest = {}
for c in counties:
    sorted_nei = dist_matrix.loc[c].sort_values()
    nearest[c] = list(sorted_nei.index[1:4])  # Skip itself at index 0

nearest_df = pd.DataFrame({
    "county": counties,
    "nearest_1": [nearest[c][0] for c in counties],
    "nearest_2": [nearest[c][1] for c in counties],
    "nearest_3": [nearest[c][2] for c in counties],
})

dist_matrix.shape
closest_to_target = nearest_df[nearest_df["county"].isin(target_counties)]
closest_to_target


all_panels = []

for c in target_counties:
    sub = kamis_week[kamis_week["county_norm"] == c].copy()
    if sub.empty:
        continue

    # Continuous weekly index from first to last observed for that county
    min_d = sub["week_start"].min()
    max_d = sub["week_start"].max()
    full_weeks = pd.date_range(min_d, max_d, freq="W-MON")

    df = pd.DataFrame({"week_start": full_weeks})
    df["county_norm"] = c

    df = df.merge(sub[["week_start", "kamis_price"]],
                  on="week_start", how="left")

    # Fill missing KAMIS prices within that county
    df["kamis_price"] = df["kamis_price"].ffill().bfill()

    # 3-week rolling mean smoothing
    df["kamis_smooth"] = df["kamis_price"].rolling(1, min_periods=1).mean()

    all_panels.append(df)

kamis_panel = pd.concat(all_panels, ignore_index=True)

# Merge with Agribora weekly prices
panel = kamis_panel.merge(
    agr_week,
    on=["county_norm", "week_start"],
    how="left"
)

panel = panel.sort_values(["county_norm", "week_start"])

for lag in [1, 2, 3]:
    panel[f"lag{lag}"] = panel.groupby("county_norm")["kamis_smooth"].shift(lag ** 3)

# Drop rows with missing lags or missing Agribora price
panel_train = panel.dropna(subset=["lag1", "lag2", "lag3", "agr_price"]).reset_index(drop=True)
X = panel_train[["kamis_smooth", "lag1", "lag2", "lag3", "county_norm"]]
y = panel_train["agr_price"]


numeric_features = ["kamis_smooth", "lag1", "lag2", "lag3"]
categorical_features = ["county_norm"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

enet = ElasticNet(alpha=1.0, l1_ratio=0.3, random_state=666)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("model", enet)
])

model.fit(X, y)

# Optionally inspect training fit quality:
y_pred_train = model.predict(X)
print("Train MAE:", mean_absolute_error(y, y_pred_train))
print("Train RMSE:", np.sqrt(mean_squared_error(y, y_pred_train)))


future_rows = []

for c in target_counties:
    hist = panel[panel["county_norm"] == c].sort_values("week_start").copy()
    if hist.empty:
        continue

    # Last three smoothed KAMIS prices (after filling + smoothing)
    last3 = hist["kamis_smooth"].tail(3).values

    # Ensure we have 3 values (fall back if series shorter)
    if len(last3) == 1:
        lag1 = lag2 = lag3 = last3[-1]
    elif len(last3) == 2:
        lag1 = last3[-1]
        lag2 = lag3 = last3[-2]
    else:
        lag1 = last3[-1]
        lag2 = last3[-2]
        lag3 = last3[-3]

    last_week = hist["week_start"].max()

    # --- Horizon 1 (next week) ---
    h1_date = last_week + timedelta(days=7)
    X_h1 = pd.DataFrame({
        "kamis_smooth": [lag2],  # use last known smooth KAMIS as base
        "lag1": [lag1],
        "lag2": [lag2],
        "lag3": [lag3],
        "county_norm": [c],
    })
    pred_h1 = model.predict(X_h1)[0]

    # --- Horizon 2 (two weeks ahead) ---
    h2_date = h1_date + timedelta(days=7)
    # For h+2, treat h+1 prediction as the new "current" level
    X_h2 = pd.DataFrame({
        "kamis_smooth": [pred_h1],
        "lag1": [pred_h1],
        "lag2": [lag1],
        "lag3": [lag2],
        "county_norm": [c],
    })
    pred_h2 = model.predict(X_h2)[0]

    future_rows.append({
        "county": c,
        "last_obs_week": last_week,
        "horizon1_date": h1_date,
        "agr_pred_h1": pred_h1,
        "horizon2_date": h2_date,
        "agr_pred_h2": pred_h2,
    })

future_df = pd.DataFrame(future_rows)
future_df
target_counties = ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]
panel_five = panel[panel["county_norm"].isin(target_counties)].copy()

# ---- 3. Create lags and training set ----
panel_five = panel_five.sort_values(["county_norm", "week_start"])

# ---- Recursive forecasts up to 2025-12-01 ----
target_start = pd.Timestamp("2025-11-24")
target_end = pd.Timestamp("2025-12-01")

forecast_rows = []

# Determine the global last observed week in the aligned panel
global_last_week = panel_five["week_start"].max()

for c in ["Kiambu", "Kirinyaga", "Mombasa", "Nairobi", "Uasin-Gishu"]:
    hist = panel_five[panel_five["county_norm"] == c].sort_values("week_start")
    if hist.empty:
        continue

    last3 = hist["kamis_smooth"].tail(3).values
    if len(last3) == 1:
        lag1 = lag2 = lag3 = last3[-1]
    elif len(last3) == 2:
        lag1 = last3[-1]
        lag2 = lag3 = last3[-2]
    else:
        lag1 = last3[-1]
        lag2 = last3[-2]
        lag3 = last3[-3]

    current_week = global_last_week

    while current_week < target_end:
        next_week = current_week + timedelta(days=7)
        X_h = pd.DataFrame({
            "kamis_smooth": [lag1],
            "lag1": [lag1],
            "lag2": [lag2],
            "lag3": [lag3],
            "county_norm": [c]
        })
        pred_h = model.predict(X_h)[0]

        forecast_rows.append({
            "county": c,
            "week_start": next_week,
            "agr_pred": pred_h
        })

        lag3 = lag2
        lag2 = lag1
        lag1 = pred_h
        current_week = next_week

forecast_df = pd.DataFrame(forecast_rows)

# Filter only weeks 2025-11-24 and 2025-12-01
mask = forecast_df["week_start"].isin([target_start, target_end])
forecast_target = forecast_df[mask].copy()

forecast_target = forecast_target.sort_values(['week_start'])
forecast_target
# We have forecast_target from previous cell
# Build submission for weeks 48 and 49 (derived from the dates)

forecast_target["week"] = forecast_target["week_start"].dt.isocalendar().week.astype(int)
forecast_target["ID"] = forecast_target["county"] + "_Week_" + forecast_target["week"].astype(str)
forecast_target["Target_RMSE"] = forecast_target["agr_pred"]
forecast_target["Target_MAE"] = forecast_target["agr_pred"]

submission = forecast_target[["ID", "Target_RMSE", "Target_MAE"]].reset_index(drop=True)

submission
samp_sub = pd.read_csv('SampleSubmission.csv')
samp_sub
submission = pd.concat([submission, samp_sub.iloc[10:]])
submission["Target_RMSE"] = submission["Target_RMSE"] * 0.98
submission_path = "SubSample.csv"
submission.to_csv(submission_path, index=False)