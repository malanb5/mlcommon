"""
implementation of the Light GBM model for this competition
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from yj import Shaper

# Correct data types for "calendar.csv"
calendarDTypes = {"event_name_1": "category",
                  "event_name_2": "category",
                  "event_type_1": "category",
                  "event_type_2": "category",
                  "weekday": "category",
                  'wm_yr_wk': 'int16',
                  "wday": "int16",
                  "month": "int16",
                  "year": "int16",
                  "snap_CA": "float32",
                  'snap_TX': 'float32',
                  'snap_WI': 'float32'}

# Read csv file
calendar = pd.read_csv("../data/calendar.csv",
                       dtype=calendarDTypes)

calendar = Shaper.apply_label_before(calendar, ['event_name_1', 'event_name_1', 'event_type_1', 'event_type_2'], 7)

calendar["date"] = pd.to_datetime(calendar["date"])

# Transform categorical features into integers
for col, colDType in calendarDTypes.items():
    if colDType == "category":
        calendar[col] = calendar[col].cat.codes.astype("int16")
        calendar[col] -= calendar[col].min()

# Correct data types for "sell_prices.csv"
priceDTypes = {"store_id": "category",
               "item_id": "category",
               "wm_yr_wk": "int16",
               "sell_price": "float32"}

# Read csv file
prices = pd.read_csv("../data/sell_prices.csv",
                     dtype=priceDTypes)

# Transform categorical features into integers
for col, colDType in priceDTypes.items():
    if colDType == "category":
        prices[col] = prices[col].cat.codes.astype("int16")
        prices[col] -= prices[col].min()

### *sales_train_validation.csv*

firstDay = 250
lastDay = 1913

# Use x sales days (columns) for training
numCols = [f"d_{day}" for day in range(firstDay, lastDay + 1)]

# Define all categorical columns
catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

# Define the correct data types for "sales_train_validation.csv"
dtype = {numCol: "float32" for numCol in numCols}
dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

# Read csv file
ds = pd.read_csv("../data/sales_train_validation.csv",
                 usecols=catCols + numCols, dtype=dtype)

# Transform categorical features into integers
for col in catCols:
    if col != "id":
        ds[col] = ds[col].cat.codes.astype("int16")
        ds[col] -= ds[col].min()

ds = pd.melt(ds,
             id_vars=catCols,
             value_vars=[col for col in ds.columns if col.startswith("d_")],
             var_name="d",
             value_name="sales")

# Merge "ds" with "calendar" and "prices" dataframe
ds = ds.merge(calendar, on="d", copy=False)
ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

print("creating features: lag features")

dayLags = [7, 28]
lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
    ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

windows = [7, 28]
for window in windows:
    for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
        ds[f"rmean_{dayLag}_{window}"] = ds[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(
            lambda x: x.rolling(window).mean())


print("creating features: date features")

dateFeatures = {"wday": "weekday",
                "week": "weekofyear",
                "month": "month",
                "quarter": "quarter",
                "year": "year",
                "mday": "day"}

for featName, featFunc in dateFeatures.items():
    if featName in ds.columns:
        ds[featName] = ds[featName].astype("int16")
    else:
        ds[featName] = getattr(ds["date"].dt, featFunc).astype("int16")

# %% md
print("cleaning up")
### Remove unnecessary rows and columns

# Remove all rows with NaN value
ds.dropna(inplace=True)

# Define columns that need to be removed
unusedCols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday"]
trainCols = ds.columns[~ds.columns.isin(unusedCols)]

X_train = ds[trainCols]
y_train = ds["sales"]

np.random.seed(777)

# Define categorical features
catFeats = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id'] + \
           ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

validInds = np.random.choice(X_train.index.values, 2_000_000, replace=False)
trainInds = np.setdiff1d(X_train.index.values, validInds)

trainData = lgb.Dataset(X_train.loc[trainInds], label=y_train.loc[trainInds],
                        categorical_feature=catFeats, free_raw_data=False)
validData = lgb.Dataset(X_train.loc[validInds], label=y_train.loc[validInds],
                        categorical_feature=catFeats, free_raw_data=False)

del ds, X_train, y_train, validInds, trainInds;
gc.collect()

# Model

params = {
    "objective": "poisson",
    "metric": "rmse",
    "force_row_wise": True,
    "learning_rate": 0.075,
    "sub_row": 0.75,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations': 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}

# %%
print("training lgb model")

# Train LightGBM model
m_lgb = lgb.train(params, trainData, valid_sets=[validData], verbose_eval=20)
# m_lgb = lgb.Booster(model_file="model.lgb")

# %%

# print("saving the model")
# # Save the model
m_lgb.save_model("model_events_before.lgb")

# %% md

# Predictions
print("making predicitons")

# exit(0)

# Last day used for training
trLast = 1913
# Maximum lag day
maxLags = 57


# Create dataset for predictions
def create_ds():
    startDay = trLast - maxLags

    numCols = [f"d_{day}" for day in range(startDay, trLast + 1)]
    catCols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']

    dtype = {numCol: "float32" for numCol in numCols}
    dtype.update({catCol: "category" for catCol in catCols if catCol != "id"})

    ds = pd.read_csv("../data/sales_train_validation.csv",
                     usecols=catCols + numCols, dtype=dtype)

    for col in catCols:
        if col != "id":
            ds[col] = ds[col].cat.codes.astype("int16")
            ds[col] -= ds[col].min()

    for day in range(trLast + 1, trLast + 28 + 1):
        ds[f"d_{day}"] = np.nan

    ds = pd.melt(ds,
                 id_vars=catCols,
                 value_vars=[col for col in ds.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    ds = ds.merge(calendar, on="d", copy=False)
    ds = ds.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    return ds


def create_features(ds):
    dayLags = [7, 28]
    lagSalesCols = [f"lag_{dayLag}" for dayLag in dayLags]
    for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
        ds[lagSalesCol] = ds[["id", "sales"]].groupby("id")["sales"].shift(dayLag)

    windows = [7, 28]
    for window in windows:
        for dayLag, lagSalesCol in zip(dayLags, lagSalesCols):
            ds[f"rmean_{dayLag}_{window}"] = ds[["id", lagSalesCol]].groupby("id")[lagSalesCol].transform(
                lambda x: x.rolling(window).mean())

    dateFeatures = {"wday": "weekday",
                    "week": "weekofyear",
                    "month": "month",
                    "quarter": "quarter",
                    "year": "year",
                    "mday": "day"}

    for featName, featFunc in dateFeatures.items():
        if featName in ds.columns:
            ds[featName] = ds[featName].astype("int16")
        else:
            ds[featName] = getattr(ds["date"].dt, featFunc).astype("int16")


# %%

fday = datetime(2016, 4, 25)
alphas = [1.028, 1.023, 1.018]
weights = [1 / len(alphas)] * len(alphas)
sub = 0.

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_ds()
    cols = [f"F{i}" for i in range(1, 29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te['date'] >= day - timedelta(days=maxLags)) & (te['date'] <= day)].copy()
        create_features(tst)
        tst = tst.loc[tst['date'] == day, trainCols]
        te.loc[te['date'] == day, "sales"] = alpha * m_lgb.naive_predict(tst)  # magic multiplier by kyakovlev

    te_sub = te.loc[te['date'] >= fday, ["id", "sales"]].copy()
    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)
    te_sub.to_csv(f"submission_{icount}.csv", index=False)
    if icount == 0:
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols] * weight
    print(icount, alpha, weight)


print("creating csv file submissions")
sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_lgbm_tree_20_4_20.csv", index=False)