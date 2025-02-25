# 🏡 House Price Prediction  

## 📌 Introduction  
This project aims to **predict house prices** based on real estate characteristics using **machine learning techniques**. The dataset is collected through **web scraping** and preprocessed before training various models.  

## 💊 Dataset Overview  
The dataset includes the following features:  
- **Property attributes**: Number of floors, bedrooms, bathrooms, area (m²), road frontage, legal status.  
- **Geographical location**: Latitude and longitude coordinates.  
- **Market trends**: Historical transaction data.  

## 🔄 Project Workflow  
Below is the workflow illustrating the key steps from data collection to model prediction:  

![Project Workflow](image.png)  

### 1⃣ Data Collection  
- **Web scraping** is used to extract real estate data from property listing websites.  
- The dataset consists of **7 key features** used for model training.  

```python
vn = scrape_this(province, num_page, district_list, province_wards)
```

### 2⃣ Data Preprocessing  
- **Removing duplicate entries**  
```python
data_VN = data_VN.drop_duplicates().reset_index(drop=True)
```
- **Handling inconsistent pricing** (converting currency units)  
```python
data_VN = handling_price(data_VN, 'Price')
```
- **Encoding categorical variables** (road frontage, legal status)  
```python
data_VN["Roadfrontage"] = list(map(int, data_VN["Roadfrontage"]))
data_VN.loc[data_VN['Legal'] == 'Good', 'Legal'] = int(1)
data_VN.loc[data_VN['Legal'].isna(), 'Legal'] = int(0)
data_VN['Legal'] = pd.to_numeric(data_VN['Legal'])
```
- **Converting addresses into geographical coordinates**  
```python
data_VN_encoding = createLatLong(data_VN)
```

### 3⃣ Model Development  
We train multiple machine learning models to predict house prices:  
- **eXtreme Gradient Boosting (XGBoost)**  
- **Histogram-based Gradient Boosting**  
- **CatBoost**  

```python
# Initialize models
model1 = xgb.XGBRegressor(max_depth=9,
                          learning_rate=0.1216,
                          n_estimators=632,
                          min_child_weight=52,
                          gamma=0.5109,
                          subsample=0.9977,
                          colsample_bytree=0.9719,
                          reg_alpha=0.9110,
                          reg_lambda=0.8850)

model2 = HistGradientBoostingRegressor(l2_regularization=5.39,
                                       learning_rate=0.1518,
                                       max_iter=201,
                                       max_depth=17,
                                       max_bins=251,
                                       max_leaf_nodes=37,
                                       min_samples_leaf=5)

model3 = CatBoostRegressor(l2_leaf_reg=4.3173,
                           max_bin=91,
                           subsample=0.9085,
                           learning_rate=0.1437,
                           n_estimators=423,
                           max_depth=7,
                           min_data_in_leaf=46,
                           verbose=False)

# Train models
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)

model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)

model3.fit(X_train_scaled, y_train)
y_pred3 = model3.predict(X_test_scaled)
```

### 4⃣ Model Ensemble (Weighted Averaging)  
To enhance performance, we apply **ensemble learning** by combining multiple models' predictions.  

```python
# Weighted Average
y_pred = y_pred1*0.25 + y_pred2*0.54 + y_pred3*0.21
```

- **Final Evaluation Metrics:**  
  - **R2 Score:** 0.711  
  - **RMSE:** 1.60  

## 🎯 Conclusion  
This project successfully applies **Machine Learning** techniques to **predict house prices**, improving valuation accuracy and supporting decision-making in the real estate market. 🚀  
