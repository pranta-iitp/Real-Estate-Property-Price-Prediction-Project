# Real Estate Price Prediction with Machine Learning

This project develops a comprehensive machine learning pipeline for predicting real estate property prices in Gurgaon, India. The system utilizes advanced data preprocessing techniques, multiple encoding strategies, and ensemble learning algorithms optimized with Optuna to achieve accurate price predictions across various property segments.

## 📌 Project Highlights

- 🏘️ **Real Estate Dataset**: About 3500 properties ranging from ₹0.1 to ₹31 crores
- 🧹 **Advanced Data Preprocessing**: Outlier detection, missing value treatment, and domain-specific cleaning
- 🏗️ **Feature Engineering**: Luxury scoring system, area-based metrics, and categorical transformations
- 🎯 **Multi-Strategy Encoding**: One-Hot, Ordinal, and Target encoding for optimal categorical handling
- 🤖 **11 ML Algorithms**: From Linear Regression to XGBoost with comprehensive evaluation
- 🔧 **Hyperparameter Optimization**: GridSearchCV and Optuna implementation for Random Forest tuning
- 📊 **Comprehensive Metrics**: R², MAE, RMSE with business-relevant interpretations
- ⚡ **Production-Ready Pipeline**: Scikit-learn pipelines with cross-validation and robust preprocessing

## 🏠 Dataset Overview

### Property Distribution
- **Total Properties**: 3,500 real estate listings
- **Property Types**: 
  - Flats: 76% of dataset
  - Houses: 24% of dataset
- **Price Range**: ₹0.1 to ₹31 crores
- **Location**: Gurgaon, India with sector-based pricing variations

### Key Features
- **Categorical Variables**: Sector, property type, furnishing type, age possession, luxury category
- **Numerical Variables**: Built-up area, bedrooms, bathrooms, floor number, luxury score
- **Target Variable**: Property price (log-transformed for modeling)

## 🔧 Data Preprocessing Pipeline

### Data Quality Improvements
- **Negative Price Correction**: Replaced negative values with 0 using domain logic
- **Floor Number Capping**: Applied realistic bounds (-3 to 50 floors) using percentile-based outlier treatment
- **Price per Sqft Filtering**: Domain-specific bounds (₹1,000 - ₹50,000 per sqft) to preserve legitimate luxury properties

### Feature Engineering
- **Luxury Score Development**: Comprehensive scoring system based on property amenities
- **Area per Bedroom**: Calculated as `built_up_area / bedrooms` for space efficiency analysis
- **Categorical Hierarchies**: Converted luxury scores to Low/Medium/High categories using price-informed thresholds

## 🎯 Categorical Encoding Strategies

### Three-Tier Encoding Approach

#### One-Hot Encoding
```python
columns_to_encode_OHE = ['property_type', 'agePossession']
```
- **Use Case**: Nominal categorical variables with no natural ordering
- **Configuration**: `drop='first'`, `handle_unknown='ignore'`

#### Ordinal Encoding  
```python
columns_to_encode_OE = ['balcony', 'luxury_category', 'floor_category', 'furnishing_type']
```
- **Use Case**: Naturally ordered categories (Low < Medium < High)
- **Advantage**: Preserves hierarchical relationships

#### Target Encoding
```python
target_encoding_cols = ['sector']
```
- **Use Case**: High-cardinality categorical variables with strong price relationships
- **Implementation**: Cross-validation with regularization to prevent overfitting

## 🤖 Model Development & Evaluation

### Algorithm Portfolio
```python
model_dict = {
    'linear_reg': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'svr': SVR(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'extra_trees': ExtraTreesRegressor(),
    'gradient_boosting': GradientBoostingRegressor(),
    'adaboost': AdaBoostRegressor(),
    'mlp': MLPRegressor(),
    'xgboost': XGBRegressor()
}
```

### Performance Metrics
| Metric | Description | Business Value |
|--------|-------------|----------------|
| **CV_R2_Mean** | Cross-validation R² average | Overall model predictive power |
| **CV_R2_Std** | R² standard deviation | Model stability and consistency |
| **MAE_Crores** | Mean absolute error in crores | Average prediction error magnitude |
| **RMSE_Crores** | Root mean squared error | Penalty for large prediction errors |

## 🔍 Hyperparameter Optimization

### GridSearchCV Implementation
```python
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, 30, None],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 'log2', 0.3],
    'regressor__max_samples': [0.8, 0.9, 1.0]
}
```


## 📈 Results & Performance

### Model Performance Range
- **Cross-Validation R²**: 0.88 - 0.90 (achieved by Random Forest variants)
- **Test Set Performance**: Consistent with CV results indicating good generalization
- **Error Rates**: MAE typically 35-40 lakhs(moderate)

### Feature Importance Insights
1. **Built-up Area**: Strongest predictor of property prices
2. **Luxury Score**: Significant impact on premium property valuations  
3. **Location (Sector)**: High influence through target encoding
4. **Property Type**: Clear differentiation between flats and houses

## 🚀 Technical Implementation

### Technology Stack
- **Core ML**: scikit-learn, pandas, numpy
- **Specialized Tools**: category_encoders, xgboost
- **Visualization**: matplotlib, seaborn
- **Optimization**: GridSearchCV, Optuna TPE sampler

### Pipeline Architecture
```python
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_columns),
        ('cat', OrdinalEncoder(), ordinal_columns),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), nominal_columns),
        ('target', TargetEncoder(cv=5), high_cardinality_columns)
    ])),
    ('regressor', RandomForestRegressor(**optimized_params))
])
```

## 🎯 Key Insights & Business Impact

### Market Dynamics
- **Price Distribution**: Heavily right-skewed with concentration in 2-9 crore range
- **Luxury Segment**: Properties above 15 crores represent distinct market with different pricing factors
- **Location Premium**: Sector-based pricing variations captured effectively through target encoding

### Model Learnings
- **Ensemble Superiority**: Random Forest and Gradient Boosting outperformed linear models
- **Feature Engineering Impact**: Luxury scoring and area-based features significantly improved predictions
- **Encoding Strategy**: Mixed encoding approach optimal for real estate categorical variables

## 🔮 Future Enhancements

### Potential Improvements
- **Temporal Analysis**: Time-series components for market trend incorporation
- **Ensemble Stacking**: Advanced ensemble methods combining multiple base models
- **Feature Expansion**: Additional property characteristics and market indicators
- **Real-time Deployment**: Production-ready scoring system for live predictions

*This project demonstrates advanced machine learning techniques applied to real estate price prediction, showcasing comprehensive data preprocessing, feature engineering, and model optimization strategies suitable for production deployment.*
