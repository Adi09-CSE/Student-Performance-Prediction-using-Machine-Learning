   Student Performance Prediction using Machine Learning 


A comprehensive machine learning project for predicting student performance using multiple regression algorithms. This project compares 6 different ML models with hyperparameter optimization and achieves 91.9% R² score using XGBoost.



   📋 Table of Contents
- [Overview]( overview)
- [Dataset]( dataset)
- [Project Workflow]( project-workflow)
- [Models Implemented]( models-implemented)
- [Results & Performance]( results--performance)
- [Installation]( installation)
- [Usage]( usage)
- [Project Structure]( project-structure)
- [Key Features]( key-features)
- [Visualizations]( visualizations)
- [Technologies Used]( technologies-used)
- [Future Enhancements]( future-enhancements)
- [Contributing]( contributing)
- [License]( license)
- [Contact]( contact)

---

   🎯 Overview

Student academic performance prediction is crucial for educational institutions to identify at-risk students early and provide timely interventions. This project builds predictive models to forecast student final grades (G3) based on various demographic, social, and academic factors.

    🎖️ Key Achievements
- ✅ Implemented   6 different ML algorithms   with optimization
- ✅ Achieved   91.9% R² score   with optimized XGBoost
- ✅ Comprehensive   exploratory data analysis   with 15+ visualizations
- ✅ Feature engineering   and selection using Decision Trees
- ✅ GridSearchCV   hyperparameter tuning for all models
- ✅ Handled   outliers   using Z-score method
- ✅ Complete   end-to-end ML pipeline  

---

   📊 Dataset

  Source:   Student Performance Dataset 
  Origin:   Kaggle 

    Dataset Statistics
-   Total Records:   649 students
-   Total Features:   33 (30 predictive + 3 grade features)
-   Target Variable:   G3 (Final Grade, 0-20 scale)
-   Missing Values:   None
-   Duplicates:   None

    Feature Categories

| Category | Features | Count |
|----------|----------|-------|
|   Demographic   | school, sex, age, address, famsize, Pstatus | 6 |
|   Educational   | Medu, Fedu, studytime, failures, schoolsup, paid, higher | 7 |
|   Social   | Mjob, Fjob, reason, guardian, traveltime, activities, nursery | 7 |
|   Behavioral   | goout, Dalc, Walc, freetime, romantic, famrel, health, absences | 8 |
|   Academic   | G1, G2, G3 (grades for three periods) | 3 |
|   Support   | internet, famsup | 2 |

    Selected Features (After Feature Selection)
Based on Decision Tree feature importance:
-   Walc   - Weekend alcohol consumption
-   absences   - Number of school absences
-   G1   - First period grade
-   G2   - Second period grade

---

   🔬 Project Workflow

    1.   Data Loading & Exploration  
- Loaded Portuguese student dataset (649 records, 33 features)
- Performed statistical analysis using `.describe()` and `.info()`
- Verified data integrity (no missing values, no duplicates)

    2.   Exploratory Data Analysis (EDA)  
Comprehensive visualization including:
- Gender distribution analysis
- Failure rates by sex, age, address type, and parent status
- Grade distribution across all three periods (G1, G2, G3)
- Correlation analysis with failures, study time, going out, and absences
- School-wise performance comparison

    3.   Data Preprocessing  
-   Label Encoding:   Converted 17 categorical features to numerical
-   Outlier Detection & Removal:   Applied Z-score method (threshold < 3)
- Reduced dataset from 649 to 528 records after outlier removal
-   Correlation Analysis:   Generated heatmap for feature relationships

    4.   Feature Engineering & Selection  
- Used `DecisionTreeClassifier` with `SelectFromModel`
- Selected 4 most important features: Walc, absences, G1, G2
- Focused on features with highest predictive power

    5.   Model Development  
Implemented and optimized 6 regression models:
1. Random Forest Regressor (RFR)
2. Decision Tree Regressor (DTR)
3. Linear Regression (LR)
4. Support Vector Machine Regressor (SVMR)
5. XGBoost Regressor (XGBR) ⭐   Best Model  
6. K-Nearest Neighbors Regressor (KNNR)

    6.   Hyperparameter Optimization  
- Applied `GridSearchCV` with 5-fold cross-validation
- Tuned parameters specific to each algorithm
- Compared baseline vs optimized performance

---

   🤖 Models Implemented

    1. Random Forest Regressor
  Baseline Model:  
```python
RandomForestRegressor(random_state=100, criterion='squared_error', 
                      max_depth=30, min_samples_leaf=5, n_jobs=1)
```
  Optimized Parameters:   `criterion='squared_error', max_depth=4, n_jobs=-1, random_state=13`

    2. Decision Tree Regressor
  Baseline Model:  
```python
DecisionTreeRegressor(random_state=100, criterion='squared_error',
                      max_depth=30, min_samples_leaf=5)
```
  Optimized Parameters:   `criterion='squared_error', max_depth=5, min_samples_leaf=12, random_state=5`

    3. Linear Regression
  Baseline Model:  
```python
LinearRegression(fit_intercept=True, n_jobs=1)
```
  Optimized Parameters:   `fit_intercept=True, n_jobs=1`

    4. Support Vector Machine Regressor
  Baseline Model:  
```python
SVR(kernel='poly')
```
  Optimized Parameters:   `degree=1, gamma='scale', kernel='linear'`

    5. XGBoost Regressor ⭐   Best Performance  
  Baseline Model:  
```python
XGBRegressor(gamma=0.3, random_state=42, n_estimators=11,
             n_jobs=-1, max_depth=10)
```
  Optimized Parameters:   `max_depth=2, n_estimators=10, n_jobs=1, random_state=5`

    6. K-Nearest Neighbors Regressor
  Baseline Model:  
```python
KNeighborsRegressor(n_neighbors=7, n_jobs=1, metric='manhattan')
```
  Optimized Parameters:   `metric='manhattan', n_jobs=1, n_neighbors=13`

---

   📈 Results & Performance

    Model Comparison Table

| Model | MSE (Baseline) | RMSE (Baseline) | R² Score (Baseline) | MSE (Optimized) | RMSE (Optimized) | R² Score (Optimized) | Improvement |
|-------|----------------|-----------------|---------------------|-----------------|------------------|----------------------|-------------|
|   Random Forest   | 0.4961 | 0.7043 | 0.9118 | 0.4885 | 0.6990 | 0.9132 | ✅ +0.14% |
|   Decision Tree   | 0.5481 | 0.7404 | 0.9026 | 0.5325 | 0.7298 | 0.9053 | ✅ +0.27% |
|   Linear Regression   | 0.4986 | 0.7061 | 0.9114 | 0.4986 | 0.7061 | 0.9114 | - |
|   SVM   | 0.7646 | 0.8744 | 0.8641 | 0.4950 | 0.7036 | 0.9120 | ✅ +4.79% |
|   XGBoost   ⭐ | 0.5638 | 0.7508 | 0.8998 |   0.4546   |   0.6742   |   0.9192   | ✅   +1.94%   |
|   KNN   | 0.5762 | 0.7591 | 0.8976 | 0.6091 | 0.7805 | 0.8917 | ❌ -0.59% |

    🏆 Best Model: Optimized XGBoost Regressor
-   Mean Squared Error (MSE):   0.4546
-   Root Mean Squared Error (RMSE):   0.6742
-   R² Score:   0.9192 (91.92% variance explained)
-   Average Prediction Error:   ±0.67 grade points (on 0-20 scale)

    Key Insights
1.   XGBoost   achieved the best performance after optimization
2.   SVM   showed the most significant improvement (+4.79%) after tuning
3.   Linear Regression   performed surprisingly well without optimization
4.   KNN   was the only model that performed worse after optimization
5. All models achieved R² > 0.89, indicating strong predictive power

---

   ✨ Key Features

    1.   Comprehensive EDA  
- 15+ visualization plots
- Gender-based failure analysis
- Age-wise performance distribution
- Parental status impact on grades
- School comparison analysis

    2.   Advanced Preprocessing  
- Label encoding for 17 categorical variables
- Z-score based outlier removal
- Correlation analysis
- Feature selection using tree-based importance

    3.   Multiple Model Comparison  
- 6 different regression algorithms
- Baseline vs optimized comparison
- GridSearchCV hyperparameter tuning
- Cross-validation with 5 folds

    4.   Robust Evaluation  
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)
- Before/after optimization comparison

---

   📊 Visualizations

The project includes extensive visualizations:

1.   Distribution Plots  
   - Gender distribution count plot
   - Failure rates by demographics

2.   Relationship Analysis  
   - Failures vs Sex, Age, Address, Pstatus
   - Study time impact on grades (G1, G2, G3)
   - Going out frequency vs performance
   - Absences effect on all grade periods

3.   Correlation Analysis  
   - 33x33 feature correlation heatmap
   - Feature importance from Decision Trees

4.   Performance Metrics  
   - Model comparison charts
   - Before/after optimization results

---

   🛠️ Technologies Used

    Core Libraries
-   Python 3.8+   - Programming language
-   pandas   - Data manipulation and analysis
-   NumPy   - Numerical computations
-   scikit-learn   - Machine learning algorithms
-   XGBoost   - Gradient boosting framework

    Visualization
-   Matplotlib   - Static plots
-   Seaborn   - Statistical visualizations

    Machine Learning Models
-   RandomForestRegressor   - Ensemble learning
-   DecisionTreeRegressor   - Tree-based model
-   LinearRegression   - Linear model
-   SVR   - Support Vector Regression
-   XGBRegressor   - Gradient boosting
-   KNeighborsRegressor   - Instance-based learning

    Optimization & Validation
-   GridSearchCV   - Hyperparameter tuning
-   train_test_split   - Data splitting
-   Cross-validation   - Model validation

---

   🔮 Future Enhancements

- [ ] Implement deep learning models (Neural Networks, LSTM)
- [ ] Add ensemble methods (Stacking, Blending)
- [ ] Feature engineering with polynomial features
- [ ] Develop web application using Flask/Streamlit
- [ ] Create REST API for predictions
- [ ] Add real-time prediction dashboard
- [ ] Implement SHAP for model interpretability
- [ ] Extend to multi-class classification (Pass/Fail/Excellent)
- [ ] Add time-series analysis for grade progression
- [ ] Deploy model on cloud (AWS/GCP/Azure)
- [ ] Create mobile app for predictions
- [ ] Implement AutoML for automated model selection

---

   📧 Contact

  Abu Taher

  . LinkedIn: https://www.linkedin.com/in/abu-taher-adi-/
  . Email: abutaher643309@gmail.com
  . Kaggle: https://www.kaggle.com/adicse09

---

   🙏 Acknowledgments

-   Dataset:       Kaggle
-   Inspiration:   Student performance analysis research papers
-   Libraries:     scikit-learn, XGBoost, pandas communities
-   References:  
  - Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance.
  - Educational data mining research papers

---

   📊 Project Statistics

  Dataset Size:   649 students  
  Features:   33 variables  
  Models:   6 algorithms  
  Best R² Score:   91.92%  
  Code Lines:   ~1000+  
  Visualizations:   15+

---

   📈 Model Performance Summary

```
╔══════════════════════╦═══════════╦══════════╦═══════════╗
║ Model                ║ RMSE      ║ R² Score ║ Rank      ║
╠══════════════════════╬═══════════╬══════════╬═══════════╣
║ XGBoost (Optimized)  ║ 0.6742    ║ 0.9192   ║ 🥇 1st    ║
║ Random Forest (Opt)  ║ 0.6990    ║ 0.9132   ║ 🥈 2nd    ║
║ SVM (Optimized)      ║ 0.7036    ║ 0.9120   ║ 🥉 3rd    ║
║ Linear Regression    ║ 0.7061    ║ 0.9114   ║    4th    ║
║ Decision Tree (Opt)  ║ 0.7298    ║ 0.9053   ║    5th    ║
║ KNN (Optimized)      ║ 0.7805    ║ 0.8917   ║    6th    ║
╚══════════════════════╩═══════════╩══════════╩═══════════╝
```

