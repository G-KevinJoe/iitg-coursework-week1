# 🌿 NDVI Classification for Agricultural Analysis

This project leverages NDVI (Normalized Difference Vegetation Index) time-series data to classify crop health or land types using statistical and machine learning techniques. It uses logistic regression with feature engineering and calibrated classification.

## 📁 Files

- `ndvi_classification_fixed.ipynb`: Jupyter Notebook containing the complete data pipeline, from preprocessing to model prediction and export.
- `hacktrain.csv`: Training dataset.
- `hacktest.csv`: Test dataset.
- `ndvi_final_submission.csv`: Output file with predicted class labels for the test set.
- `ndvi_classification_fixed.zip`: Zipped notebook for easy download and sharing.

## 📊 Dataset Description

The datasets contain NDVI readings over time (e.g., columns like `NDVI_1_N`, `NDVI_2_N`, ...), which are used to classify each sample into a respective class label.

- **Train file:** Includes NDVI features + `class` label.
- **Test file:** Includes NDVI features only.
- **Target:** Predict the `class` column for test data.

## 🔍 Feature Engineering

Custom statistical features are extracted:
- Mean, median, standard deviation, skewness, kurtosis of NDVI values
- Quartile-wise means
- Last-3 NDVI mean (recent trend)
- Missing value counts and ratios
- Derived metrics: NDVI stability and seasonality

## ⚙️ Model Pipeline

The pipeline includes:
1. **RobustScaler**: Scales features while being robust to outliers.
2. **SelectFromModel**: Feature selection using L1-penalized logistic regression.
3. **CalibratedClassifierCV**: Logistic regression with probability calibration.

### Model Details:
- Base estimator: Logistic Regression (L2)
- Calibration: 5-fold cross-validation
- Evaluation: 10-fold stratified cross-validation (Accuracy)

## 📈 Results

The model achieves stable performance across folds with high accuracy and reliability due to calibration.

## 🚀 How to Run

1. Ensure you have Python 3.8+ and install required libraries:
    ```bash
    pip install pandas numpy scikit-learn
    ```
2. Place `hacktrain.csv` and `hacktest.csv` in the same directory.
3. Run the notebook `ndvi_classification_fixed.ipynb`.
4. The predictions will be saved as `ndvi_final_submission.csv`.

## 📌 Author

Kevin Joe Prakash  
Computer Engineering | St. Francis Institute of Technology  
GitHub: [G-KevinJoe](https://github.com/G-KevinJoe)

---

**Note**: This is part of a machine learning-based solution for classifying satellite-based vegetation indices. The approach can be extended for time-series crop monitoring and yield estimation.
