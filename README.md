# CustomGBM: A Lightweight Gradient Boosting Model  

## 📌 Overview  
CustomGBM is a **gradient boosting algorithm** designed for structured datasets like **salary prediction**. Unlike XGBoost, it follows a **simplified update mechanism** using residual correction, ensuring **interpretability and efficiency** while maintaining strong predictive performance.  

## ✨ Features  
✔ **Gradient Boosting with Decision Trees**  
✔ **Residual-based Updates (First-order Gradient Only)**  
✔ **No Complex Regularization (High Interpretability)**  
✔ **Customizable Learning Rate & Model Depth**  
✔ **Ideal for Small-to-Medium Structured Datasets**  

## 🔬 How It Works  
CustomGBM improves predictions by iteratively **correcting residual errors**:  
1. **Initialization**: Predict the mean salary as the baseline.  
2. **Boosting Iterations**:  
   - Compute **residuals**: \( r_i = y_i - \hat{y}_i \)  
   - Train a Decision Tree on **residuals**  
   - Update predictions:  
     \[
     \hat{y}_i^{(t+1)} = \hat{y}_i^{(t)} + \eta h_m(X_i)
     \]
3. **Final Prediction**: Sum initial prediction and weak learner outputs.  

## 📊 Comparison: CustomGBM vs. XGBoost  
| Feature        | CustomGBM  | XGBoost |
|---------------|-----------|---------|
| **Loss Function** | MSE \( \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \) | Supports MSE, MAE, Log Loss, etc. |
| **Update Rule**   | \( F_{m+1} = F_m + \eta h_m \) | \( F_{m+1} = F_m + \sum \gamma_j h_{m,j} \) |
| **Regularization** | None | L1/L2 Regularization |
| **Speed** | Slower (Single-threaded) | Faster (Parallelized, GPU-optimized) |
| **Interpretability** | Higher | Lower (Due to Complex Regularization) |
