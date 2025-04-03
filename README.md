# CustomGBM: A Lightweight Gradient Boosting Model  

## 📌 Overview  
CustomGBM is a **custom-built gradient boosting model** designed for structured datasets like **salary prediction**. Unlike XGBoost, it follows a **simplified update mechanism** using residual correction, ensuring **interpretability and efficiency** while maintaining strong predictive performance.  

## ✨ Features  
✔ **Gradient Boosting with Decision Trees**  
✔ **Residual-based Updates (First-order Gradient Only)**  
✔ **No Complex Regularization (High Interpretability)**  
✔ **Customizable Learning Rate & Model Depth**  
✔ **Ideal for Small-to-Medium Structured Datasets**  

## 🔬 How It Works  
CustomGBM improves predictions by iteratively correcting residual errors:  
1. **Initialization**: Predicts the average salary as the baseline.  
2. **Boosting Iterations**:  
   - Computes residual errors.  
   - Trains a Decision Tree on the residuals.  
   - Updates predictions using the trained tree.  
3. **Final Prediction**: Aggregates all corrections to generate the final output.  

## 📊 Comparison: CustomGBM vs. XGBoost  
| Feature        | CustomGBM  | XGBoost |
|---------------|-----------|---------|
| **Loss Function** | Mean Squared Error (MSE) | Supports MSE, MAE, Log Loss, etc. |
| **Update Rule**   | Residual-based updates | Advanced optimization techniques |
| **Regularization** | None | L1/L2 Regularization |
| **Speed** | Slower (Single-threaded) | Faster (Parallelized, GPU-optimized) |
| **Interpretability** | Higher | Lower (Due to Complex Regularization) |
