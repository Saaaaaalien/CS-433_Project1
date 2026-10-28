import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import reg_logistic_regression
from src.preprocessing import to_01, to_pm1, standardize, add_bias

def main():
    print("Load data")
    # Load the FULL dataset (sub_sample=False)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/", sub_sample=False)
    
    # 1. Clean the data
    # Drop the 13 metadata columns
    cols_to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x_train_clean = np.delete(x_train, cols_to_drop, axis=1)
    x_test_clean = np.delete(x_test, cols_to_drop, axis=1)
    
    # Replace NaNs and Infs with 0
    x_train_clean = np.nan_to_num(x_train_clean, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_clean = np.nan_to_num(x_test_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert labels from {-1, 1} to {0, 1}
    y_train_01 = to_01(y_train)
    
    print("Undersample the training data")
    # 2. Undersample the majority class to match the minority class
    np.random.seed(42)  # IMPORTANT: Fix the seed for reproducibility!
    minority_indices = np.where(y_train_01 == 1)[0]
    majority_indices = np.where(y_train_01 == 0)[0]
    sampled_majority = np.random.choice(majority_indices, size=len(minority_indices), replace=False)
    
    balanced_indices = np.concatenate([minority_indices, sampled_majority])
    np.random.shuffle(balanced_indices)
    
    x_train_bal = x_train_clean[balanced_indices]
    y_train_bal = y_train_01[balanced_indices]
    
    print("Standardize and add bias")
    # 3. Standardize (fit on balanced train, apply to test)
    x_train_bal_norm, mean, std = standardize(x_train_bal)
    x_test_norm, _, _ = standardize(x_test_clean, mean, std)
    
    # 4. Add bias
    tx_train_bal = add_bias(x_train_bal_norm)
    tx_test = add_bias(x_test_norm)
    
    print("Train model")
    # 5. Train the model
    initial_w = np.zeros(tx_train_bal.shape[1])
    w, loss = reg_logistic_regression(
        y_train_bal, tx_train_bal, lambda_=1.0, initial_w=initial_w, max_iters=1000, gamma=0.1
    )
    
    print("Predict on test set")
    # 6. Predict
    test_predictions_01 = (tx_test @ w > 0.5).astype(int)
    test_predictions_pm1 = to_pm1(test_predictions_01)
    
    # 7. Generate submission
    create_csv_submission(test_ids, test_predictions_pm1, "submission.csv")
    print("Successfully created submission.csv")

if __name__ == "__main__":
    main()