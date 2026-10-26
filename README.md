# CS-433_Project1

## Team Members
- Li Lynn Sheryl Phoa
- Yilin Fang
- Yuet Wah Chan

## Project Structure
- `implementations.py`: Contains the 6  methods listed on the project description with one helper function.
- `run.py`: The main script to reproduce our submission.
- `src/preprocessing.py`: Helper functions for data cleaning, scaling, and label encoding.
- `notebooks/01_eda.ipynb`: Jupyter notebook containing exploratory data analysis and model tuning.

## Data Preparation & Feature Generation
To prepare the raw data for the models, we applied the following pipeline:
1. **Metadata Removal**: The first 13 columns of the dataset (e.g., `SEQNO`, `IDATE`, `DISPCODE`) are survey metadata and IDs. These hold no medical predictive power and were dropped.
2. **Missing Value Imputation**: We found that missing values in the CSV (empty fields) were parsed as `NaN`. We replaced all `NaN` and `Inf` values with `0.0` using `np.nan_to_num`.
3. **Standardization**: Features were z-score normalized (mean 0, standard deviation 1). This prevents gradient explosion when training on 300+ features.
4. **Bias Term**: A column of 1s was appended to the feature matrix to serve as the intercept (\(w_0\)).
5. **Undersampling**: The dataset is highly imbalanced (91.5% negative class, 8.5% positive class). To prevent the model from simply predicting the majority class, we randomly undersampled the negative class in the **training folds** to create a 1:1 balanced training set.

## Cross-Validation Steps
**5-Fold Cross-Validation** is used:
1. The training data was split into 5 random, equal-sized folds using a fixed seed.
2. For each fold, 4 folds were used for training and 1 for validation.
3. **Avoiding Data Leakage**: Inside the training loop, undersampling and standardization (calculating mean and std) were performed **on the 4 training folds**. The validation fold was transformed using the mean and std learned from the training folds.
4. **Hyperparameter Tuning**: We swept \(\lambda \in [0.001, 0.01, 0.1, 1.0, 10.0]\) using cross-validation and found that \(\lambda = 1.0\) maximized the F1-score.

## How to Run
1. Ensure the dataset (`x_train.csv`, `x_test.csv`, `y_train.csv`) is placed in a folder named `data/` at the root of this repository.
2. Install the required packages:
   ```bash
   pip install numpy
   ```
3. Run the submission script:
   ```bash
   python run.py
   ```
4. This will generate `submission.csv` in the root directory.
