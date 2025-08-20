import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- Helper Function to Parse Column Names ---
def parse_perturbation_name(col_name):
    """
    Parses a column name, stripping potential pandas duplicate suffixes (e.g., '.1', '.39').
    Examples: 'g0261+g0760.39' -> ('double', 'g0261', 'g0760')
              'g0495+ctrl.2' -> ('single', 'g0495', None)
    Returns a tuple: (type, gene1, gene2 or None, base_name).
    """
    base_name = col_name
    # Check for and remove pandas' duplicate column suffix (e.g., '.1', '.39')
    if '.' in base_name:
        parts = base_name.rsplit('.', 1)
        if parts[1].isdigit():
            base_name = parts[0]

    genes = base_name.split('+')
    if len(genes) == 2:
        if 'ctrl' in genes:
            gene1 = genes[0] if genes[1] == 'ctrl' else genes[1]
            return ('single', gene1, None, f"{gene1}+ctrl")
        else:
            genes.sort()
            return ('double', genes[0], genes[1], f"{genes[0]}+{genes[1]}")
    
    # Return 'unknown' if the format is not recognized
    return ('unknown', None, None, col_name)

# ==============================================================================
# --- MAIN SCRIPT ---
# ==============================================================================

# --- Step 1: User Inputs ---
# ------------------------------------------------------------------------------
# 1.1: DEFINE THE PATH TO YOUR DATA FILE
DATA_FILE_PATH = 'data/train_set.csv'
TEST_FILE_PATH = 'data/test_set.csv'

# 1.2: DEFINE THE DOUBLE-GENE PAIRS TO PREDICT
df_test_data = pd.read_csv(TEST_FILE_PATH, header=None)
rows_list = df_test_data[0].tolist()
test_double_pairs_tuples = []
for row in rows_list:
    double_gene = row.split('+')
    test_double_pairs_tuples.append((double_gene[0], double_gene[1]))

for r in test_double_pairs_tuples:
    print(r)
# ------------------------------------------------------------------------------


# --- NEW OPTIONAL SETTING ---
# Set to True to perform validation, False to skip and train on all data directly.
PERFORM_VALIDATION = True
VALIDATION_SET_SIZE = 0.2 # 20% of the data will be used for validation
# ------------------------------------------------------------------------------

# --- Step 2: Load and Preprocess Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
df_raw_data = pd.read_csv(DATA_FILE_PATH, index_col=0)
print(f"Data loaded successfully. Shape: {df_raw_data.shape}")

# Average replicates using the new parser to group correctly
print("Averaging replicates for each unique condition...")
avg_expression_by_perturbation = {}
unrecognized_columns = []
for col_name in df_raw_data.columns:
    ptype, _, _, base_name = parse_perturbation_name(col_name)
    if ptype == 'unknown':
        unrecognized_columns.append(col_name)
        continue # Skip columns with formats we don't understand
    
    if base_name not in avg_expression_by_perturbation:
        avg_expression_by_perturbation[base_name] = []
    avg_expression_by_perturbation[base_name].append(df_raw_data[col_name].values)

if unrecognized_columns:
    print(f"\nWarning: Found {len(unrecognized_columns)} column(s) with unrecognized names. They will be ignored.")
    print(f"Examples: {unrecognized_columns[:5]}")

avg_expression_profiles_df = pd.DataFrame(index=df_raw_data.index)
for name, data_list in avg_expression_by_perturbation.items():
    avg_expression_profiles_df[name] = np.mean(data_list, axis=0)
print(f"Averaged {len(avg_expression_profiles_df.columns)} unique perturbation profiles.")

print("\nCreating a 'pseudo-control' profile using the median of all averaged profiles.")
pseudo_control_profile = avg_expression_profiles_df.median(axis=1).values
avg_expression_profiles = {name: avg_expression_profiles_df[name].values for name in avg_expression_profiles_df.columns}
delta_profiles = {name: profile - pseudo_control_profile for name, profile in avg_expression_profiles.items()}
print("Delta profiles calculated relative to the pseudo-control.")


# --- Step 3: Prepare FULL Training Dataset ---
X_train_full, Y_train_full = [], []
print("\nBuilding full training dataset from all available double-gene perturbations...")
for perturbation_name in avg_expression_profiles.keys():
    ptype, gene1, gene2, _ = parse_perturbation_name(perturbation_name)
    if ptype == 'double':
        single_g1_name, single_g2_name = f"{gene1}+ctrl", f"{gene2}+ctrl"
        double_pair_name = f"{gene1}+{gene2}"
        if single_g1_name in delta_profiles and single_g2_name in delta_profiles:
            X_train_full.append(np.concatenate((delta_profiles[single_g1_name], delta_profiles[single_g2_name])))
            Y_train_full.append(delta_profiles[double_pair_name])
X_train_full, Y_train_full = np.array(X_train_full), np.array(Y_train_full)
if len(X_train_full) == 0: raise ValueError("Full training data could not be constructed.")
print(f"Full training data constructed. X_train_full shape: {X_train_full.shape}, Y_train_full shape: {Y_train_full.shape}")


# --- Optional Model Validation ---
if PERFORM_VALIDATION:
    print("\n--- Optional: Performing Model Validation ---")
    
    # Split the full dataset into a training and a validation set
    X_train_split, X_val, Y_train_split, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=VALIDATION_SET_SIZE, random_state=42
    )
    print(f"Split data into {len(X_train_split)} training samples and {len(X_val)} validation samples.")

    # Train a temporary model ONLY on the smaller training split
    validation_model = MultiOutputRegressor(LinearRegression())
    print("Training validation model on the 80% split...")
    validation_model.fit(X_train_split, Y_train_split)
    
    # Make predictions on the held-out validation set
    print("Making predictions on the 20% validation set...")
    Y_pred_val = validation_model.predict(X_val)
    
    # Calculate and report accuracy metrics
    print("\n--- Validation Accuracy Metrics ---")
    
    # 1. Mean Squared Error (MSE): Lower is better. Measures the average squared difference.
    mse = mean_squared_error(Y_val, Y_pred_val)
    print(f"Overall Mean Squared Error (MSE): {mse:.4f}")
    
    # 2. R-squared (R²) Score: Higher is better (max is 1.0). Proportion of variance explained.
    r2 = r2_score(Y_val, Y_pred_val)
    print(f"Overall R-squared (R²) Score: {r2:.4f}")

    # 3. Per-Profile Pearson Correlation: Measures how well the "shape" of each predicted profile matches the true one.
    correlations = []
    for i in range(len(Y_val)):
        true_profile = Y_val[i]
        pred_profile = Y_pred_val[i]
        # pearsonr returns (correlation, p-value), we only need the correlation
        corr, _ = pearsonr(true_profile, pred_profile)
        correlations.append(corr)
    avg_corr = np.mean(correlations)
    print(f"Average Pearson Correlation per Profile: {avg_corr:.4f}")
    print("---------------------------------------\n")

# --- Step 4: Train the Final Model ---
# This model is trained on ALL available data to make the best possible predictions for the hackathon.
print("Training final model on ALL available data...")
final_model = MultiOutputRegressor(LinearRegression())
final_model.fit(X_train_full, Y_train_full)
print("Final model training complete.")


# --- Step 5: Prepare Hackathon Test Data and Make Final Predictions ---
X_test_list, test_pairs_for_prediction = [], []
print("\nBuilding official test dataset for final prediction...")
for gene1, gene2 in test_double_pairs_tuples:
    # ... (rest of the prediction logic is the same)
    g1_sorted, g2_sorted = sorted((gene1, gene2))
    single_g1_name, single_g2_name = f"{g1_sorted}+ctrl", f"{g2_sorted}+ctrl"
    if single_g1_name in delta_profiles and single_g2_name in delta_profiles:
        X_test_list.append(np.concatenate((delta_profiles[single_g1_name], delta_profiles[single_g2_name])))
        test_pairs_for_prediction.append((g1_sorted, g2_sorted))
    else:
        print(f"Error: Cannot predict for pair {g1_sorted}+{g2_sorted} due to missing single-gene data. Skipping.")

if not test_pairs_for_prediction:
    print("\nWARNING: No final predictions could be made.")
else:
    X_test = np.array(X_test_list)
    # Use the FINAL model for this prediction
    predicted_combined_deltas = final_model.predict(X_test)
    predicted_absolute_expressions = predicted_combined_deltas + pseudo_control_profile

    # --- Step 6: Format and Output Final Predictions ---

    # --- NEW: Enforce non-negative expression constraint ---
    # finds all elements in the predicted_absolute_expressions array that are less than zero and sets their value to 0
    predicted_absolute_expressions[predicted_absolute_expressions < 0] = 0

    predicted_column_names = [f"Predicted_{g1}+{g2}" for g1, g2 in test_pairs_for_prediction]
    predicted_df_wide = pd.DataFrame(predicted_absolute_expressions.T,
                                    index=df_raw_data.index,
                                    columns=predicted_column_names)
    predicted_df_wide.reset_index(inplace=True)
    predicted_df_wide.rename(columns={predicted_df_wide.columns[0]: 'gene'}, inplace=True)
    
    print("\nConverting final prediction results to long format...")
    long_format_df = pd.melt(predicted_df_wide,
                             id_vars=['gene'],
                             var_name='perturbation',
                             value_name='expression')
    long_format_df['perturbation'] = long_format_df['perturbation'].str.replace('Predicted_', '', regex=False)
    
    print("\n--- Final Output Sample ---")
    print(long_format_df.head())
    OUTPUT_FILE_PATH = 'prediction.csv'
    long_format_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nFinal predictions saved to: {OUTPUT_FILE_PATH}")