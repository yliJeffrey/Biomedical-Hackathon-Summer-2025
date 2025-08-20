import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# --- Helper Function to Parse Column Names ---
def parse_perturbation_name(col_name):
    """
    Parses a column name like 'g0261+g0760' or 'g0495+ctrl'.
    Returns a tuple: (type, gene1, gene2 or None).
    """
    genes = col_name.split('+')
    if len(genes) == 2:
        if 'ctrl' in genes:
            gene1 = genes[0] if genes[1] == 'ctrl' else genes[1]
            return ('single', gene1, None)
        else:
            # genes.sort()
            return ('double', genes[0], genes[1])
    return ('unknown', None, None)

# ==============================================================================
# --- MAIN SCRIPT ---
# ==============================================================================

# --- Step 1: User Inputs ---
# ------------------------------------------------------------------------------
# 1.1: DEFINE THE PATH TO YOUR DATA FILE
DATA_FILE_PATH = 'data/train_set.csv'

# 1.2: DEFINE THE DOUBLE-GENE PAIRS TO PREDICT (YOUR TEST SET)
test_double_pairs_tuples = [
    ('g0037', 'g0083')
]
# ------------------------------------------------------------------------------


# --- Step 2: Load and Preprocess Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
df_raw_data = pd.read_csv(DATA_FILE_PATH, index_col=0)
print(f"Data loaded successfully. Shape: {df_raw_data.shape}")

# Get the list of unique perturbation names from the columns
unique_perturbation_names = df_raw_data.columns.unique()

# Average the replicates for each unique name
print("Averaging replicates for each unique condition...")
avg_expression_profiles_df = pd.DataFrame(index=df_raw_data.index)
for name in unique_perturbation_names:
    replicate_data = df_raw_data[name]
    
    # Check if replicate_data is a DataFrame (has replicates) or a Series (no replicates)
    if isinstance(replicate_data, pd.DataFrame):
        # If it's a DataFrame, average across the columns (replicates)
        avg_expression_profiles_df[name] = replicate_data.mean(axis=1)
    else:
        # If it's a Series, there's nothing to average. It is the profile.
        avg_expression_profiles_df[name] = replicate_data
        
print(f"Averaged {len(avg_expression_profiles_df.columns)} unique perturbation profiles.")

# --- NEW: Create a "Pseudo-Control" from the Median Profile ---
print("\nNo dedicated control column found. Creating a 'pseudo-control' profile.")
pseudo_control_profile = avg_expression_profiles_df.median(axis=1).values
print("Pseudo-control profile calculated using the median of all averaged profiles.")

# Convert the averaged DataFrame to a dictionary of numpy arrays for faster access
avg_expression_profiles = {name: avg_expression_profiles_df[name].values for name in avg_expression_profiles_df.columns}

# Calculate 'Delta' (deviation from the pseudo-control)
delta_profiles = {name: profile - pseudo_control_profile for name, profile in avg_expression_profiles.items()}
print("Delta profiles calculated relative to the pseudo-control.")

# --- Step 3: Prepare Training Data (X_train, Y_train) ---
X_train_list = []
Y_train_list = []

print("\nBuilding training dataset from measured double-gene perturbations...")
for perturbation_name in unique_perturbation_names:
    ptype, gene1, gene2 = parse_perturbation_name(perturbation_name)
    
    if ptype == 'double':
        single_g1_name = f"{gene1}+ctrl"
        single_g2_name = f"{gene2}+ctrl"
        double_pair_name = f"{gene1}+{gene2}"
        
        if single_g1_name in delta_profiles and single_g2_name in delta_profiles:
            X_train_list.append(np.concatenate((delta_profiles[single_g1_name], delta_profiles[single_g2_name])))
            Y_train_list.append(delta_profiles[double_pair_name])
        else:
            print(f"Warning: Missing single-gene data for training pair {double_pair_name}. Skipping.")

X_train = np.array(X_train_list)
Y_train = np.array(Y_train_list)

if len(X_train) == 0:
    raise ValueError("Training data could not be constructed. Check your column names and data.")
print(f"Training data constructed. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")


# --- Step 4: Train the Multi-Output Regression Model ---
model = MultiOutputRegressor(LinearRegression())
print("\nTraining the model...")
model.fit(X_train, Y_train)
print("Model training complete.")


# --- Step 5: Prepare Test Data and Make Predictions ---
X_test_list = []
test_pairs_for_prediction = []

print("\nBuilding test dataset for prediction...")
for gene1, gene2 in test_double_pairs_tuples:
    g1_sorted, g2_sorted = sorted((gene1, gene2))
    
    single_g1_name = f"{g1_sorted}+ctrl"
    single_g2_name = f"{g2_sorted}+ctrl"
    
    if single_g1_name in delta_profiles and single_g2_name in delta_profiles:
        X_test_list.append(np.concatenate((delta_profiles[single_g1_name], delta_profiles[single_g2_name])))
        test_pairs_for_prediction.append((g1_sorted, g2_sorted))
    else:
        print(f"Error: Cannot predict for pair {g1_sorted}+{g2_sorted} due to missing single-gene data. This pair will be skipped.")

if not test_pairs_for_prediction:
    print("\nWARNING: No predictions could be made. Check if the single-gene profiles for your test pairs exist in the data.")
else:
    X_test = np.array(X_test_list)
    print(f"Test data constructed. X_test shape: {X_test.shape}")
    
    predicted_combined_deltas = model.predict(X_test)
    print(f"Prediction complete. Predicted combined deltas shape: {predicted_combined_deltas.shape}")

    # --- Step 6: Format and Output Predictions ---
    # The final prediction is a delta. To convert it back to an "absolute" expression level
    # that is comparable to your input data, we add back the pseudo-control profile.
    predicted_absolute_expressions = predicted_combined_deltas + pseudo_control_profile
    
    predicted_column_names = [f"Predicted_{g1}+{g2}" for g1, g2 in test_pairs_for_prediction]
    predicted_df = pd.DataFrame(predicted_absolute_expressions.T,
                                index=df_raw_data.index,
                                columns=predicted_column_names)

    print("\n--- Prediction Results (in absolute expression scale) ---")
    print("Sample of the predicted DataFrame:")
    print(predicted_df.head())

    OUTPUT_FILE_PATH = 'hackathon_predictions.csv'
    predicted_df.to_csv(OUTPUT_FILE_PATH)
    print(f"\nPredictions saved to: {OUTPUT_FILE_PATH}")