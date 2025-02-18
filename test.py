import pickle

preprocessor_path = "artifacts/preprocessor.pkl"

# Load preprocessor
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

print("Expected Columns:", preprocessor.feature_names_in_)
