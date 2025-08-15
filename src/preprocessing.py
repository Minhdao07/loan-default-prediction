import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Identify numerical & categorical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # Fill missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    return df


