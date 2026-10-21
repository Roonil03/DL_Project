import os
import hashlib
import pandas as pd

DATA_DIR = "data"

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def audit_dataset():
    manifest_records = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found. Please create it and place your dataset files inside.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.parquet'))]
    
    if not files:
        print(f"No CSV or parquet files found in '{DATA_DIR}/'.")
        return

    print("Auditing dataset files...")
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        file_size = os.path.getsize(filepath)
        file_hash = compute_sha256(filepath)
        
        # Read a sample to inspect structure
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=1000) # Read sample or full depending on size
            full_df = pd.read_csv(filepath, low_memory=False)
        else:
            full_df = pd.read_parquet(filepath)
            
        columns = list(full_df.columns)
        dtypes = {col: str(dtype) for col, dtype in full_df.dtypes.items()}
        missing_values = full_df.isnull().sum().to_dict()
        row_count = len(full_df)
        
        record = {
            "filename": filename,
            "size_bytes": file_size,
            "sha256": file_hash,
            "row_count": row_count,
            "columns": columns,
            "dtypes": dtypes,
            "missing_values": missing_values
        }
        manifest_records.append(record)
        print(f" -> Audited: {filename} ({row_count} rows)")

    # Save manifest summary
    manifest_df = pd.DataFrame(manifest_records)
    output_path = "data/data_manifest.csv"
    manifest_df.to_csv(output_path, index=False)
    print(f"\nManifest successfully generated and saved to '{output_path}'!")

if __name__ == "__main__":
    audit_dataset()