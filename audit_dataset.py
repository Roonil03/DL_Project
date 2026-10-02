{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "afd4325d-1713-4872-9d58-1b9186e9b526",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--- DATASET AUDIT REPORT ---\n",
      "File Path: data/global_indian_markets.csv\n",
      "Total Records: 4,171\n",
      "Total Columns: 7\n",
      "Columns: ['Date', 'NIFTY50', 'SP500', 'USDINR', 'GOLD', 'BRENT', 'US10Y']\n",
      "Data Types:\n",
      "Date        object\n",
      "NIFTY50    float64\n",
      "SP500      float64\n",
      "USDINR     float64\n",
      "GOLD       float64\n",
      "BRENT      float64\n",
      "US10Y      float64\n",
      "dtype: object\n",
      "Missing Values:\n",
      "Series([], dtype: int64)\n",
      "Duplicate Rows: 0\n",
      "Dataset SHA-256 Hash: dd4fae13270fed0ab9379149e672b4b35e28059a126d8a81fccf5a64ca7c763b\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import hashlib\n",
    "\n",
    "def audit_data():\n",
    "    path = \"data/global_indian_markets.csv\"\n",
    "    if not pd.io.common.file_exists(path):\n",
    "        print(f\"Dataset not found at {path}. Please place the Kaggle CSV there.\")\n",
    "        return\n",
    "\n",
    "    df = pd.read_csv(path)\n",
    "    print(\"--- DATASET AUDIT REPORT ---\")\n",
    "    print(f\"File Path: {path}\")\n",
    "    print(f\"Total Records: {len(df):,}\")\n",
    "    print(f\"Total Columns: {len(df.columns)}\")\n",
    "    print(f\"Columns: {list(df.columns)}\")\n",
    "    print(f\"Data Types:\\n{df.dtypes}\")\n",
    "    print(f\"Missing Values:\\n{df.isnull().sum()[df.isnull().sum() > 0]}\")\n",
    "    print(f\"Duplicate Rows: {df.duplicated().sum()}\")\n",
    "    \n",
    "    # Generate SHA-256 Hash for reproducibility\n",
    "    with open(path, \"rb\") as f:\n",
    "        file_hash = hashlib.sha256(f.read()).hexdigest()\n",
    "    print(f\"Dataset SHA-256 Hash: {file_hash}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    audit_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02426a0a-8968-48b5-ba9c-335314ecd6f1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
