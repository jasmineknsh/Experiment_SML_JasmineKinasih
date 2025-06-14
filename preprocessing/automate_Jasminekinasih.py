import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def standardize_features(df, features_to_scale, scaler):
 
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.transform(df[features_to_scale])
    return df_scaled, scaler

def main():
    # Path dataset dan model
    input_file = "../Graduate_Admission2_raw.csv" 
    scaler_path = "model/scaler.pkl"
    output_dir = "Graduate_Admission2_preprocessing"
    
    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Loading dataset mentah...")
    df = pd.read_csv(input_file)

    print("🔧 Rename kolom dan hapus kolom tidak relevan...")
    df.rename(columns={
        'Serial No.': 'Serial_No',
        'GRE Score': 'GRE_Score',
        'TOEFL Score': 'TOEFL_Score',
        'University Rating': 'University_Rating',
        'SOP': 'SOP',
        'LOR ': 'LOR',
        'CGPA': 'CGPA',
        'Research': 'Research',
        'Chance of Admit ': 'Chance_of_Admit'
    }, inplace=True)
    df.drop('Serial_No', axis=1, inplace=True)

    print("🔍 Menentukan fitur...")
    features_to_scale = ['GRE_Score', 'TOEFL_Score', 'CGPA', 'SOP', 'LOR']
    other_features = ['University_Rating', 'Research']
    target_column = 'Chance_of_Admit'

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print("📊 Membagi data menjadi training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"⚖️ Memuat model scaler dari: {scaler_path}")
    scaler = joblib.load(scaler_path)

    print("🔄 Melakukan standarisasi fitur numerik...")
    X_train_scaled, _ = standardize_features(X_train, features_to_scale, scaler)
    X_test_scaled, _ = standardize_features(X_test, features_to_scale, scaler)

    print("🔗 Menggabungkan kembali fitur dan target...")
    train_df = pd.concat([
        X_train_scaled[features_to_scale + other_features],
        y_train.reset_index(drop=True)
    ], axis=1)

    test_df = pd.concat([
        X_test_scaled[features_to_scale + other_features],
        y_test.reset_index(drop=True)
    ], axis=1)

    print("💾 Menyimpan data hasil preprocessing...")
    train_df.to_csv(os.path.join(output_dir, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_clean.csv"), index=False)

    print("✅ Preprocessing selesai. Data disimpan di folder:", output_dir)

if __name__ == "__main__":
    main()
