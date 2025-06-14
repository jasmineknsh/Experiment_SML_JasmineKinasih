import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def handle_outliers_iqr(df, columns):
    """
    Menangani outlier pada kolom numerik dengan metode capping (IQR).
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

def main():
    input_file = "../Graduate_Admission2_raw.csv"
    output_dir = "../preprocessing/Graduate_Admission2_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Membaca data...")
    df = pd.read_csv(input_file)
    df = df.dropna()

    print("🔧 Rename kolom...")
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
    df.drop(columns=["Serial_No"], inplace=True)

    print("📉 Menangani outlier dengan IQR...")
    columns_to_handle = ['LOR', 'CGPA']
    df = handle_outliers_iqr(df, columns_to_handle)

    print("📊 Membagi data...")
    target_column = 'Chance_of_Admit'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    print("💾 Menyimpan hasil split...")
    train_df.to_csv(os.path.join(output_dir, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_clean.csv"), index=False)

    print("✅ Preprocessing selesai!")

main()