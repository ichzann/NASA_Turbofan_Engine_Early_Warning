import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pca_plot(df):
    features = df[df.columns[2:]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_cor = pca.fit_transform(features_scaled)
    df_pca_cor = pd.DataFrame(data=pca_cor, columns=["PCA1", "PCA2"], index=features.index)

    df_merged = pd.concat([df, df_pca_cor], axis=1)

    min_train = df_merged.groupby(0)[["PCA1", "PCA2"]].min()
    max_train = df_merged.groupby(0)[["PCA1", "PCA2"]].max()

    plt.figure(figsize=(12, 6))

    # 1. Invisible dots to automatically size the graph correctly
    plt.scatter(min_train["PCA1"], min_train["PCA2"], alpha=0)
    plt.scatter(max_train["PCA1"], max_train["PCA2"], alpha=0)

    # 2. Print Green numbers (Min)
    for engine_id, row in min_train.iterrows():
        plt.text(row["PCA1"], row["PCA2"], str(engine_id), color='green', fontsize=6)

    # 3. Print Red numbers (Max)
    for engine_id, row in max_train.iterrows():
        plt.text(row["PCA1"], row["PCA2"], str(engine_id), color='red', fontsize=6)

    return plt.show()

def cal_PCA_train(df):
    if "PCA1" not in df.columns or "PCA2" not in df.columns:
        features = df[df.columns[2:]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=2)

        PCA_cor = pca.fit_transform(scaled_features)
        df_pca_cor = pd.DataFrame(data= PCA_cor, columns=["PCA1", "PCA2"], index=features.index)

        df = pd.concat([df,df_pca_cor ], axis=1)
        return df, scaler, pca
    else:
        return "already there"



def cal_PCA_test(df, fitted_scaler, fitted_pca):
    if "PCA1" not in df.columns or "PCA2" not in df.columns:
        features = df[df.columns[2:]]


        scaled_features = np.nan_to_num(fitted_scaler.transform(features))
        pca_cor = fitted_pca.transform(scaled_features)


        df_pca_cor = pd.DataFrame(data=pca_cor, columns=["PCA1", "PCA2"], index=df.index)
        df = pd.concat([df, df_pca_cor], axis=1)
    
        return df
    else:
        return "already there"
    
def add_labels(df):
    max_cycles = df.groupby(0)[1].transform('max')
    df['rul'] = max_cycles - df[1]
    
    df['fail_5'] = np.where(df['rul'] <= 50, 1, 0)
    return df

def sliding_window(df, sequence_length, feature_cols, pca_horizon=10):
    X = []
    y_rul = []
    y_fail = []
    y_pca = []

    pad_size = sequence_length - 1
    for eingine_id in df["0"].unique():

        engine_data = df[df["0"] == eingine_id]

        engine_features = engine_data[feature_cols].values
        engine_target_rul = engine_data["rul"].values
        engine_target_fail = engine_data["fail_5"].values
        engine_pca = engine_data[["PCA1", "PCA2"]].values
        n = len(engine_data)

        first_row_features = engine_features[0]

        padding_features = np.tile(first_row_features, (pad_size, 1))
        padded_features = np.vstack([padding_features, engine_features])

        for i in range(n):

            window_x = padded_features[i : i + sequence_length]

            window_y_rul = engine_target_rul[i]
            window_y_fail = engine_target_fail[i]

            future_idx = min(i + pca_horizon, n - 1)
            window_y_pca = engine_pca[future_idx]

            X.append(window_x)
            y_rul.append(window_y_rul)
            y_fail.append(window_y_fail)
            y_pca.append(window_y_pca)

    return np.array(X), np.array(y_rul), np.array(y_fail), np.array(y_pca)




