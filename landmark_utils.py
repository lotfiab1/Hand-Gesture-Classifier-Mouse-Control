import numpy as np

NUM_LANDMARKS = 21


def normalize_landmarks(landmarks):

    arr = np.asarray(landmarks, dtype=np.float64).reshape(NUM_LANDMARKS, 3)

    wrist = arr[0].copy()
    arr = arr - wrist

    scale = np.linalg.norm(arr, axis=1).max()
    if scale > 1e-9:
        arr = arr / scale

    return arr.reshape(-1)


def normalize_dataframe_columns(df, label_col="label"):

    feature_cols = [c for c in df.columns if c != label_col]
    normalized = df[feature_cols].apply(
        lambda row: normalize_landmarks(row.values), axis=1, result_type="expand"
    )
    normalized.columns = feature_cols
    normalized[label_col] = df[label_col].values
    return normalized
