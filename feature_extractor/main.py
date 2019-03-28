"""
PYTHONPATH=. python youtube-8m/feature_extractor/main.py
"""

from pooling_feature import pooling_video, pooling_dataset, pad_if_need
from feature_extractor import *
import pandas as pd
import cv2
import numpy as np
import os
import glob
from tqdm import *
import pickle


def extract_features(paths):
    extractor = YouTube8MFeatureExtractor()
    features = []
    for path in tqdm(paths, ncols=10):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature = extractor.extract_rgb_frame_features(image)
        features.append(feature)

    features = np.asarray(features)
    return features


def main():
    os.makedirs("./temporal_features", exist_ok=True)
    train_df = pd.read_csv("/media/ngxbac/Bac/competition/emotiw_temporal/csv/frames/train.csv.gz", nrows=None)
    valid_df = pd.read_csv("/media/ngxbac/Bac/competition/emotiw_temporal/csv/frames/valid.csv.gz", nrows=None)

    train_features = extract_features(train_df['path'].values)
    valid_features = extract_features(valid_df['path'].values)
    
    assert train_df.shape[0] == train_features.shape[0]
    assert valid_df.shape[0] == valid_features.shape[0]

    with open("./temporal_features/train_context_raw.pkl", 'wb') as f:
        pickle.dump(train_features, f)

    with open("./temporal_features/valid_context_raw.pkl", 'wb') as f:
        pickle.dump(valid_features, f)

    train_video_features = pooling_dataset(train_df, train_features, stride=2, pool_out=8)
    valid_video_features = pooling_dataset(valid_df, valid_features, stride=2, pool_out=8)

    with open("./temporal_features/train_context.pkl", 'wb') as f:
        pickle.dump(train_video_features, f)

    with open("./temporal_features/valid_context.pkl", 'wb') as f:
        pickle.dump(valid_video_features, f)


if __name__ == '__main__':
    main()