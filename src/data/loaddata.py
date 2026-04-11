import glob
import os.path
import numpy as np
from src.data.extractFeture import preprocess_and_extract_features_mne_with_timestamps
from src.data.extractTarget import extractTarget

def extract_data_and_labels(edf_file_path, summary_file_path):

    # Extract features
    X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path)
    # Extract labels
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    y = np.array([1 if seizure_start_time <= row[0] <= seizure_end_time else 0 for row in X])

    # Remove the first column (timestamp) from X
    X = X[:,1:]
    return X,y


def load_data(subject_id,base_path):
    """
    Load data for a given subject.
    Reads all EDF files for the given CHB subject and extracts features from each file.
    Returns a list of all data arrays and a list of all label arrays.
    Each data array has shape (n_samples, n_features), each label array has shape (n_samples,).
    """
    edf_file_path = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
    summary_file_path = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
    all_X = []
    all_y = []
    for edf_file_path in edf_file_path:
        X, y = extract_data_and_labels(edf_file_path, summary_file_path)
        all_X.append(X)
        all_y.append(y)
    return all_X,all_y

# Usage:
# subject_id = 1
# base_path = "data"
# all_X, all_y = load_data(subject_id, base_path)

# Count positive/negative labels in all_y and print
# total_n_count = 0
# total_p_count = 0
# for y in all_y:
#     p_count = 0
#     n_count = 0
#     for label in y:
#         if label == 1:
#             p_count += 1
#         else:
#             n_count += 1
#     total_n_count += n_count
#     total_p_count += p_count
# print("total_p_count/total_count:", total_p_count/(total_n_count+total_p_count))

## total_p_count/total_count: 0.018808777429467086