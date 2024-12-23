import numpy as np
from sklearn.preprocessing import LabelEncoder

# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score


def get_labels(df):
    y = np.array(df['group'].values)
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(y)
    print(integer_labels[:5])
    return integer_labels