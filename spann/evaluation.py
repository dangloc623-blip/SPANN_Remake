import numpy as np
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

def evaluate_standard_classification(y_true, y_pred):
    """
    Calculate standard metrics for fully annotated datasets (All Known).
    """
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return {
        "ACC": acc,
        "NMI": nmi,
        "ARI": ari
    }

def evaluate_novel_discovery(y_true, y_pred, known_types, unknown_label='Unknown'):
    """
    Calculate metrics for novel cell type discovery (Open-set classification).
    Computes K-ACC, U-ACC, T-ACC, and the Harmonic Mean (H-score).
    """
    is_known = np.isin(y_true, known_types)
    is_novel = ~is_known

    # K_ACC (Known Cell Annotation Accuracy)
    known_idx = np.where(is_known)[0]
    if len(known_idx) > 0:
        k_acc = np.sum(y_pred[known_idx] == y_true[known_idx]) / len(known_idx)
    else:
        k_acc = 0.0

    # U_ACC (Novel Cell Discovery Rate)
    novel_idx = np.where(is_novel)[0]
    if len(novel_idx) > 0:
        u_acc = np.sum(y_pred[novel_idx] == unknown_label) / len(novel_idx)
    else:
        u_acc = 0.0

    # T_ACC (Total Accuracy)
    total_correct = np.sum(y_pred[known_idx] == y_true[known_idx]) + np.sum(y_pred[novel_idx] == unknown_label)
    t_acc = total_correct / len(y_true) if len(y_true) > 0 else 0.0

    # H-SCORE (Harmonic Mean)
    if (k_acc + u_acc) > 0:
        h_score = 2 * (k_acc * u_acc) / (k_acc + u_acc)
    else:
        h_score = 0.0

    return {
        "K_ACC": k_acc,
        "U_ACC": u_acc,
        "T_ACC": t_acc,
        "H_SCORE": h_score,
        "n_known": len(known_idx),
        "n_novel": len(novel_idx)
    }