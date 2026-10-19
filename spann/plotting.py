import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_standard_confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)

    ax.set_title('Standard Confusion Matrix', pad=20, fontweight='bold')
    ax.set_xlabel('Predicted Label', labelpad=10)
    ax.set_ylabel('Ground Truth', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_novel_discovery_confusion_matrix(y_true, y_pred, known_types):
    all_true_labels = np.unique(y_true)
    novel_types = sorted([t for t in all_true_labels if t not in known_types])

    display_labels_true = list(known_types) + novel_types
    display_labels_pred = list(known_types) + ['Unknown']

    cm_df = pd.crosstab(
        pd.Series(y_true, name='Ground Truth'),
        pd.Series(y_pred, name='Prediction')
    )
    cm_df = cm_df.reindex(index=display_labels_true, columns=display_labels_pred, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar=False, ax=ax)

    ax.set_title('Confusion Matrix: Known Labels vs Novel Discovery', pad=20, fontweight='bold')
    ax.set_xlabel('Prediction', labelpad=10)
    ax.set_ylabel('Ground Truth', labelpad=10)

    ax.axhline(y=len(known_types), color='red', linestyle='--', linewidth=2)
    
    ax.text(len(display_labels_pred) + 0.5, len(known_types) / 2, 'KNOWN CELLS',
             color='red', rotation=270, va='center', fontweight='bold')
    ax.text(len(display_labels_pred) + 0.5, len(known_types) + len(novel_types) / 2, 'NOVEL CELLS',
             color='red', rotation=270, va='center', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_celltype_distribution(adata_dict, label_col='cell_type'):
    num_plots = len(adata_dict)
    if num_plots == 0:
        return
        
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    for ax, (name, adata) in zip(axes, adata_dict.items()):
        if label_col not in adata.obs.columns:
            print(f"Warning: Label column '{label_col}' not found in {name}.")
            continue
            
        labels = adata.obs[label_col].astype(str).values
        metrics = stats.calculate_imbalance_metrics(labels)
        
        counts = adata.obs[label_col].value_counts().reset_index()
        counts.columns = ['Cell Type', 'Count']
        
        local_order = counts['Cell Type'].tolist()
        
        sns.barplot(
            data=counts, 
            x='Cell Type', 
            y='Count', 
            order=local_order,
            palette='muted',
            ax=ax
        )
        
        title_str = f"{name}\nIR: {metrics['IR']:.2f} | Gini: {metrics['Gini']:.4f}"
        ax.set_title(title_str, pad=15, fontweight='bold')
        ax.set_xlabel('Cell Type', labelpad=10)
        ax.set_ylabel('Number of Cells', labelpad=10)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        sns.despine(ax=ax) 
        
    plt.tight_layout()
    plt.show()