import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_curves(history, model_name, save_dir="results/figures/"):
    """
    Plots and saves training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title(f'Training & Validation Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, target_names, model_name, save_dir="results/figures/"):
    """
    Plots and saves the token-level confusion matrix.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_weights(tokens, attention_weights, gold_labels, pred_labels, example_id, save_dir="results/figures/attention/"):
    """
    Visualizes attention weights over tokens.
    """
    plt.figure(figsize=(14, 2))
    
    # Heatmap expects 2D
    sns.heatmap([attention_weights], cmap='Reds', xticklabels=tokens, yticklabels=False, cbar_kws={'orientation': 'horizontal'})
    
    plt.title(f'Attention Weights - Example {example_id}')
    plt.xlabel('Tokens')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'attention_example_{example_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print out aligned info for the text report
    print(f"\n--- Attention Example {example_id} ---")
    print(f"{'Token':<20} | {'Att Weight':<10} | {'Gold':<15} | {'Pred':<15}")
    print("-" * 65)
    for t, w, g, p in zip(tokens, attention_weights, gold_labels, pred_labels):
        print(f"{t:<20} | {w:<10.4f} | {g:<15} | {p:<15}")
