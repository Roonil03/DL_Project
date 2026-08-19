import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

def flatten_predictions(y_true, y_pred, pad_id):
    """
    Flattens sequence labels to token labels, removing padding tokens.
    """
    true_flat = []
    pred_flat = []
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            if y_true[i][j] != pad_id:
                true_flat.append(y_true[i][j])
                pred_flat.append(y_pred[i][j])
    return np.array(true_flat), np.array(pred_flat)

def evaluate_token_level(y_true, y_pred, id_to_label, pad_id):
    """
    Evaluates at the token level using scikit-learn.
    """
    true_flat, pred_flat = flatten_predictions(y_true, y_pred, pad_id)
    
    labels = [i for i in id_to_label.keys() if i != pad_id]
    target_names = [id_to_label[i] for i in labels]
    
    acc = np.mean(true_flat == pred_flat)
    macro_p = precision_score(true_flat, pred_flat, labels=labels, average='macro', zero_division=0)
    macro_r = recall_score(true_flat, pred_flat, labels=labels, average='macro', zero_division=0)
    macro_f1 = f1_score(true_flat, pred_flat, labels=labels, average='macro', zero_division=0)
    
    report = classification_report(true_flat, pred_flat, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    
    return {
        "Accuracy": acc,
        "Macro Precision": macro_p,
        "Macro Recall": macro_r,
        "Macro F1": macro_f1,
        "Report": report,
        "True Flat": true_flat,
        "Pred Flat": pred_flat
    }

def extract_entities_from_bio(seq_labels, id_to_label):
    """
    Extracts exact entities (type, start, end) from a BIO sequence.
    """
    entities = []
    current_ent = None
    
    for i, label_id in enumerate(seq_labels):
        label = id_to_label[label_id]
        if label.startswith('B-'):
            if current_ent:
                entities.append(current_ent)
            current_ent = {'type': label[2:], 'start': i, 'end': i + 1}
        elif label.startswith('I-'):
            if current_ent and current_ent['type'] == label[2:]:
                current_ent['end'] = i + 1
            else:
                # Invalid I- without B- or type mismatch. Treat as O or new B-. Policy: ignore/end
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None
        else: # O or PAD
            if current_ent:
                entities.append(current_ent)
                current_ent = None
                
    if current_ent:
        entities.append(current_ent)
        
    return entities

def evaluate_entity_level(y_true, y_pred, id_to_label, pad_id):
    """
    Evaluates exact entity matches.
    """
    exact_matches = 0
    true_entities_count = 0
    pred_entities_count = 0
    
    type_metrics = {}
    
    for i in range(len(y_true)):
        true_seq = [t for t in y_true[i] if t != pad_id]
        pred_seq = y_pred[i][:len(true_seq)]
        
        true_ents = extract_entities_from_bio(true_seq, id_to_label)
        pred_ents = extract_entities_from_bio(pred_seq, id_to_label)
        
        true_entities_count += len(true_ents)
        pred_entities_count += len(pred_ents)
        
        for p_ent in pred_ents:
            ent_type = p_ent['type']
            if ent_type not in type_metrics:
                type_metrics[ent_type] = {'tp': 0, 'fp': 0, 'fn': 0}
            
            if p_ent in true_ents:
                exact_matches += 1
                type_metrics[ent_type]['tp'] += 1
            else:
                type_metrics[ent_type]['fp'] += 1
                
        for t_ent in true_ents:
            ent_type = t_ent['type']
            if ent_type not in type_metrics:
                type_metrics[ent_type] = {'tp': 0, 'fp': 0, 'fn': 0}
            if t_ent not in pred_ents:
                type_metrics[ent_type]['fn'] += 1
                
    precision = exact_matches / pred_entities_count if pred_entities_count > 0 else 0
    recall = exact_matches / true_entities_count if true_entities_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    for t, m in type_metrics.items():
        p = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        m['precision'] = p
        m['recall'] = r
        m['f1'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
    return {
        "Exact Entity Precision": precision,
        "Exact Entity Recall": recall,
        "Exact Entity F1": f1,
        "Type Metrics": type_metrics
    }
