import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, GlobalAveragePooling1D

def prepare_re_data(normalized_data, tokenizer, max_seq_length, negative_ratio=1.0):
    """
    Prepares input for Relation Extraction with negative sampling.
    Input: Context sequence, Head Entity Start, Tail Entity Start
    (Simplified: Just using the sequence and pooling for a lightweight classifier)
    """
    X_seq = []
    y_rel = []
    
    # Collect unique relation types
    rel_types = set(['NONE']) # 'NONE' is the negative class
    for item in normalized_data:
        for rel in item['relations']:
            rel_types.add(rel['relation'])
            
    rel_to_id = {r: i for i, r in enumerate(sorted(list(rel_types)))}
    id_to_rel = {i: r for r, i in rel_to_id.items()}
    
    for item in normalized_data:
        tokens = item['tokens']
        text = " ".join(tokens)
        seq = tokenizer.texts_to_sequences([text])[0]
        
        # Pad/Truncate
        seq = seq[:max_seq_length]
        seq = seq + [0] * (max_seq_length - len(seq))
        
        entities = item['entities']
        relations = item['relations']
        
        # Positive samples
        pos_pairs = set()
        for rel in relations:
            X_seq.append(seq)
            y_rel.append(rel_to_id[rel['relation']])
            pos_pairs.add((rel['head'], rel['tail']))
            
        # Negative samples
        if len(entities) > 1:
            num_neg_to_sample = int(len(relations) * negative_ratio)
            neg_sampled = 0
            # Brute force random sampling
            attempts = 0
            while neg_sampled < num_neg_to_sample and attempts < 100:
                h = random.choice(entities)['start']
                t = random.choice(entities)['start']
                if h != t and (h, t) not in pos_pairs:
                    X_seq.append(seq)
                    y_rel.append(rel_to_id['NONE'])
                    pos_pairs.add((h, t))
                    neg_sampled += 1
                attempts += 1
                
    return np.array(X_seq), np.array(y_rel), rel_to_id, id_to_rel

def build_re_model(vocab_size, num_relations, max_seq_length, embedding_dim=64):
    """
    Lightweight Relation Classifier using context tokens.
    """
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    
    # Simple pooling over context
    x = GlobalAveragePooling1D()(emb)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_relations, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Relation_Classifier")
    return model

def plot_knowledge_graph(predictions, id_to_rel, entities_map, max_edges=50, save_dir="results/figures/"):
    """
    Constructs and visualizes a knowledge graph from predicted relations using NetworkX.
    entities_map: dict mapping entity ID/index to surface string.
    """
    G = nx.DiGraph()
    
    edges_added = 0
    for head, tail, rel_id in predictions:
        if edges_added >= max_edges:
            break
            
        rel_str = id_to_rel[rel_id]
        if rel_str != 'NONE':
            h_str = entities_map.get(head, f"Ent_{head}")
            t_str = entities_map.get(tail, f"Ent_{tail}")
            
            G.add_edge(h_str, t_str, label=rel_str)
            edges_added += 1
            
    if len(G.edges) == 0:
        print("No valid relations found to plot.")
        return
        
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=1.5)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    plt.title('Harry Potter Knowledge Graph')
    plt.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'harry_potter_relation_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()
