def span_to_bio(tokens, entities):
    """
    Converts span annotations into BIO token labels.
    """
    bio_tags = ['O'] * len(tokens)
    
    # Sort entities by start index to handle logically
    entities = sorted(entities, key=lambda x: x['start'])
    
    conflict_count = 0
    for ent in entities:
        start = ent['start']
        end = ent['end']
        label = ent['label']
        
        # Bounds check
        if start < 0 or end > len(tokens) or start >= end:
            continue
            
        # Check for overlaps
        overlap = False
        for i in range(start, end):
            if bio_tags[i] != 'O':
                overlap = True
                break
                
        if overlap:
            conflict_count += 1
            # Simple policy: skip overlapping entity (first come, first served based on sort)
            continue
            
        # Apply BIO tags
        bio_tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            bio_tags[i] = f"I-{label}"
            
    return bio_tags, conflict_count

def convert_dataset_to_bio(normalized_data):
    """
    Applies BIO conversion to the entire dataset.
    """
    total_conflicts = 0
    bio_dataset = []
    
    for item in normalized_data:
        bio_tags, conflicts = span_to_bio(item['tokens'], item['entities'])
        total_conflicts += conflicts
        item['bio_tags'] = bio_tags
        bio_dataset.append(item)
        
    print(f"Total overlapping entities detected and skipped: {total_conflicts}")
    return bio_dataset

def build_label_vocab(bio_dataset):
    """
    Creates a label mapping dynamically from the dataset.
    """
    unique_labels = set(['O'])
    for item in bio_dataset:
        for tag in item['bio_tags']:
            unique_labels.add(tag)
            
    # Sort to ensure reproducibility
    unique_labels = sorted(list(unique_labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    return label_to_id, id_to_label
