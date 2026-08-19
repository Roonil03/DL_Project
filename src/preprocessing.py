import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def split_dataset(dataset, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits the dataset at the document/paragraph level.
    """
    # First split off the test set
    train_val, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
    
    # Calculate the proportion of the validation set relative to the remaining training data
    val_prop = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_prop, random_state=random_state)
    
    return train, val, test

def prepare_sequences(train_data, val_data, test_data, max_seq_length=None):
    """
    Fits tokenizer on training data only and prepares padded sequences.
    """
    train_texts = [" ".join(item['tokens']) for item in train_data]
    val_texts = [" ".join(item['tokens']) for item in val_data]
    test_texts = [" ".join(item['tokens']) for item in test_data]
    
    # Fit tokenizer on training data
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    # Transform to sequences
    train_seq = tokenizer.texts_to_sequences(train_texts)
    val_seq = tokenizer.texts_to_sequences(val_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)
    
    if max_seq_length is None:
        # Determine from 95th percentile of train data
        lengths = [len(seq) for seq in train_seq]
        max_seq_length = int(np.percentile(lengths, 95))
        print(f"Calculated 95th percentile max_seq_length: {max_seq_length}")
        
    # Pad sequences
    X_train = pad_sequences(train_seq, maxlen=max_seq_length, padding='post', truncating='post')
    X_val = pad_sequences(val_seq, maxlen=max_seq_length, padding='post', truncating='post')
    X_test = pad_sequences(test_seq, maxlen=max_seq_length, padding='post', truncating='post')
    
    return X_train, X_val, X_test, tokenizer, max_seq_length

def encode_labels(data, label_to_id, max_seq_length):
    """
    Encodes and pads BIO labels for sequence modeling.
    """
    encoded = []
    for item in data:
        seq = [label_to_id[tag] for tag in item['bio_tags']]
        # Truncate
        seq = seq[:max_seq_length]
        # Pad with O label (usually mapped to an ID, e.g., 0)
        pad_id = label_to_id['O']
        seq = seq + [pad_id] * (max_seq_length - len(seq))
        encoded.append(seq)
    
    return np.array(encoded)
