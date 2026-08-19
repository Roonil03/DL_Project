import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, SimpleRNN, LSTM, Bidirectional

class TokenSelfAttention(tf.keras.layers.Layer):
    """
    A lightweight custom self-attention/token-context layer that produces 
    contextual token representations while preserving the sequence dimension.
    """
    def __init__(self, **kwargs):
        super(TokenSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(TokenSelfAttention, self).build(input_shape)

    def call(self, x, mask=None):
        # x is (batch_size, seq_len, hidden_dim)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b) # (batch, seq_len, 1)
        # Squeeze the last dimension
        e = tf.squeeze(e, axis=-1) # (batch, seq_len)
        
        # Apply mask to avoid attending to padding tokens
        if mask is not None:
            # Mask should be (batch, seq_len)
            mask = tf.cast(mask, dtype=e.dtype)
            # Apply a large negative value to masked positions before softmax
            e = e - (1.0 - mask) * 1e9
            
        alpha = tf.keras.backend.softmax(e) # (batch, seq_len)
        alpha = tf.expand_dims(alpha, axis=-1) # (batch, seq_len, 1)
        
        # We need to preserve the sequence dimension, so we'll multiply the weights by the sequence elements
        # Instead of a global context vector, we return the attention-weighted sequence or a combination
        # For sequence-to-sequence preservation, we can return the element-wise multiplication
        # Another option for self-attention is multi-head attention, but we will keep this simple and interpretable
        
        context_aware_x = x * alpha # (batch, seq_len, hidden_dim)
        
        # Returning both the context-aware representations and the attention weights for visualization
        return context_aware_x, alpha
        
    def compute_mask(self, input, input_mask=None):
        # Do not pass the mask forward to subsequent layers, let them handle their own
        return input_mask

def build_mlp_model(vocab_size, num_labels, max_seq_length, embedding_dim=64):
    """
    Model 1: Feed-forward baseline.
    Token representation -> Embedding -> Dense -> Dropout -> Dense -> Softmax per token.
    """
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    # Mask zero is crucial for ignoring padding
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_labels, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="MLP_Token_Classifier")
    return model

def build_simplernn_model(vocab_size, num_labels, max_seq_length, embedding_dim=64, hidden_dim=64):
    """
    Model 2: Simple RNN
    Token IDs -> Embedding -> SimpleRNN -> Dropout -> Dense -> Softmax
    """
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = SimpleRNN(hidden_dim, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_labels, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Simple_RNN")
    return model

def build_bilstm_model(vocab_size, num_labels, max_seq_length, embedding_dim=64, hidden_dim=64):
    """
    Model 3: Bidirectional LSTM
    Token IDs -> Embedding -> Bidirectional LSTM -> Dropout -> Dense -> Softmax
    """
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    x = Bidirectional(LSTM(hidden_dim, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_labels, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM")
    return model

def build_bilstm_attention_model(vocab_size, num_labels, max_seq_length, embedding_dim=64, hidden_dim=64):
    """
    Model 4: Bidirectional LSTM + Attention
    Token IDs -> Embedding -> Bidirectional LSTM -> Attention mechanism -> Dense -> Softmax
    Note: The attention mechanism must preserve the sequence dimension.
    """
    inputs = Input(shape=(max_seq_length,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_dim, return_sequences=True))(emb)
    
    # Custom Attention Layer that returns sequence-aligned outputs and attention weights
    att_out, att_weights = TokenSelfAttention()(lstm_out)
    
    # Combine LSTM out and Attention out for rich representation
    combined = tf.keras.layers.Concatenate()([lstm_out, att_out])
    x = Dropout(0.3)(combined)
    outputs = Dense(num_labels, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=[outputs, att_weights], name="BiLSTM_Attention")
    return model

def get_model_summary_data(model):
    """
    Returns trainable, non-trainable, and total parameter counts for logging.
    """
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return {
        "Trainable Parameters": int(trainable_count),
        "Non-Trainable Parameters": int(non_trainable_count),
        "Total Parameters": int(trainable_count + non_trainable_count)
    }
