import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=25, is_attention=False):
    """
    Trains a compiled Keras model.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    
    callbacks = [early_stopping, reduce_lr]
    
    # If the model has multiple outputs (like the attention model), we need to handle labels appropriately
    if is_attention:
        # We only train on the first output (the softmax). The second output is attention weights.
        # So we pass y_train only for the first output.
        # Note: compile must have loss={'dense_...': 'sparse_categorical_crossentropy', 'token_self_attention': None}
        pass # Expected to be handled in fit if compiled correctly
        
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    return history, training_time
