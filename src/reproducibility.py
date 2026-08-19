import os
import random
import numpy as np
import tensorflow as tf

def set_all_seeds(seed=42):
    """
    Sets seeds for Python random, NumPy, and TensorFlow to ensure reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_device_info():
    """
    Detects and returns CPU/GPU availability.
    """
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    return {
        "GPUs": len(gpus),
        "GPU_Details": gpus,
        "CPUs": len(cpus),
        "CPU_Details": cpus
    }

def print_environment_info():
    """
    Prints version info for key packages and device info.
    """
    import pandas as pd
    import sklearn
    print(f"Python versions and environments can affect bit-level determinism.")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    device_info = get_device_info()
    print(f"Available CPUs: {device_info['CPUs']}")
    print(f"Available GPUs: {device_info['GPUs']}")
    if device_info['GPUs'] > 0:
        print("GPU will be automatically used by TensorFlow.")
