try:
    import sklearn
    import tensorflow as tf
    from tensorflow import keras
    print('sklearn version:', sklearn.__version__)
    print('tensorflow version:', tf.__version__)
    if hasattr(keras, '__version__'):
        print('keras version:', keras.__version__)
    else:
        print('keras module is available')
except Exception as e:
    print('Import failed:', repr(e))
