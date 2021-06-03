from tensorflow.keras.models import load_model

def init(h5_path):
    model = load_model(h5_path)
    print('Model has Loaded')

    return model
