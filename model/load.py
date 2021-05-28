from tensorflow.keras.models import model_from_json

def init(json_path, h5_path):
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(h5_path)
    print('Model has Loaded')

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model
