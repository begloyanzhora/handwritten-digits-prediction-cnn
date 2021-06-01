import re
import os
import base64
import numpy as np
from flask import Flask, render_template, request
from model.load import *
from PIL import Image

model = init('model/model.h5')

app = Flask(__name__)

def convert_image(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    print('hello')
    img_data = request.get_data()
    convert_image(img_data)
    x = Image.open('output.png')
    x = x.resize((28,28))
    x = np.array(x)[:,:,0]
    x = np.invert(x)
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255

    out = model.predict(x)
    response = np.array_str(np.argmax(out, axis=-1))

    return response

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    app.debug = True
