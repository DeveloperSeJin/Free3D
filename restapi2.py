import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import time
import Diffuse

Prompt = ""

app = Flask(__name__)
CORS(app)
@app.route('/call', methods= ['GET', 'POST'])
def call():
    if request.method == 'POST':
        get_type = request.args.get("type")
        if get_type == "Text" :
            
            return "Return text"
        elif get_type == "Image" :
            Image = request.files['Image']
            Image.save('./image')
            return send_file('./image', mimetype='image/jpeg')
        
        elif get_type == "Diffusion":
            Prompt = request.get_text()
            Image = Diffuse.run(Prompt)
            Image.save('./image')
            return send_file('./image', mimetype='image/jpeg')
    return 200  

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)