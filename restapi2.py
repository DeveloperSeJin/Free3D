import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import time
import Diffuse
import GetPrompt


app = Flask(__name__)
CORS(app)
@app.route('/call', methods= ['GET', 'POST'])
def call():
    if request.method == 'GET':
        get_type = request.args.get("type")
        if get_type == "Text" :
            
            return "Return text"
        elif get_type == "Image" :
            Image = request.files['Image']
            Image.save('./image')
            return send_file('./image', mimetype='image/jpeg')
        
        elif get_type == "Diffusion":
            text = request.args.get("text")
            prompt = GetPrompt.getPrompt(text)
            Image = Diffuse.run(prompt)
            Image.save('./image')
            return send_file('./image', mimetype='image/jpeg')
    return 200  

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)