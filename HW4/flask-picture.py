#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/test', methods=['POST']) 
def test():    
    data = request.get_json()
    print(data)
    resp = jsonify(success=True)
    return resp

    

if __name__ == '__main__':
    app.run('0.0.0.0')