from flask import Flask
import numpy as np
import math

app = Flask(__name__)
i = 0;

@app.route('/')
def hello_world():
    global i
    i= i+1
    return 'Hello, World!' + str(i)


@app.route('/action/check')
def action_check():
    return str(int( math.floor( (np.random.random() * 100))));


@app.route('/action/step-ret/<string:ret>')
def action_ret(ret):
    return ret;
