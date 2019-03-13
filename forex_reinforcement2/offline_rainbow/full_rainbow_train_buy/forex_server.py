from flask import Flask
from flask import request
import numpy as np
import math
from forex_agent import ForexAgent
from forex_environment import ForexEnvironment
import threading;

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

app = Flask(__name__)

app.logger.disabled = True

myAgent = ForexAgent(ForexEnvironment());

def threadMethod(data):
    myAgent.mainLoop();

th = None;
th = threading.Thread(target=threadMethod,args=[None]);
th.start();

    


@app.route('/')
def hello_world():
   
    return 'Hello, World!';


@app.route('/action/check')
def action_check():
    i=0;
    while(not myAgent.env._step_started):
        i = i+ 1;
    myAgent.env._step_started = False;
    action = myAgent.env._last_action;
    strAction =  str(action);

    return strAction;


@app.route('/action/step-ret',methods=[ 'POST'])
def action_ret():
    #"<high>,<low>,<open>,<close>,<avgm>,<avgh>,<avgd>,<month>,<daym>,<dayw>,<hour>,<min>,<ask>,<bid>"
    ret = request.form["ret"]
    
    
    arr = ret.split(',');
    arrTitles = arr[0:14];
    arr = arr[14:];
    arrFloat = [float(i) for i in arr];
    reward = arrFloat[-2];
    gameOver = arrFloat[-1];
    last_pos = arrFloat[-3];
    myAgent.env._last_game_over = (True  if gameOver > 0  else False);
    myAgent.env._last_reward = reward;
    myAgent.env._last_pos = last_pos;
    if len(arr) < (50*60*4*14):
        myAgent.env._last_state = None;
        myAgent.env._step_ended = True;

        return  ret;

   
    
    state = arrFloat[0:(50*60*4*14)];
    stateNum = np.array(state,dtype=np.float32);
    stateNum = np.reshape(stateNum,(50*60*4,14));
    
    

    myAgent.env._last_state = stateNum;
    myAgent.env._step_ended = True;

    return ret;
