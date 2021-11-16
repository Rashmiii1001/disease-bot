from flask import Flask, request, make_response, render_template
import json
from flask_cors import cross_origin
# from logger import logger
import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search
import warnings
warnings.filterwarnings("ignore")
import requests

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    print("INSIDE WEBHOOK")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):
    # log = logger.Log()
    print("INSIDE PROCESSREQUEST")

    sessionID=req.get('responseId')

    result = req.get("queryResult")
    user_says=result.get("queryText")
    # log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    user_symptom = parameters.get("Symptoms")
    print(user_symptom)
 	
    intent = result.get("intent").get('displayName')
    if (intent=='Symptoms-1'):

        fulfillmentText=""

       

        for i in range(len(user_symptom)):
            print("Inside for loop")
            detail = user_symptom[i] + "\n" + "\n"
            fulfillmentText += detail
    

    
        # log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": fulfillmentText
        }
    else:
        # log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)
        print("else part")


if __name__ == '__main__':
    app.run(debug=True)


