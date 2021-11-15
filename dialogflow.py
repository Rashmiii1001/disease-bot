from flask import Flask, request, make_response
import json
from flask_cors import cross_origin
from logger import logger
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)

    #print("Request:")
    #print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# processing the request from dialogflow
def processRequest(req):
    log = logger.Log()

    sessionID=req.get('responseId')

    result = req.get("queryResult")
    user_says=result.get("queryText")
    log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    user_symptoms=parameters.get("Symptoms")
    #print(user_symptoms)
 	
    intent = result.get("intent").get('displayName')
    if (intent=='Symptoms-1'):

        fulfillmentText=""

        def diseaseDetail(term):
            diseases=[term]
            ret=term+"\n"
            for dis in diseases:
                # search "disease wilipedia" on google
                query = dis+' wikipedia'
                # tld="co.in"
                # ,stop=10,pause=0.5
                for sr in search(query):
                    # open wikipedia link
                    match=re.search(r'wikipedia',sr)
                    filled = 0
                    if match:
                        wiki = requests.get(sr,verify=False)
                        soup = BeautifulSoup(wiki.content, 'html5lib')
                        # Fetch HTML code for 'infobox'
                        info_table = soup.find("table", {"class":"infobox"})
                        if info_table is not None:
                            # Preprocess contents of infobox
                            for row in info_table.find_all("tr"):
                                data=row.find("th",{"scope":"row"})
                                if data is not None:
                                    symptom=str(row.find("td"))
                                    symptom = symptom.replace('.','')
                                    symptom = symptom.replace(';',',')
                                    symptom = symptom.replace('<b>','<b> \n')
                                    symptom=re.sub(r'<a.*?>','',symptom) # Remove hyperlink
                                    symptom=re.sub(r'</a>','',symptom) # Remove hyperlink
                                    symptom=re.sub(r'<[^<]+?>',' ',symptom) # All the tags
                                    symptom=re.sub(r'\[.*\]','',symptom) # Remove citation text                    
                                    symptom=symptom.replace("&gt",">")
                                    ret+=data.get_text()+" - "+symptom+"\n"
                                    # print(data.get_text(),"-",symptom)
                                    filled = 1
                        if filled:
                            break
            return ret

            for i in range(len(user_symptoms)):
                detail = user_symptom[i] + "\n" + diseaseDetail(user_symptom[i]) + "\n"
                fulfillmentText += detail
        

        
            log.write_log(sessionID, "Bot Says: "+fulfillmentText)
            return {
                "fulfillmentText": fulfillmentText
            }
    else:
        log.write_log(sessionID, "Bot Says: " + result.fulfillmentText)


if __name__ == '__main__':
    app.run(debug=True)


