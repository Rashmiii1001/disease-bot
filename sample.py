from flask import Flask,render_template,request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import part1


app = Flask(__name__)
english_bot = ChatBot("Chatterbot", storage_adapter = "chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")
trainer.train("data/data.yml")
userTextArr = []
botAnswerArr = []

@app.route("/")
def index():
    return render_template("new.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg")
    botAnswer=""
    userTextArr.append(userText)
    print("array-" ,userTextArr)

    if (len(userTextArr)==1):
        botAnswer = part1.part1(userText)
    elif (len(userTextArr)==2):
        botAnswer = part1.part2(userText, botAnswerArr[0])
    elif (len(userTextArr)==3):
        botAnswer = part1.part3(userText, botAnswerArr[1])
    elif (len(userTextArr)==4):
        botAnswer = part1.part4()
    # botAnswer = str(english_bot.get_response(userText))

    botAnswerArr.append(botAnswer)
    print("sample - " ,str(botAnswer))
    
    return str(botAnswer)

if __name__=="__main__":
    app.run(debug=True)