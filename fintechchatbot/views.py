from django.shortcuts import render,HttpResponse
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import json
from keras.models import load_model
import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


#preparing initial variables.
# make sure chatbot.h5 and intents.json are in the same directory as that of the code

file_location = os.path.join(settings.BASE_DIR,'fintech_chatbot.h5')
classifier=load_model(file_location)

file_location = os.path.join(settings.BASE_DIR,"intents.json")

with open(file_location) as json_data:
    intents = json.load(json_data)
# classes must be changed/updated after change in intents.json
classes = ['buybits', 'funny', 'goodbye', 'greeting', 'investments', 'profitloss', 'thanks', 'transactions']
words  =['a', 'ad', 'al', 'and', 'any', 'anyon', 'ar', 'bitpoint', 'buy', 'bye', 'can', 'day', 'did', 'do', 'don', 'ev', 'funny', 'go', 'good', 'goodby', 'hello', 'help', 'hey', 'hi', 'hist', 'how', 'i', 'int', 'invest', 'is', 'jok', 'know', 'lat', 'loss', 'lot', 'me', 'mor', 'morn', 'my', 'nee', 'net', 'of', 'ok', 'pay', 'profit', 'see', 'seey', 'show', 'som', 'someth', 'spend', 'tel', 'thank', 'that', 'the', 'ther', 'to', 'tot', 'transact', 'ty', 'tysm', 'tyvm', 'want', 'wassup', 'wher', 'ya', 'you']




def stem_string(ipstring,mainwords):

  words_second =  [stemmer.stem(word.lower()) for word in nltk.word_tokenize(ipstring)]
  finlist = []
  for x in mainwords:
    if x in words_second:
      finlist.append(1)
    else:
      finlist.append(0)
  return finlist


def respond(category_word):
  import random
  for x in intents['intents']:
    if x['tag'] == category_word:
      resp  =x['responses'][random.randint(0,len(x['responses'])-1)]
      break
  return resp

def predict_output(user_input_string):
  to_predict = stem_string(user_input_string,words)
  output_vector = list(classifier.predict(np.reshape(to_predict,[1,len(to_predict)]))[0])
  print(output_vector)
  print(classes)
  return(respond(classes[output_vector.index(max(output_vector))]))

print("TEST OUTPUT : ",predict_output('Show my transactions'))
@csrf_exempt
def chatbot_api(request):
  if request.method == 'POST':
    print('********call at chatbot API ***********')
    #print(request.POST)
    #print(request.body)
    #print('PRINT 3: ',request.POST.get('user_message'))
    user_message = request.POST.get('user_message')
    print('User Message is : ',user_message)
    result = {'bot_message': predict_output(user_message)}
    resp = JsonResponse(result)
    resp['Access-Control-Allow-Origin'] = '*'
    return resp
   
  else:
    return HttpResponse('Invalid Request')


def index(request):
    return HttpResponse('''To Use the Chatbot API, post requests in the following format:

    	{user_message: "Your Message Here ex: Hi "}

    	Post The Request on /chatbotapi

    	''')