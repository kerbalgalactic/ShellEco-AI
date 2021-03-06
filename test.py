import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

import random
import json
import pickle

import openai

import deep_translator
from deep_translator import GoogleTranslator

tr2en = GoogleTranslator(source='tr', target='en')
en2tr = GoogleTranslator(source='en', target='tr')


openai.api_key = ""



with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=200, batch_size=2, show_metric=True)
    model.save("model.tflearn")


def apiac(inp):
    global completion, prefix
    ### Prompt engineering :)
    ### Or as you may call it, "dark magic"
    prefix_file = open("prefix.txt","r")
    prefix = prefix_file.read()
    prefix_file.close()
    response = completion.create(
        prompt=prefix+inp, engine='text-davinci-002', stop=['Human'], temperature=0,
        top_p=0,presence_penalty = 0.4, frequency_penalty=0, max_tokens=400)
    answer = response.choices[0].text.strip()
    answer = "\n".join(answer.split("\n"))
    return answer


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

statics = {'name': '"AI"',
           'radio_frequency': '91.2',
           'charge':'"%80"',
           'rangeleft':'"150 km"'}


history = ""

allowed_history_length = 3

completion = openai.Completion()

treshold = .99

while True:
    inp = input(">>> ")
    if inp.lower() == "quit":
        break

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    history += "\nHuman: "+inp
    
    if results.max() > treshold:
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        response = random.choice(responses)
        
        if "{" in response and not "_" in response:
            responses = response.split("\n")
            for response in responses:
                strip = response.split("{")[1]
                strip = strip.split("}")[0]
                exec(strip+" = "+statics[strip])
                exec("response = response.format("+strip+"="+strip+")")
                print(response)
        else:
            print(response)
        history += "\nAI: "+response
        history = history.strip('\n')
    else:
        response = apiac(tr2en.translate(history)+"\nAI: ")
        if len(response) == 0:
            continue
        if "{" not in response: response = en2tr.translate(response)
        history += "\nAI: "+response
        print(response)
    try:
        "Human"+"Human".join(history.split('Human')[len(history.split('Human'))-allowed_history_length:len(history.split('Human'))])
    except:
        pass
    print("CS:",results.max(),"|||", "Mode: DNN" if results.max() > treshold else "Mode: GPT")
        

