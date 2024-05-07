import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image 
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array

base_model = InceptionV3(weights = 'base_model.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)

def preprocess_img(img_path):
    img = load_img(img_path, target_size = (299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess_img(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec


pickle_in = open("wordtoix.pkl", "rb")
wordtoix = load(pickle_in)
pickle_in = open("ixtoword.pkl", "rb")
ixtoword = load(pickle_in)
max_length = 74

def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = image
            preds = model.predict([e, np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


model = load_model('new-model-1.h5')

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Image Caption Generator')
top.configure(background='#CDCDCD')
label2=Label(top,background='#CDCDCD', font=('arial',15))
label1=Label(top,background='#CDCDCD', font=('arial',15))
label=Label(top,background='#CDCDCD', font=('arial',15))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    enc = encode(file_path)
    image = enc.reshape(1, 2048)
    pred = greedy_search(image)
    print(pred)
    label.configure(foreground='#000', text= 'Greedy: ' + pred)
    label.pack(side=BOTTOM,expand=True)
    beam_3 = beam_search(image)
    print(beam_3)
    label1.configure(foreground='#011638', text = 'Beam-3: ' + beam_3)
    label1.pack(side = BOTTOM, expand = True)
    beam_5 = beam_search(image, 5)
    print(beam_5)
    label2.configure(foreground='#228B22', text = 'Beam-5: ' + beam_5)
    label2.pack(side = BOTTOM, expand = True)

def show_classify_button(file_path):
    classify_b=Button(top,text="Generate",command=lambda: classify(file_path),padx=30,pady=15)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        label1.configure(text='')
        label2.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image",command=upload_image,padx=30,pady=15)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)


heading = Label(top, text="Image Caption Generator",pady=20, font=('arial',22,'bold'))
heading.configure(background='#CDCDCD',foreground='#FF6347')
heading.pack()
top.mainloop()
