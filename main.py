import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import time
from spellchecker import SpellChecker
from tkinter import StringVar
import pickle

spell = SpellChecker()

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

root = tk.Tk()
root.title("SilentScript")

frame_label = tk.Label(root)
frame_label.pack()

character_label = tk.Label(root, text="Predicted Character: ")
character_label.pack()

sentence_label = tk.Label(root, text="Predicted Sentence: ")
sentence_label.pack()

current_word_entry = tk.Entry(root)
current_word_entry.pack()

suggestions_label = tk.Label(root, text="Suggestions:")
suggestions_label.pack()

suggestion_text = StringVar()
suggestion_display = tk.Label(root, textvariable=suggestion_text)
suggestion_display.pack()

current_word = ''
predicted_sentence = ''
new_character_detected = False
start_time = None


def read_aloud(predicted_character):
    engine = pyttsx3.init()
    engine.say(predicted_character)
    engine.runAndWait()


def suggest_words(current_word):
    corrected_word = spell.correction(current_word)
    if corrected_word is not None:
        suggestions = spell.candidates(corrected_word)
        if not suggestions:
            suggestion_text.set("No suggestions available")
        else:
            suggestion_text.set(', '.join(suggestions))
    else:
        suggestion_text.set("No suggestions available")


def on_character_predicted(predicted_character):
    current_word = current_word_entry.get()
    current_word += predicted_character
    current_word_entry.delete(0, tk.END)
    current_word_entry.insert(0, current_word)
    suggest_words(current_word)


def clear_last_character():
    global current_word
    if len(current_word) > 0:
        current_word = current_word[:-1]
        current_word_entry.delete(0, tk.END)
        current_word_entry.insert(0, current_word)
        suggest_words(current_word)


clear_button = tk.Button(root, text="Clear Last Character", command=clear_last_character)
clear_button.pack()


def update_frame():
    global current_word, predicted_sentence, new_character_detected, start_time

    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    no_hands_detected = not results.multi_hand_landmarks

    if no_hands_detected:
        predicted_character = 'NO HANDS'
        new_character_detected = False
        start_time = None
    else:
        if start_time is None:
            start_time = time.time()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = []
            y_ = []
            data_aux = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10

            x2 = int(max(x_) * frame.shape[1]) - 10
            y2 = int(max(y_) * frame.shape[0]) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    character_label.config(text="Predicted Character: " + predicted_character)

    if predicted_character != 'NO HANDS' and not new_character_detected and start_time is not None:
        elapsed_time = time.time() - start_time
        if elapsed_time > 2.0:
            if predicted_character == 'space':
                current_word += ' '
                new_character_detected = True
                start_time = None
            else:
                current_word += predicted_character
                read_aloud(predicted_character)
                new_character_detected = True
                start_time = None

    sentence_label.config(text="Possible word: " + predicted_sentence + current_word)
    suggest_words(current_word)


    img = ImageTk.PhotoImage(image=Image.fromarray(frame))
    frame_label.img = img
    frame_label.config(image=img)
    frame_label.after(10, update_frame)


update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
