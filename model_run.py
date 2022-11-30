
# model_runだけで完結します

import pickle
import numpy as np
from gensim.models import KeyedVectors
from gensim.matutils import unitvec

WV_FILE = 'key_word2vec.bin'
MODEL_NAME = '221130_takky_random.sav'

def predict_level(WV_FILE, MODEL_NAME, INPUT):
    averaging_data = input_kakou(WV_FILE, INPUT).reshape(1, -1)
    loaded_model = pickle.load(open(MODEL_NAME, 'rb'))
    result = loaded_model.predict(averaging_data)
    return result

def input_kakou(WV_FILE, text):
    key_wv = KeyedVectors.load_word2vec_format(WV_FILE, binary=True)
    averaging_result = averaging(key_wv, w2v_tokenize_text(text))
    return averaging_result

def w2v_tokenize_text(text):
    tokens = text.split()
    return tokens

def averaging(key_wv, words):
    mean = np.mean([key_wv .get_vector(word) for word in words if word in key_wv], axis=0)
    return unitvec(mean)


if __name__ == '__main__':

    input_data = 's 239 SPACE 167 i 180 n 238 SPACE 172 Shift 1144 "" 244 a 248 b 385 c 214 d 397 e 313 f 251 g 153 Shift 238 "" 213 Shift'

    loaded_model = pickle.load(open(MODEL_NAME, 'rb'))
    result = predict_level(WV_FILE, MODEL_NAME, input_data)

    print(f'あなたは…{result}')
