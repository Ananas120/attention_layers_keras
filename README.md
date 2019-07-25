# attention_layers_keras
Implementations of differents attention layer with keras library

These layers are tested with 
- tensorflow == 1.14.0
- keras == 2.2.1

The SimpleAttentionLayer is inspired by this repo : 
https://github.com/thushv89/attention_keras
but the implementation did not work for me (i don't understand why) but now it works fine. 
The LocationSensitiveAttentionLayer is inspired by this implementation : 
https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/models/attention.py
This implementation is in tensorflow and use the BahdanauAttentionMechanism from tensorflow.xx.attention_wrapper.py. 
I think my implementation is correct but if if anyone can confirm...

Example usage : 

from keras.layers import *
from keras.models import *

attention_dim = 128

#simple encoder
input_encoder = Input(shape=(None,))
embedding = Embedding(input_dim=26, output_dim=32)(input_encoder)
x = CuDNNLSTM(attention_dim, return_sequences=True)(embedding)

#simple decoder with attention
input_decoder = Input(shape=(None, 26))  
processed_decoder = Dense(attention_dim, activation='relu')(input_decoder)
attention_layer = LocationSensitiveAttentionLayer(units=attention_dim, filters=16)
context, weights = attention_layer([x, processed_decoder], verbose=True) #verbose is to see the different shape during the attention mechanism

out = Concatenate()([context, processed_decoder])

out = Dense(26, activation='softmax')(out)

full_model = Model([input_encoder, input_decoder], out)
#full_model = Model([input_encoder, input_decoder], [out, weights]) to see get the attention_weights 


full_model.summary(150)

i1 = np.ones((1, 26))
i2 = np.ones((1, 4, 26))

full_model.compile(loss='mse', optimizer='adam')
full_model.predict([i1, i2])
