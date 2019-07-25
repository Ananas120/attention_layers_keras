import keras.backend as K

from keras.layers import Layer

        
class SimpleAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a',
                                   shape=(input_shape[0][2], input_shape[0][2]),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=(input_shape[1][2], input_shape[0][2]),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=(input_shape[0][2], 1),
                                   initializer='uniform',
                                   trainable=True)
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, verbose=False):
        """
        Inputs : [encoder_output_sequence, decoder_input_sequence]
        decoder_input_sequence = last_decoder_output_sequence
        """
        assert isinstance(inputs, list)
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print("encoder_out_seq shape (batch_size, input_timesteps, encoder_size): {}".format(encoder_out_seq.shape))
            print("decoder_out_seq shape (batch_size, last_outputs_timesteps, decoder_size): {}".format(decoder_out_seq.shape))
      
        def energy_step(self, inputs, states):
            """ Step function for computing energy for a single decoder state """""
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping Tensors """""
            en_seq_len, en_hidden = K.shape(encoder_out_seq)[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s, ..., si] """""

            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))

            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print("Wa x s :{}".format(W_a_dot_s.shape))

            """ Computing hj.Ua """""
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)
            if verbose:
                print("Ua x h : {}".format(U_a_dot_h.shape))

            """ tanh(S.Wa + hj.Ua) """""

            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print("Ws + Uh : {}".format(reshaped_Ws_plus_Uh.shape))

            """ Softmax (va.tanh(S.Wa + hj.Ua)) """""

            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))

            e_i = K.softmax(e_i)

            if verbose:
                print("E_i : {}".format(e_i.shape))

            return e_i, [e_i]

        def context_step(self, inputs, states):
            """ Step function for computing c_i using e_i """""

            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print("c_i : {}".format(c_i.shape))
            return c_i, [c_i]

        def create_initial_state(self, inputs, hidden_size):
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])
            fake_state = K.expand_dims(fake_state)
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state
    

        fake_state_c = self.create_initial_state(encoder_out_seq, K.shape(encoder_out_seq)[-1])
        fake_state_e = self.create_initial_state(encoder_out_seq, K.shape(encoder_out_seq)[1])
        if verbose:
            print("fake_state_c : {}".format(fake_state_c.shape))
            print("fake_state_e : {}".format(fake_state_e.shape))

        """ Computing energy outputs """""
        last_out, e_outputs, _ = K.rnn(self.energy_step,
                                       decoder_out_seq, 
                                       [fake_state_e])
        """ Computing context vectors """""
        last_out, c_outputs, _ = K.rnn(self.context_step,
                                       e_outputs,
                                       [fake_state_c])

        if verbose:
            print("energy outputs : {}".format(e_outputs.shape))
            print("context vectors : {}".format(c_outputs.shape))

        return [c_outputs, e_outputs]
  


    def comute_output_shape(self, input_shape):
        """ Outputs produced by the layer """""
        return [
            (input_shape[1][0], input_shape[1][1], input_shape[1][2]),
            (input_shape[1][0], input_shape[1][1], input_shape[0][1])
        ]
