import keras.backend as K

from keras.layers import Layer
from tensorflow.python.ops import math_ops

        
class LocationSensitiveAttentionLayer(Layer):
    def __init__(self, units, filters, kernel=3, smoothing=False, cumulate_weights=True, **kwargs):
        super(LocationSensitiveAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self._cumulate = cumulate_weights
        
        self.location_convolution = Conv1D(filters=filters, kernel_size=kernel, padding='same', bias_initializer='zeros', name='location_features_convolution')
        self.location_layer = Dense(units, use_bias=False, name='location_features_layer')
        self.query_layer = None
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.v_a = self.add_weight(name='V_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_a = self.add_weight(name='b_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        
        super(LocationSensitiveAttentionLayer, self).build(input_shape)
    
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
      
    
        def energy_step(query, states):
            """ Step function for computing energy for a single decoder state """""
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping Tensors """""

            previous_alignments = states[0]
            
            processed_query = self.query_layer(query) if self.query_layer else query
            
            expanded_alignments = K.expand_dims(previous_alignments, axis=2)
            
            f = self.location_convolution(expanded_alignments)
            
            processed_location_features = self.location_layer(f)
            
            if verbose:
                print("query : {}".format(query.shape))
                print("previous_alignments : {}".format(previous_alignments.shape))
                print("processed_query : {}".format(processed_query.shape))
                print("f : {}".format(f.shape))
                print("processed_location_features : {}".format(processed_location_features.shape))
                
            
            e_i = K.sum(self.v_a * K.tanh(encoder_out_seq + processed_query + processed_location_features + self.b_a), [2])
            alignments = K.softmax(e_i)
            
            if self._cumulate:
                next_state = alignments + previous_alignments
            else:
                next_state = alignments

            if verbose:
                print("E_i : {}".format(e_i.shape))

            return alignments, [next_state]

        def context_step(inputs, states):
            """ Step function for computing c_i using e_i """""
            
            alignments = inputs
            expanded_alignments = K.expand_dims(alignments, 1)
            
            if verbose:
                print("expanded_alignments : {}".format(expanded_alignments.shape))
            
            c_i = math_ops.matmul(expanded_alignments, encoder_out_seq)
            c_i = K.squeeze(c_i, 1)
            
            if verbose:
                print("c_i : {}".format(c_i.shape))
            return c_i, [c_i]

        def create_initial_state(inputs, hidden_size):
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])
            fake_state = K.expand_dims(fake_state)
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        fake_state_c = create_initial_state(encoder_out_seq, K.shape(encoder_out_seq)[-1])
        fake_state_e = create_initial_state(encoder_out_seq, K.shape(encoder_out_seq)[1])
        if verbose:
            print("fake_state_c : {}".format(fake_state_c.shape))
            print("fake_state_e : {}".format(fake_state_e.shape))

        """ Computing energy outputs """""
        last_out, e_outputs, _ = K.rnn(energy_step,
                                       decoder_out_seq, 
                                       [fake_state_e])
        """ Computing context vectors """""
        last_out, c_outputs, _ = K.rnn(context_step,
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
