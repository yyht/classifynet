from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time
import numpy as np
import tensorflow as tf

def slstm_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states, num_layers):
    with tf.name_scope(name_scope_name):
        #Word parameters 
        #forget gate for left 
        with tf.name_scope("f1_gate"):
            #current
            Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
            #left right
            Whf1 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            #initial state
            Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
            #dummy node
            Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
        #forget gate for right 
        with tf.name_scope("f2_gate"):
            Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
            Whf2 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
            Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
        #forget gate for inital states     
        with tf.name_scope("f3_gate"):
            Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
            Whf3 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
            Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
        #forget gate for dummy states     
        with tf.name_scope("f4_gate"):
            Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
            Whf4 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
            Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
        #input gate for current state     
        with tf.name_scope("i_gate"):
            Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxi")
            Whi = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whi")
            Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wii")
            Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdi")
        #input gate for output gate
        with tf.name_scope("o_gate"):
            Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
            Who = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wio")
            Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdo")
        #bias for the gates    
        with tf.name_scope("biases"):
            bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
            bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
            bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf1")
            bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf2")
            bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf3")
            bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf4")

        #dummy node gated attention parameters
        #input gate for dummy state
        with tf.name_scope("gated_d_gate"):
            gated_Wxd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
            gated_Whd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
        #output gate
        with tf.name_scope("gated_o_gate"):
            gated_Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
            gated_Who = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
        #forget gate for states of word
        with tf.name_scope("gated_f_gate"):
            gated_Wxf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
            gated_Whf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
        #biases
        with tf.name_scope("gated_biases"):
            gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
            gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
            gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")

    #filters for attention        
    mask_softmax_score=tf.cast(tf.sequence_mask(lengths), tf.float32)*1e25-1e25
    mask_softmax_score_expanded=tf.expand_dims(mask_softmax_score, dim=2)               
    #filter invalid steps
    sequence_mask=tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32),axis=2)
    #filter embedding states
    initial_hidden_states=initial_hidden_states*sequence_mask
    initial_cell_states=initial_cell_states*sequence_mask
    #record shape of the batch
    shape=tf.shape(initial_hidden_states)
    
    #initial embedding states
    embedding_hidden_state=tf.reshape(initial_hidden_states, [-1, hidden_size])      
    embedding_cell_state=tf.reshape(initial_cell_states, [-1, hidden_size])

    #randomly initialize the states
    if config.random_initialize:
        initial_hidden_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
        initial_cell_states=tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None, name=None)
        #filter it
        initial_hidden_states=initial_hidden_states*sequence_mask
        initial_cell_states=initial_cell_states*sequence_mask

    #inital dummy node states
    dummynode_hidden_states=tf.reduce_mean(initial_hidden_states, axis=1)
    dummynode_cell_states=tf.reduce_mean(initial_cell_states, axis=1)

    for i in range(num_layers):
        #update dummy node states
        #average states
        combined_word_hidden_state=tf.reduce_mean(initial_hidden_states, axis=1)
        reshaped_hidden_output=tf.reshape(initial_hidden_states, [-1, hidden_size])
        #copy dummy states for computing forget gate
        transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
        #input gate
        gated_d_t = tf.nn.sigmoid(
            tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd
        )
        #output gate
        gated_o_t = tf.nn.sigmoid(
            tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo
        )
        #forget gate for hidden states
        gated_f_t = tf.nn.sigmoid(
            tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf
        )

        #softmax on each hidden dimension 
        reshaped_gated_f_t=tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size])+ mask_softmax_score_expanded
        gated_softmax_scores=tf.nn.softmax(tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)
        #split the softmax scores
        new_reshaped_gated_f_t=gated_softmax_scores[:,:shape[1],:]
        new_gated_d_t=gated_softmax_scores[:,shape[1]:,:]
        #new dummy states
        dummy_c_t=tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) + tf.squeeze(new_gated_d_t, axis=1)*dummynode_cell_states
        dummy_h_t=gated_o_t * tf.nn.tanh(dummy_c_t)

        #update word node states
        #get states before
        initial_hidden_states_before=[tf.reshape(self.get_hidden_states_before(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
        initial_hidden_states_before=self.sum_together(initial_hidden_states_before)
        initial_hidden_states_after= [tf.reshape(self.get_hidden_states_after(initial_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
        initial_hidden_states_after=self.sum_together(initial_hidden_states_after)
        #get states after
        initial_cell_states_before=[tf.reshape(self.get_hidden_states_before(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
        initial_cell_states_before=self.sum_together(initial_cell_states_before)
        initial_cell_states_after=[tf.reshape(self.get_hidden_states_after(initial_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
        initial_cell_states_after=self.sum_together(initial_cell_states_after)
        
        #reshape for matmul
        initial_hidden_states=tf.reshape(initial_hidden_states, [-1, hidden_size])
        initial_cell_states=tf.reshape(initial_cell_states, [-1, hidden_size])

        #concat before and after hidden states
        concat_before_after=tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

        #copy dummy node states 
        transformed_dummynode_hidden_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
        transformed_dummynode_cell_states=tf.reshape(tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1],1]), [-1, hidden_size])

        f1_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) + 
            tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1)+ bf1
        )

        f2_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) + 
            tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2)+ bf2
        )

        f3_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) + 
            tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
        )

        f4_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) + 
            tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
        )
        
        i_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) + 
            tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi)+ bi
        )
        
        o_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) + 
            tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
        )
        
        f1_t, f2_t, f3_t, f4_t, i_t=tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1),tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)


        five_gates=tf.concat([f1_t, f2_t, f3_t, f4_t,i_t], axis=1)
        five_gates=tf.nn.softmax(five_gates, dim=1)
        f1_t,f2_t,f3_t, f4_t,i_t= tf.split(five_gates, num_or_size_splits=5, axis=1)
        
        f1_t, f2_t, f3_t, f4_t, i_t=tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1),tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1),tf.squeeze(i_t, axis=1)

        c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after)+(f3_t * embedding_cell_state)+ (f4_t * transformed_dummynode_cell_states)+ (i_t * initial_cell_states)
        
        h_t = o_t * tf.nn.tanh(c_t)

        #update states
        initial_hidden_states=tf.reshape(h_t, [shape[0], shape[1], hidden_size])
        initial_cell_states=tf.reshape(c_t, [shape[0], shape[1], hidden_size])
        initial_hidden_states=initial_hidden_states*sequence_mask
        initial_cell_states=initial_cell_states*sequence_mask

        dummynode_hidden_states=dummy_h_t
        dummynode_cell_states=dummy_c_t

    initial_hidden_states = tf.nn.dropout(initial_hidden_states,self.dropout)
    initial_cell_states = tf.nn.dropout(initial_cell_states, self.dropout)

    return initial_hidden_states, initial_cell_states, dummynode_hidden_states