import tensorflow as tf

def RandomUniformInitializer(shape, initrange = 0.1):
    return tf.random_uniform(shape = shape, minval = -initrange, maxval = initrange)
def NormalInitializer(shape, mean = 0, stddev = 0.02):
    return tf.random_normal(shape = shape, mean = mean, stddev = stddev)

class LeakyReluActivation(object):
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, input):
        return tf.nn.relu(input) - self.alpha * tf.nn.relu(-input)

class LinearLayer(object):
    def __init__(self, ninput, noutput, initialization_fn = None, activation_fn = None, batch_norm = False, batch_norm_w = None):
        if initialization_fn == None:
            initialization_fn = RandomUniformInitializer((ninput, noutput))
        self.W_values = tf.Variable(initialization_fn)
        self.b_values = tf.Variable(tf.zeros((noutput,), tf.float32))
        if batch_norm_w != None:
            self.bn_values = batch_norm_w
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
    def __call__(self, input, is_train = tf.constant(True)):
        output = tf.matmul(input, self.W_values) + self.b_values
        if self.batch_norm == True:
            output = tf.layers.batch_normalization(output, momentum=0.1, epsilon=1e-05, training = is_train)
        if self.activation_fn != None:
            output = self.activation_fn(output)
        return output

class EmbeddingLayer(object):
    def __init__(self, nhidden, ntokens, initrange):
        self.W_values = tf.Variable(tf.random_uniform(shape = (nhidden, ntokens), minval = -initrange, maxval = initrange))
    def __call__(self, input):
        return tf.nn.embedding_lookup(self.W_values, input)

class MLP_G(object):
    def __init__(self, ninput, noutput, layers, activation=tf.nn.relu):
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            initializer = NormalInitializer((layer_sizes[i], layer_sizes[i + 1]))

            layer = LinearLayer(layer_sizes[i], layer_sizes[i + 1], initialization_fn = initializer, activation_fn = activation, batch_norm = True)
            self.layers.append(layer)

        initializer = NormalInitializer((layer_sizes[-1], noutput))

        layer = LinearLayer(layer_sizes[-1], noutput, initialization_fn = initializer, activation_fn = None, batch_norm = False)
        self.layers.append(layer)

    def __call__(self, x, is_train = tf.constant(True), layer_id = None):
        for i, layer in enumerate(self.layers):
            x = layer(x, is_train)
            if ((layer_id != None) and (i == layer_id)):
                return x
        return x # batch_size x nhidden

class MLP_D(object):
    def __init__(self, ninput, noutput, layers, activation=LeakyReluActivation(0.2)):
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            initializer = NormalInitializer((layer_sizes[i], layer_sizes[i + 1]))

            bnorm = False
            if i != 0:
                bnorm = True
            layer = LinearLayer(layer_sizes[i], layer_sizes[i + 1], initialization_fn = initializer, activation_fn = activation, batch_norm = bnorm)
            self.layers.append(layer)

        initializer = NormalInitializer((layer_sizes[-1], noutput))

        layer = LinearLayer(layer_sizes[-1], noutput, initialization_fn = initializer, activation_fn = None, batch_norm = False)
        self.layers.append(layer)

    def __call__(self, x, reduce_mean = False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        final_layer = x # batch_size x 1
        if reduce_mean == True:
            return tf.identity(tf.reduce_mean(final_layer, name = "d_reduce_mean"))
        else:
            return final_layer

class Seq2SeqLayer(object):
    def __init__(self, batch_size, emsize, nhidden, ntokens, nlayers, noise_radius, hidden_init, dropout):
        self.emsize = emsize
        self.nhidden = nhidden
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.batch_size = batch_size

        initrange = 0.1

        self.start_symbols = tf.Variable(tf.ones([2, 3], tf.int64))

        self.embedding = EmbeddingLayer(ntokens, emsize, initrange)
        self.embedding_decoder = EmbeddingLayer(ntokens, emsize, initrange)

        with tf.variable_scope('encoder'):
            self.encoder = tf.contrib.rnn.BasicLSTMCell(nhidden)

        self.decoder_input_size = emsize+nhidden

        with tf.variable_scope('decoder'):
            self.decoder = tf.contrib.rnn.BasicLSTMCell(nhidden)

        self.linear = LinearLayer(nhidden, ntokens)

    def __call__(self, indices, lengths, noise, encode_only=False, reuse = False):
        maxlen = tf.shape(indices)[1]

        with tf.variable_scope('encoder', reuse = reuse):
            hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        with tf.variable_scope('decoder', reuse = reuse):
            decoded = self.decode(hidden, maxlen, indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices) # batch_size x maxLen x embSize

        # Encode
        # batch_size x maxLen x hSize
        packed_output, state = tf.nn.dynamic_rnn(self.encoder, embeddings, sequence_length = lengths, dtype = tf.float32)
        cell, hidden = state

        # normalize to unit ball
        hidden = tf.nn.l2_normalize(hidden, 1)

        if noise and self.noise_radius > 0:
            gauss_noise = tf.random_normal(tf.shape(hidden), mean=0, stddev=self.noise_radius)
            hidden = hidden + gauss_noise

        return hidden # batch_size x nHidden

    def init_hidden(self):
        return tf.contrib.rnn.LSTMStateTuple(tf.zeros((self.batch_size, self.nhidden)), tf.zeros((self.batch_size, self.nhidden)))

    def init_state(self):
        return tf.zeros((self.batch_size, self.nhidden))

    def decode(self, hidden, maxlen, indices=None, lengths=None):
        # batch x nHidden -> batch_size x maxLen x nHidden
        all_hidden = tf.tile(tf.expand_dims(hidden, 1), [1, maxlen, 1])

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = tf.contrib.rnn.LSTMStateTuple(hidden, self.init_state()) # (batch_size x nHidden, batch_size x nHidden)
        else:
            state = self.init_hidden() # (batch_size x nHidden, batch_size x nHidden)

        embeddings_decoded = self.embedding_decoder(indices) # batch_size x maxLen x embSize
        augmented_embeddings = tf.concat([embeddings_decoded, all_hidden], 2) # batch_size x maxLen x (embSize + nHidden)

        # batch_size x maxLen x (embSize + nHidden) -> batch_size x maxLen x nHidden
        output, state = tf.nn.dynamic_rnn(self.decoder, augmented_embeddings, sequence_length = lengths, initial_state = state, dtype = tf.float32)

        decoded = tf.reshape(self.linear(tf.reshape(output, (self.batch_size * maxlen, self.nhidden))), (self.batch_size, maxlen, self.ntokens))

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0, reuse = False):
        """Generate through decoder; no backprop"""

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = tf.contrib.rnn.LSTMStateTuple(hidden, self.init_state()) # (batch_size x nHidden, batch_size x nHidden)
        else:
            state = self.init_hidden() # (batch_size x nHidden, batch_size x nHidden)

        # <sos>
        start_symbols = tf.ones((self.batch_size, 1), dtype = tf.int64)
        embeddings_decoded = self.embedding_decoder(start_symbols) # batch_size x maxLen x embSize
        inputs = tf.concat([embeddings_decoded, tf.expand_dims(hidden, 1)], 2)

        # unroll
        all_indices = []

        with tf.variable_scope('decoder', reuse = reuse):
            for i in range(maxlen):
                output, state = tf.nn.dynamic_rnn(self.decoder, inputs, initial_state = state, dtype = tf.float32)
                overvocab = self.linear(tf.squeeze(output), tf.constant(False))

                if not sample:
                    indices = tf.expand_dims(tf.argmax(overvocab, axis = 1), 1)
                    probs = tf.reduce_max(overvocab, axis = 1)
                else:
                    # sampling
                    #probs = tf.nn.softmax(overvocab/temp)
                    #indices = tf.multinomial(probs, 1)
                    indices = tf.multinomial(overvocab/temp, 1)

                all_indices.append(indices)

                embedding = self.embedding_decoder(indices)
                inputs = tf.concat([embedding, tf.expand_dims(hidden, 1)], 2)

        max_indices = tf.concat(all_indices, 1)

        return max_indices
