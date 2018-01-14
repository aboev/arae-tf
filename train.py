import tensorflow as tf
from utils import Corpus, batchify, get_string
from utils_debug import sentence_to_batch, get_string
import time
import numpy as np
from layers import MLP_D, MLP_G, Seq2SeqLayer, LeakyReluActivation, LinearLayer, NormalInitializer, RandomUniformInitializer
import random

HOME_PATH = '/home/alex/workbooks/ipython/arae-test/'
scope_autoencoder = 'autoencoder'
scope_critic = 'critic'
scope_generator = 'generator'
class Args:
    batch_size = 64
    temp = 1
    clip = 1
    log_interval = 200
    z_size = 100
    epochs = 15
    vocab_size = 11000
    maxlen = 30
    data_path = HOME_PATH + 'data/'
    emsize = 300
    nhidden = 300
    nlayers = 1
    noise_radius = 0.2
    noise_anneal = 0.995
    hidden_init = True
    dropout = 0.0
    niters_ae = 1
    lr_ae = 1.0
    log_interval = 50
    
    gan_clamp = 0.01
    niters_gan_d = 5
    niters_gan_g = 1
    gan_toenc = -0.01
    lr_gan_g = 5e-05
    lr_gan_d = 1e-05
    beta1 = 0.9
    arch_d = '300-300'
    arch_g = '300-300'
    niters_gan_schedule = '2-4-6'
args = Args()

corpus = Corpus(args.data_path, maxlen=args.maxlen, vocab_size=args.vocab_size, lowercase=True)

# Prepare data
ntokens = len(corpus.dictionary.word2idx)
args.ntokens = ntokens

test_data = batchify(corpus.test, args.batch_size, args.maxlen, shuffle=False)
train_data = batchify(corpus.train, args.batch_size, args.maxlen, shuffle=False)

def cost(output, target):
  # Compute cross entropy for each frame.
  cross_entropy = target * tf.log(tf.nn.softmax(output))
  cross_entropy = -tf.reduce_sum(cross_entropy, 2)
  mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
  cross_entropy *= mask
  # Average over actual sequence lengths.
  cross_entropy = tf.reduce_sum(cross_entropy, 1)
  cross_entropy /= tf.reduce_sum(mask, 1)
  return tf.reduce_mean(cross_entropy)

tf.reset_default_graph()

# Build graph
fixed_noise = tf.Variable(tf.random_normal(shape = (args.batch_size, args.z_size), mean=0.0, stddev=1.0, dtype=tf.float32))

with tf.variable_scope(scope_autoencoder):
    autoencoder = Seq2SeqLayer(batch_size = args.batch_size, emsize=args.emsize, nhidden=args.nhidden, ntokens=ntokens, nlayers=args.nlayers, noise_radius=args.noise_radius, hidden_init=args.hidden_init, dropout=args.dropout)
with tf.variable_scope(scope_critic):
    gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
with tf.variable_scope(scope_generator):
    gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)

source = tf.placeholder(tf.int64, [None, args.maxlen], name = 'source') # batch_size x maxLen
target = tf.placeholder(tf.int64, [None, args.maxlen], name = 'target') # batch_size x maxLen
lengths = tf.placeholder(tf.int64, [None], name = 'lengths')
hidden_input = tf.placeholder(tf.float32, [None, args.nhidden], name = 'hidden_input')
is_train = tf.placeholder(tf.bool, name='is_train')

# Create sentence length mask over padding
mask = tf.greater(target, 0)
masked_target = tf.boolean_mask(target, mask)

# examples x ntokens
output_mask = tf.tile(tf.expand_dims(mask, 1), [tf.shape(mask)[0], ntokens, 1])

# output: batch_size x maxLen x nHidden
#output = autoencoder(source, lengths, noise=True)
output = autoencoder(source, lengths, noise=True)
#output_test = autoencoder(source, lengths, noise=False, reuse=True)


# output: batch_size x maxLen x ntokens
#output_logits = tf.contrib.layers.fully_connected(output, ntokens, activation_fn=None) / args.temp
output_logits = output / args.temp
#output_logits_test = tf.contrib.layers.fully_connected(output_test, ntokens, activation_fn=None) / args.temp
# output: batch_size x maxLen
output_predictions = tf.argmax(output_logits, 2)

# Loss/Accuracy for AE
loss = cost(output_logits, tf.one_hot(target, depth=args.ntokens, dtype=tf.float32))
#stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(target, depth=args.ntokens, dtype=tf.float32), logits=output_logits)
#loss = tf.reduce_mean(stepwise_cross_entropy)
pred_idx = tf.argmax(output_logits, 2)
#pred_idx_test = tf.argmax(output_logits_test, 2)
mask = tf.logical_not(tf.equal(pred_idx, tf.constant(0, dtype = tf.int64)))

accuracy = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred_idx, target), mask), tf.float32), 1)
accuracy /= tf.cast(lengths, tf.float32)
accuracy = tf.reduce_mean(accuracy)

try:
    @tf.RegisterGradient("CustomGradOne")
    def constant_grad_one(unused_op, grad):
      return tf.ones_like(grad)

    @tf.RegisterGradient("CustomGradMinusOne")
    def constant_grad_minus_one(unused_op, grad):
      return -1.0 * tf.ones_like(grad)
except:
    print("Gradient hooks already registered")
  
# Generator
noise = tf.random_normal(shape = (args.batch_size, args.z_size), mean = 0, stddev = 1)
fake_hidden = gan_gen(noise)

with tf.get_default_graph().gradient_override_map({"Identity": "CustomGradOne"}):
    err_G = gan_disc(fake_hidden, reduce_mean = True)

# Discriminator/Critic
gan_disc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_critic)
for p in gan_disc_params:
    tf.clip_by_value(p, -args.gan_clamp, args.gan_clamp)

real_hidden = autoencoder(source, lengths, noise=False, encode_only=True, reuse = True)
#real_hidden_noise = autoencoder(source, lengths, noise=True, encode_only=True, reuse = True)

with tf.get_default_graph().gradient_override_map({"Identity": "CustomGradOne"}):
    err_D_real = gan_disc(real_hidden, reduce_mean = True)

with tf.get_default_graph().gradient_override_map({"Identity": "CustomGradMinusOne"}):
    err_D_fake = gan_disc(fake_hidden, reduce_mean = True)

autoencoder_params = tf.get_collection(scope_autoencoder)
for p in autoencoder_params:
    tf.clip_by_value(p, -args.clip, args.clip)

# Optimization
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if scope_generator in var.name]
d_vars = [var for var in t_vars if scope_critic in var.name]

bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(bn_update_ops):

    # Optimizer AE

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr_ae)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [((grad if grad == None else tf.clip_by_value(grad, -args.clip, args.clip)), var) for grad, var in gvs]
    train_op_ae = optimizer.apply_gradients(capped_gvs)

    # Optimizer GAN

    train_op_g = tf.train.AdamOptimizer(learning_rate = args.lr_gan_g, beta1 = args.beta1, beta2 = 0.999).minimize(err_G, var_list=g_vars)
    train_op_d_real = tf.train.AdamOptimizer(learning_rate = args.lr_gan_d, beta1 = args.beta1, beta2 = 0.999).minimize(err_D_real, var_list=d_vars)
    train_op_d_fake = tf.train.AdamOptimizer(learning_rate = args.lr_gan_d, beta1 = args.beta1, beta2 = 0.999).minimize(err_D_fake, var_list=d_vars)

# Evaluate
max_indices = autoencoder.generate(fake_hidden, args.maxlen, sample=False, reuse = True)

max_indices_hidden = autoencoder.generate(hidden_input, args.maxlen, sample=False, reuse = True)

writer = tf.summary.FileWriter(logdir='/tmp/tensorboard', graph=tf.get_default_graph())
writer.flush()

# Train

saver = tf.train.Saver()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.66)

if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    init.run()
    
    for epoch in range(1, args.epochs+1):
        
        if epoch in gan_schedule:
            niter_gan += 1
        
        total_loss_ae = 0
        niter = 0
        niter_global = 1
        
        start_time = time.time()

        # loop through all batches in training data
        while niter < len(train_data):
            for i in range(args.niters_ae):
                saver.save(sess, '/data/tf-models/arae/arae-tf-120118-iter', global_step=i)
                
                if niter == len(train_data):
                    break  # end of epoch
                source_batch, target_batch, lengths_batch = train_data[niter]
                _, loss_val, acc_val = sess.run([train_op_ae, loss, accuracy], {source: source_batch, target: target_batch, lengths: lengths_batch, is_train: True})
                
                total_loss_ae += loss_val
                elapsed = time.time() - start_time
                
                if niter % args.log_interval == 0 and niter > 0:
                    cur_loss = total_loss_ae / args.log_interval
                    total_loss_ae = 0
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | acc {:8.2f}'.format(epoch, niter, len(train_data), elapsed * 1000 / niter, cur_loss, acc_val))

                niter += 1  

            for k in range(niter_gan):
                for i in range(args.niters_gan_d):
                    source_batch, target_batch, lengths_batch = train_data[random.randint(0, len(train_data)-1)]
                    _, _, _, err_D_fake_val, err_D_real_val = sess.run([train_op_d_real, train_op_d_fake, train_op_ae, err_D_fake, err_D_real], {source: source_batch, target: target_batch, lengths: lengths_batch, is_train: True})
                    
                for i in range(args.niters_gan_g):
                    source_batch, target_batch, lengths_batch = train_data[random.randint(0, len(train_data)-1)]
                    _, G_loss_val = sess.run([train_op_g, err_G], {source: source_batch, target: target_batch, lengths: lengths_batch, is_train: True})
                    
            niter_global += 1
            
            if niter_global % 100 == 0:
                autoencoder.noise_radius = autoencoder.noise_radius*args.noise_anneal
                
                print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f Loss_D_fake: %.8f) Loss_G: %.8f' % (epoch, args.epochs, niter, len(train_data), err_D_fake_val - err_D_real_val, err_D_real_val, err_D_fake_val, G_loss_val))
                
                if niter_global % 3000 == 0:
                    source_batch, target_batch, lengths_batch = train_data[random.randint(0, len(train_data)-1)]
                    max_ind = sess.run([max_indices], {source: source_batch, target: target_batch, lengths: lengths_batch, is_train: True})
                    print('Evaluating generator: %s' % get_string(max_ind[0], corpus))

