import sys
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
import math
import scipy.io as sio


peoplenum = 32
freq = 128
trials = 40
eggchannels = 32
phychannels = 8
channels = eggchannels + phychannels
base_num = 3
raw_num = 60
labels = 4
data_num = base_num + raw_num
wavedepth = 5
#window_size = 3
feature_map_size_width = 11
feature_map_size_height = 11
divide_hz_num = 64



final_fuse = "concat"

conv_1_shape = '5*5*32'
pool_1_shape = 'None'

conv_2_shape = '6*6*64'
pool_2_shape = 'None'

conv_3_shape = '6*6*128'
pool_3_shape = 'None'

conv_4_shape = '1*1*13'
pool_4_shape = 'None'

window_size = 32
#n_lstm_layers = 2

# lstm full connected parameter
n_hidden_state = 32
print("\nsize of hidden state", n_hidden_state)
n_fc_out = 1024
n_fc_in = 1024

dropout_prob = 0.5
np.random.seed(32)

norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True

#cnn_suffix        =".mat_win_128_cnn_dataset.pkl"
#rnn_suffix        =".mat_win_128_rnn_dataset.pkl"
#label_suffix    =".mat_win_128_labels.pkl"

#data_file    =sys.argv[1]
#arousal_or_valence = sys.argv[2]
#with_or_without = sys.argv[3]

dataset_dir = "./cnn_wavelet_data(hz=64)/"
###load training set

cnn_1st_dataset = np.load(dataset_dir+'1st_baseremoved_5_3.npy')
cnn_1st_label = np.load(dataset_dir+"1st_baseremoved_labelsV_5_3.npy")
cnn_2st_dataset = np.load(dataset_dir+"2st_baseremoved_5_3.npy")
cnn_2st_label = np.load(dataset_dir+"2st_baseremoved_labelsV_5_3.npy")
cnn_3st_dataset = np.load(dataset_dir+"3st_baseremoved_5_3.npy")
cnn_3st_label = np.load(dataset_dir+"3st_baseremoved_labelsV_5_3.npy")

lables_backup = labels
print("cnn_dataset shape before reshape:", np.shape(cnn_3st_dataset))
#cnn_datasets = cnn_datasets.reshape(len(cnn_datasets), window_size, 9,9, 1)
#print("cnn_dataset shape after reshape:", np.shape(cnn_datasets))
#one_hot_labels = np.array(list(pd.get_dummies(labels)))

labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

# shuffle data
index = np.array(range(0, len(labels)))
np.random.shuffle(index)

#cnn_datasets   = cnn_datasets[index]
#rnn_datasets   = rnn_datasets[index]
#labels  = labels[index]

print("**********(" + time.asctime(time.localtime(time.time())) + ") Load and Split dataset End **********\n")
print("**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions Begin: **********\n")

# input parameter
n_input_ele = 32
n_time_step = window_size

input_channel_num = 1
input_height = 11
input_width = 11

n_labels = 2
# training parameter
lambda_loss_amount = 0.5
training_epochs = 10

batch_size = 200


# kernel parameter
kernel_height_1st =5
kernel_width_1st = 5

kernel_height_2nd = 6
kernel_width_2nd = 6

kernel_height_3rd = 6
kernel_width_3rd = 6

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 32

# algorithn parameter
learning_rate = 1e-4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def conv3d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.elu(tf.layers.batch_normalization(conv2d(x, weight, kernel_stride)))

def apply_conv3d(x, filter_height, filter_width, filter_depth, in_channels, out_channels, kernel_stride):
    weight = weight_variable([filter_height, filter_width, filter_depth, in_channels, out_channels])
    bias = bias_variable([out_channels])  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.elu(tf.layers.batch_normalization(conv3d(x, weight, kernel_stride)))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
    fc_weight = weight_variable([x_size, fc_size])
    fc_bias = bias_variable([fc_size])
    return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
    readout_weight = weight_variable([x_size, readout_size])
    readout_bias = bias_variable([readout_size])
    return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")
print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# input placeholder
cnn_in_1st = tf.placeholder(tf.float32, shape=[None, input_width, input_height, input_channel_num], name='cnn_in_1st')
cnn_in_2st = tf.placeholder(tf.float32, shape=[None, input_width, input_height, input_channel_num], name='cnn_in_2st')
cnn_in_3st = tf.placeholder(tf.float32, shape=[None, input_width, input_height, input_channel_num], name='cnn_in_3st')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

###########################################################################################
# add cnn_in_1st parallel to network
###########################################################################################
# first CNN layer
with tf.name_scope("conv_1_1st"):
    conv_1_1st = apply_conv2d(cnn_in_1st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
    print("conv_1 shape:", conv_1_1st.shape)
# second CNN layer
with tf.name_scope("conv_2_1st"):
    conv_2_1st = apply_conv2d(conv_1_1st, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,kernel_stride)
    print("conv_2 shape:", conv_2_1st.shape)
# third CNN layer
with tf.name_scope("conv_3_1st"):
    conv_3_1st = apply_conv2d(conv_2_1st, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,kernel_stride)
    print("conv_3 shape:", conv_3_1st.shape)
# depth concatenate
with tf.name_scope("depth_concatenate_1st"):
    cube_1st = tf.reshape(conv_3_1st,[-1,11,11,conv_channel_num * 4 * window_size])
    print("cube shape:", cube_1st.shape)
# fourth CNN layer
with tf.name_scope("conv_4_1st"):
    conv_4_1st = apply_conv2d(cube_1st, kernel_height_4th, kernel_width_4th, conv_channel_num * 4 * window_size, 13,kernel_stride)
    print("\nconv_4 shape:", conv_4_1st.shape)

# flatten (13*9*9) cube into a 1053 vector.
shape = conv_4_1st.get_shape().as_list()
conv_3_flat_1st = tf.reshape(conv_4_1st, [-1, shape[1] * shape[2] * shape[3]])

cnn_out_fuse_1st = conv_3_flat_1st


###########################################################################################
# add 2nd cnn parallel to network
###########################################################################################
# first CNN layer
with tf.name_scope("conv_1_2st"):
    conv_1_2st = apply_conv2d(cnn_in_2st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
    print("conv_1_2st shape:", conv_1_2st.shape)
# second CNN layer
with tf.name_scope("conv_2_2st"):
    conv_2_2st = apply_conv2d(conv_1_2st, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,kernel_stride)
    print("conv_2_2nd shape:", conv_2_2st.shape)
# third CNN layer
with tf.name_scope("conv_3_2st"):
    conv_3_2st = apply_conv2d(conv_2_2st, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,kernel_stride)
    print("conv_3_2nd shape:", conv_3_2st.shape)
# depth concatenate
with tf.name_scope("depth_concatenate_2st"):
    cube_2st = tf.reshape(conv_3_2st,[-1,11,11,conv_channel_num * 4 * window_size])
    print("cube shape:", cube_2st.shape)
# fourth CNN layer
with tf.name_scope("conv_4"):
    conv_4_2st = apply_conv2d(cube_2st, kernel_height_4th, kernel_width_4th, conv_channel_num * 4 * window_size, 13,kernel_stride)
    print("\nconv_4_2st shape:", conv_4_2st.shape)

# flatten (13*9*9) cube into a 1053 vector.
shape = conv_4_2st.get_shape().as_list()
conv_3_flat_2st = tf.reshape(conv_4_2st, [-1, shape[1] * shape[2] * shape[3]])

cnn_out_fuse_2st = conv_3_flat_2st


###########################################################################################
# add 3rd cnn parallel to network
###########################################################################################
# first CNN layer
with tf.name_scope("conv_1_3st"):
    conv_1_3st = apply_conv2d(cnn_in_3st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
    print("conv_1_3st shape:", conv_1_3st.shape)
# second CNN layer
with tf.name_scope("conv_2_3st"):
    conv_2_3st = apply_conv2d(conv_1_3st, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,kernel_stride)
    print("conv_2_3st shape:", conv_2_3st.shape)
# third CNN layer
with tf.name_scope("conv_3_3st"):
    conv_3_3st = apply_conv2d(conv_2_3st, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,kernel_stride)
    print("conv_3_3st shape:", conv_3_3st.shape)
# depth concatenate
with tf.name_scope("depth_concatenate_3st"):
    cube_3st = tf.reshape(conv_3_3st,[-1,11,11,conv_channel_num * 4 * window_size])
    print("cube shape_3st:", cube_3st.shape)
# fourth CNN layer
with tf.name_scope("conv_4_3st"):
    conv_4_3st = apply_conv2d(cube_3st, kernel_height_4th, kernel_width_4th, conv_channel_num * 4 * window_size, 13,kernel_stride)
    print("\nconv_4 shape:", conv_4_3st.shape)

# flatten (13*9*9) cube into a 1053 vector.
shape_3st = conv_4_3st.get_shape().as_list()
conv_3_flat_3st = tf.reshape(conv_4_3st, [-1, shape[1] * shape[2] * shape[3]])

cnn_out_fuse_3st = conv_3_flat_3st




###########################################################################################
# fuse parallel 3 cnn
###########################################################################################
print("final fuse method: concat")
fuse_cnn = tf.concat([cnn_out_fuse_1st, cnn_out_fuse_2st, cnn_out_fuse_3st], axis=1)

fuse_cnn_shape = fuse_cnn.get_shape().as_list()
print("\nfuse_cnn:", fuse_cnn_shape)
# readout layer
y_ = apply_readout(fuse_cnn, fuse_cnn_shape[1], n_labels)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
    tf.summary.scalar('cost_with_L2',cost)
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')
    tf.summary.scalar('cost',cost)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy',accuracy)

#print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

#print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

merged = tf.summary.merge_all()


result = np.zeros((peoplenum))
result_mean=0


# .dat?ŒŒ?¼?„ ë¶ˆëŸ¬?˜¨?‹¤ (ì´? 32ëª…ì˜ ì°¸ê???ž)
for p in range(peoplenum):
    cnn_1st_each_dataset = cnn_1st_dataset[p]
    cnn_2st_each_dataset = cnn_2st_dataset[p]
    cnn_3st_each_dataset = cnn_3st_dataset[p]
    labels_each = cnn_1st_label[p]
    print("test",cnn_1st_each_dataset.shape)

    cnn_each_dataset = np.zeros((trials, int(raw_num/3), eggchannels, 3, feature_map_size_width, feature_map_size_height))

    for t in range(40):
        for w in range(20):
            for c in range(32):
                cnn_each_dataset[t][w][c][0] = cnn_1st_each_dataset[t][w][c]
                cnn_each_dataset[t][w][c][1] = cnn_1st_each_dataset[t][w][c]
                cnn_each_dataset[t][w][c][2] = cnn_1st_each_dataset[t][w][c]


    cnn_1st_each_dataset = cnn_1st_each_dataset.reshape((trials*int(raw_num/3), eggchannels, feature_map_size_width, feature_map_size_height))
    cnn_2st_each_dataset = cnn_2st_each_dataset.reshape((trials*int(raw_num/3), eggchannels, feature_map_size_width, feature_map_size_height))
    cnn_3st_each_dataset = cnn_3st_each_dataset.reshape((trials*int(raw_num/3), eggchannels, feature_map_size_width, feature_map_size_height))
    print(cnn_1st_each_dataset[0][0])
    
    labels_each = labels_each.reshape((trials*int(raw_num/3)))
    one_hot_label = np.zeros((trials*int(raw_num/3),2))
    for fs in range(800):
        if labels_each[fs] >= 1 :
            one_hot_label[fs][1] = 1
        else:
            one_hot_label[fs][0] = 1


    #one_hot_labels = np.transpose(one_hot_labels)
    #labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)


    np.random.seed(1671)    #?ž¬?˜„?„ ?œ„?•œ ?„¤? •
  
    #print(Matrixs2)

    NB_EPOCH = 10
    BATCH_SIZE = 200
    VERBOSE = 1
    VALIDATION_SPLIT = 0.1
    #IMG_ROWS, IMG_COLS = eggchannels ,3*(wavedepth+1)     #?ž…? ¥ ?´ë¯¸ì?? ì°¨ì›
    NB_CLASSES = 2          #ì¶œë ¥ ê°œìˆ˜ = ?ˆ«?ž?˜ ê°œìˆ˜

    print("wow")
    #one_hot_labels = one_hot_labels.transpose()
    print(one_hot_label.shape)
    # print(one_hot_label[0:100])
    #?°?´?„° ë¬´ìž‘?œ„ë¡? ?„žê³?, ?•™?Šµ?°?´?„°??? ?…Œ?Š¤?Š¸ ?°?´?„°ë¡? ?‚˜?ˆ”
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, label_train, label_test = train_test_split(cnn_1st_each_dataset, cnn_2st_each_dataset, cnn_3st_each_dataset, one_hot_label, test_size=VALIDATION_SPLIT)


    #print(X_test)

    #? •ê·œí™”

    # X_train /= 255
    # X_test /= 255
    #print(X_train[6], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_test.shape[1], 'test samples')


    batch_num_per_epoch = math.floor(X_train.shape[0]/batch_size)+ 1

    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(X_test.shape[0]/batch_size)+ 1
   # saver = tf.train.Saver(max_to_keep=1)
    #W = tf.Variable(tf.random_normal([1]), name='weight')
    #b = tf.Variable(tf.random_normal([1]), name='bias')

    with tf.Session(config=config) as session:
        #train_writer.add_graph(session.graph)
        count_cost = 0
        train_count_accuracy = 0
        test_count_accuracy = 0

        session.run(tf.global_variables_initializer())
        train_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)
        for epoch in range(training_epochs):
           # print("learning rate: ",learning_rate)
            cost_history = np.zeros(shape=[0], dtype=float)
            for b in range(batch_num_per_epoch):
                start = b* batch_size
                if (b+1)*batch_size>Y_train.shape[0]:
                    offset = Y_train.shape[0] % batch_size
                else:
                    offset = batch_size
                
                #print(X_train)
                # print("start:",start,"end:",start+offset)
                cnn_1st_batch = X_train[start:(start + offset), :, :, :]
                cnn_1st_batch = cnn_1st_batch.reshape(len(cnn_1st_batch) * window_size, 11, 11, 1)
                cnn_2st_batch = Y_train[start:(start + offset), :, :, :]
                cnn_2st_batch = cnn_2st_batch.reshape(len(cnn_2st_batch) * window_size, 11, 11, 1)
                cnn_3st_batch = Z_train[start:(start + offset), :, :, :]
                cnn_3st_batch = cnn_3st_batch.reshape(len(cnn_3st_batch) * window_size, 11, 11, 1)
                batch_y = label_train[start:(start + offset), :]
                _ , c = session.run([optimizer, cost],
                                   feed_dict={cnn_in_1st: cnn_1st_batch, cnn_in_2st: cnn_2st_batch, cnn_in_3st: cnn_3st_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                              phase_train: True})
                cost_history = np.append(cost_history, c)
                count_cost += 1
            if (epoch % 1 == 0):
                train_accuracy = np.zeros(shape=[0], dtype=float)
                test_accuracy = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)
                train_loss = np.zeros(shape=[0], dtype=float)

                for i in range(train_accuracy_batch_num):
                    start = i* batch_size
                    if (i+1)*batch_size>Y_train.shape[0]:
                        offset = label_train.shape[0] % batch_size
                    else:
                        offset = batch_size
                    train_cnn_1st_batch = X_train[start:(start + offset), :, :, :]
                    train_cnn_1st_batch = train_cnn_1st_batch.reshape(len(train_cnn_1st_batch) * window_size, 11, 11, 1)
                    train_cnn_2st_batch = Y_train[start:(start + offset), :, :, :]
                    train_cnn_2st_batch = train_cnn_2st_batch.reshape(len(train_cnn_2st_batch) * window_size, 11, 11, 1)
                    train_cnn_3st_batch = Z_train[start:(start + offset), :, :, :]
                    train_cnn_3st_batch = train_cnn_1st_batch.reshape(len(train_cnn_3st_batch) * window_size, 11, 11, 1)


                    train_batch_y = label_train[start:(start + offset), :]

                    tf_summary,train_a, train_c = session.run([merged,accuracy, cost],
                                                   feed_dict={cnn_in_1st: train_cnn_1st_batch, cnn_in_2st: train_cnn_2st_batch,
                                                              cnn_in_3st: train_cnn_3st_batch,
                                                              Y: train_batch_y, keep_prob: 1.0, phase_train: False})
                    #train_writer.add_summary(tf_summary,train_count_accuracy)
                    train_loss = np.append(train_loss, train_c)
                    train_accuracy = np.append(train_accuracy, train_a)
                    train_count_accuracy += 1
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                      np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
                train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
                train_loss_save = np.append(train_loss_save, np.mean(train_loss))

                if(np.mean(train_accuracy)<0.8):
                    learning_rate=1e-4
                elif(0.8<np.mean(train_accuracy)<0.85):
                    learning_rate=5e-5
                elif(0.85<np.mean(train_accuracy)):
                    learning_rate=5e-6

                for j in range(test_accuracy_batch_num):
                    start = j * batch_size
                    # print(start)
                    if (j+1)*batch_size>Y_test.shape[0]:
                        offset = Y_test.shape[0] % batch_size
                    else:
                        offset = batch_size
                    test_cnn_1st_batch = X_test[start:(start + offset), :, :, :]
                    test_cnn_1st_batch = test_cnn_1st_batch.reshape(len(test_cnn_1st_batch) * window_size, 11, 11, 1)
                    test_cnn_2st_batch = Y_test[start:(start + offset), :, :, :]
                    test_cnn_2st_batch = test_cnn_2st_batch.reshape(len(test_cnn_2st_batch) * window_size, 11, 11, 1)
                    test_cnn_3st_batch = Z_test[start:(start + offset), :, :, :]
                    test_cnn_3st_batch = test_cnn_3st_batch.reshape(len(test_cnn_3st_batch) * window_size, 11, 11, 1)

                    test_batch_y = label_test[start:(start + offset), :]

                    tf_test_summary,test_a, test_c = session.run([merged,accuracy, cost],
                                                 feed_dict={cnn_in_1st: test_cnn_1st_batch, cnn_in_2st: test_cnn_2st_batch,
                                                            cnn_in_3st: test_cnn_3st_batch, Y: test_batch_y,
                                                            keep_prob: 1.0, phase_train: False})
                    #test_writer.add_summary(tf_test_summary,test_count_accuracy)
                    test_accuracy = np.append(test_accuracy, test_a)
                    test_loss = np.append(test_loss, test_c)
                    test_count_accuracy += 1 
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                      np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
                test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
                test_loss_save = np.append(test_loss_save, np.mean(test_loss))

            # learning_rate decay
            if(np.mean(train_accuracy)<0.9):
                learning_rate=1e-4
            elif(0.9<np.mean(train_accuracy)<0.95):
                learning_rate=5e-5
            elif(0.99<np.mean(train_accuracy)):
                learning_rate=5e-6

        test_accuracy = np.zeros(shape=[0], dtype=float)
        test_loss = np.zeros(shape=[0], dtype=float)
        test_pred = np.zeros(shape=[0], dtype=float)
        test_true = np.zeros(shape=[0, 2], dtype=float)
        test_posi = np.zeros(shape=[0, 2], dtype=float)
        for k in range(test_accuracy_batch_num):
            start = k * batch_size
            if (k+1)*batch_size>Y_test.shape[0]:
                offset = label_test.shape[0] % batch_size
            else:
                offset = batch_size
            test_cnn_1st_batch = X_test[start:(start + offset), :, :, :]
            test_cnn_1st_batch = test_cnn_1st_batch.reshape(len(test_cnn_1st_batch) * window_size, 11, 11, 1)
            test_cnn_2st_batch = Y_test[start:(start + offset), :, :, :]
            test_cnn_2st_batch = test_cnn_2st_batch.reshape(len(test_cnn_2st_batch) * window_size, 11, 11, 1)
            test_cnn_3st_batch = Z_test[start:(start + offset), :, :, :]
            test_cnn_3st_batch = test_cnn_3st_batch.reshape(len(test_cnn_3st_batch) * window_size, 11, 11, 1)
            test_batch_y = label_test[start:(start + offset), :]

            test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi],
                                                         feed_dict={cnn_in_1st: test_cnn_1st_batch, cnn_in_2st: test_cnn_2st_batch,
                                                                    cnn_in_3st: test_cnn_3st_batch,
                                                                    Y: test_batch_y, keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_accuracy = np.append(test_accuracy, test_a)
            test_loss = np.append(test_loss, test_c)
            test_pred = np.append(test_pred, test_p)
            test_true = np.vstack([test_true, test_t])
            test_posi = np.vstack([test_posi, test_r])
        test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
        test_true_list = tf.argmax(test_true, 1).eval()
        # recall
        test_recall = recall_score(test_true, test_pred_1_hot, average=None)
        # precision
        test_precision = precision_score(test_true, test_pred_1_hot, average=None)
        # f1 score
        test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
        # confusion matrix
        # confusion_matrix = confusion_matrix(test_true_list, test_pred)
        # print("********************recall:", test_recall)
        # print("*****************precision:", test_precision)
        # print("******************f1_score:", test_f1)
        # print("**********confusion_matrix:\n", confusion_matrix)

        print((p+1),"people (" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test Accuracy: ", np.mean(test_accuracy))

        model_dir = "./save_model/"
        saver.save(session, model_dir+'cnn3d_model%02d_BaseRemoved_valence.dat'%(p+1), global_step=0)
        # result = pd.DataFrame(
        #     {'epoch': range(1, epoch + 2), "train_accuracy": train_accuracy_save, "test_accuracy": test_accuracy_save,
        #      "train_loss": train_loss_save, "test_loss": test_loss_save})

        result[p] = np.mean(test_accuracy)
        # print(result)




# plt.figure(figsize=(20, 10))  
# plt.plot(result, color='red', linestyle='dashed', marker='o',  
#          markerfacecolor='blue', markersize=10)
# plt.title('accuracy on person')  
# plt.xlabel('peoplenum')  
# plt.ylabel('accuracyr')  
# plt.show()

for i in range(32) :
    result_mean += result[i]
    print(str(i+1) + "person : " + str(result[i]) + "\n")


result_mean = result_mean/32

print("result_mean:", result_mean)

print("finish!")
