import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

class Dataset(object):
    def __init__(self,  train_path, test_path, oneHot = False):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.test_id = self.test['project_id']
        self.pos_train = self.train.loc[self.train['final_status'] == 1].copy()
        self.neg_train = self.train.loc[self.train['final_status'] == 0].copy()

        self.pos_y_ = self.pos_train['final_status']
        self.neg_y_ = self.neg_train['final_status']


        self.pos_train.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline','final_status','backers_count'], inplace=True, axis=1)
        self.neg_train.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline','final_status','backers_count'], inplace=True, axis=1)
        self.test.drop(['project_id', 'name', 'desc', 'keywords', 'country', 'currency', 'state_changed_at', 'created_at', 'launched_at', 'deadline'], inplace=True, axis=1)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #self.pos_train = min_max_scaler.fit_transform(self.pos_train)
        #self.neg_train = min_max_scaler.fit_transform(self.neg_train)
        #self.test = min_max_scaler.fit_transform(self.test)

        self.pos_train = self.pos_train.as_matrix()
        self.neg_train = self.neg_train.as_matrix()

        self.pos_index_in_epoch = 0
        self.neg_index_in_epoch = 0
        self.pos_num_examples = self.pos_train.shape[0]
        self.neg_num_examples = self.neg_train.shape[0]
        if oneHot == True:
            ohm_p =  np.zeros((self.pos_y_.shape[0], 2))
            ohm_p[np.arange(self.pos_y_.shape[0]), 1] = 1
            self.pos_y_ = ohm_p
            ohm_n =  np.zeros((self.neg_y_.shape[0], 2))
            ohm_n[np.arange(self.neg_y_.shape[0]), 0] = 1
            self.neg_y_ = ohm_n

    def train(self):
        return self.train

    def labels(self):
        return self.pos_y_

    def test(self):
        return self.test

    def test_id(self):
        return self.test_id

    def next_batch(self, batch_size):
        pos_start = self.pos_index_in_epoch
        neg_start = self.neg_index_in_epoch
        pos_ratio = float(self.pos_train.shape[0])/float(self.train.shape[0])
        neg_ratio = float(self.neg_train.shape[0])/float(self.train.shape[0])
        self.pos_index_in_epoch += int(pos_ratio*batch_size)
        self.neg_index_in_epoch += int(neg_ratio*batch_size)
        if self.pos_index_in_epoch > self.pos_num_examples:
            perm = np.arange(self.pos_num_examples)
            np.random.shuffle(perm)
            self.pos_train = self.pos_train[perm]
            self.pos_y_ = self.pos_y_[perm]

            pos_start = 0
            self.pos_index_in_epoch = int(pos_ratio*batch_size)

        if self.neg_index_in_epoch > self.neg_num_examples:
            perm = np.arange(self.neg_num_examples)
            np.random.shuffle(perm)
            self.neg_train = self.neg_train[perm]
            self.neg_y_ = self.neg_y_[perm]

            neg_start = 0
            self.neg_index_in_epoch = int(neg_ratio*batch_size)

        pos_end = self.pos_index_in_epoch
        neg_end = self.neg_index_in_epoch
        batch_pos = self.pos_train[pos_start:pos_end]
        batch_neg = self.neg_train[neg_start:neg_end]
        batch = np.concatenate((batch_pos, batch_neg), axis=0)
        batch_y_pos = self.pos_y_[pos_start:pos_end]
        batch_y_neg = self.neg_y_[neg_start:neg_end]
        batch_y = np.concatenate((batch_y_pos, batch_y_neg), axis=0)
        #np.concatenate((batch,self.neg_train[neg_start:neg_end]), axis = 0)
        #np.concatenate((batch_y, self.neg_y_[neg_start:neg_end]), axis = 0)
        perm = np.arange(len(batch))
        np.random.shuffle(perm)
        return batch[perm], batch_y[perm]


# Load Data
data = Dataset('/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/train_clean.csv',
                            '/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/test_clean.csv',
                            oneHot = True)
#print (data.next_batch(100))
# Scale the Data
#min_max_scaler = preprocessing.MinMaxScaler()
#train_scaled = min_max_scaler.fit_transform(train)
#train = pd.DataFrame(train_scaled)

#test_scaled = min_max_scaler.fit_transform(test)
#test = pd.DataFrame(test_scaled)
print(data.labels)

INPUT_FEATURES = data.train.shape[1]
INPUT_LENGTH = data.train.shape[0]
print (INPUT_FEATURES, INPUT_LENGTH)

# Model Functions
def init_weights(shape):
    weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


x = tf.placeholder(tf.float32, [None, 44])
y_ = tf.placeholder(tf.float32, [None, 2])
x = tf.reshape(x, [-1, 44])

W1 = init_weights([44, 25])
B1 = init_bias([25])

o1 = tf.nn.sigmoid(tf.matmul(x, W1) + B1)

W2 = init_weights([25, 8])
B2 = init_bias([8])

o2 = tf.nn.sigmoid(tf.matmul(o1, W2) + B2)


W3 = init_weights([8, 2])
B3 = init_bias([2])

o3 = tf.matmul(o2, W3) + B3

cross_entropy_cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=o3))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_cost_function)
prediction = tf.argmax(o3, 1)
correct_prediction = tf.equal(tf.argmax(o3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = data.next_batch(100)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  pred = prediction.eval(feed_dict={x: batch[0]})

out_column = ['final_status']
sub = pd.DataFrame(data=pred, columns=out_column)
sub['project_id'] = data.test_id
sub = sub[['project_id', 'final_status']]
sub.head()
sub.to_csv("/Users/namakilam/workspace/Kickstart_ML/3149def2-5-datafiles/ann2Starter.csv",index = False)
