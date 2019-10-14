import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
#Why the fuck doesn't this work??
#import tensorflow_probability.edward2.Normal as Normal

if __name__ == "__main__":
    x_train = np.linspace(-3, 3, num=50)
    y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
    x_train = x_train.astype(np.float32).reshape((50, 1))
    y_train = y_train.astype(np.float32).reshape((50, 1))



    W_0 = ed.Normal(loc=tf.zeros([1, 2]), scale=tf.ones([1, 2]))
    W_1 = ed.Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
    b_0 = ed.Normal(loc=tf.zeros(2), scale=tf.ones(2))
    b_1 = ed.Normal(loc=tf.zeros(1), scale=tf.ones(1))

    x = x_train
    y = ed.Normal(loc=tf.matmul(tf.tanh(tf.matmul(x, W_0) + b_0), W_1) + b_1,
            scale=0.1)

    qW_0 = ed.Normal(loc=tf.Variable("qW_0/loc", [1, 2]),
                scale=tf.nn.softplus(tf.Variable("qW_0/scale", [1, 2])))
    qW_1 = ed.Normal(loc=tf.Variable("qW_1/loc", [2, 1]),
                scale=tf.nn.softplus(tf.Variable("qW_1/scale", [2, 1])))
    qb_0 = ed.Normal(loc=tf.Variable("qb_0/loc", [2]),
                scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [2])))
    qb_1 = ed.Normal(loc=tf.Variable("qb_1/loc", [1]),
                scale=tf.nn.softplus(tf.Variable("qb_1/scale", [1])))


    inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                        W_1: qW_1, b_1: qb_1}, data={y: y_train})
    inference.run(n_iter=1000)