import tensorflow as tf

def demean(x, axis=1):
    return x - tf.reduce_mean(x, axis=axis, keepdims=True)

def l2_normalize(x, eps=1e-27, axis=1):
    l2_norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True))
    return x / (eps + l2_norm)

def l2_distance(x, y, axis=[1, 2]):
    x = demean(x)
    y = demean(y)

    distance = tf.sqrt(tf.reduce_mean(tf.square(x - y), axis=axis))
    return distance