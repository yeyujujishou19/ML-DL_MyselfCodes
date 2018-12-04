import tensorflow as tf
import numpy as np


# Notice: init = tf.global_variables_initializer() is unnecessary
with tf.Session() as sess:
    # 提取变量
    new_saver = tf.train.import_meta_graph('./folder_for_nn/model.ckpt-100.meta')
    new_saver.restore(sess, "./folder_for_nn/model.ckpt-100")
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    y = tf.get_collection("pred_network")[0]
    print("109的预测值是:", sess.run(y, feed_dict={x: [[109]]}))



