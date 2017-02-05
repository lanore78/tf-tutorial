import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = np.transpose(xy[:-1]) # slicing ndarray, 입력값만 자른, index에 -1을 한 이유는 입력값의 개수에 상관없이 Y값 빼고 Slicing하려고.
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder('float32', name="X")
Y = tf.placeholder('float32', name="Y")

y_hist = tf.summary.histogram("Y", Y)

with tf.name_scope("Weight") as scope:
    W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name="Weight1")
    W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name="Weight2")
    tf.summary.histogram("weight1", W1)
    tf.summary.histogram("weight2", W2)

with tf.name_scope("Bias") as scope:
    b1 = tf.Variable(tf.zeros([2]), name="Bias1")
    b2 = tf.Variable(tf.zeros([1]), name="Bias2")
    tf.summary.histogram("bias1", b1)
    tf.summary.histogram("bias2", b2)

#h = tf.matmul(W, X)
with tf.name_scope("Layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("Layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2)+b2)

with tf.name_scope("Cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("Cost", cost)


rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/minst_logs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(10000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            # b1과 b2는 출력 생략. 한 줄에 출력하기 위해 reshape 사용
            #r1, (r2, r3) = sess.run(merged, cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2])
            #print('{:5} {:10.8f} {} {}'.format(step+1, r1, np.reshape(r2, (1,4)), np.reshape(r3, (1,2))))
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    print('-'*50)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    #Calculate accuraty
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    param = [hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy]
    result = sess.run(param, feed_dict={X:x_data, Y:y_data})

    print(*result[0])
    print(*result[1])
    print(*result[2])
    print( result[-1])
    print('Accuracy :', accuracy.eval({X:x_data, Y:y_data}))
