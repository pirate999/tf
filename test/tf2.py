import tensorflow as tf

A = tf.constant([[1], [2], [3], [4]])
B = tf.constant([[1, 2], [3, 4], [5, 6]])

print("A:",A.shape)
print("B:",B.shape)


with tf.Session() as sess:
  #tf.argmax() 0:col, 1:row
  #find the max number in every col in A matirx
  print(sess.run(tf.argmax(A, 0)))
  
  #find the max number in every row in A matirx
  print(sess.run(tf.argmax(A, 1)))
  
  #find the max number in every col in B matirx
  print(sess.run(tf.argmax(B, 0)))
  
  #find the max number in every row in B matirx
  print(sess.run(tf.argmax(B, 1)))

