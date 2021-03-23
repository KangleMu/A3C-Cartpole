# import tensorflow as tf
# import numpy as np
#
# x = tf.constant(3.0)
# with tf.GradientTape() as g:
#   g.watch(x)
#   c = np.clip(np.random.normal(0, 5), -0.5, 0.5)
#   print('c=',c)
#   d = np.clip(np.random.normal(0, 5), -0.5, 0.5)
#   print('d=', d)
#   y = d * c * x * x
# dy_dx = g.gradient(y, x) # Will compute to 6.0
#
# print(dy_dx)

a = list(range(10))
b = list(range(10, 20))

for i, j in zip(a, b):
  print(i, j)