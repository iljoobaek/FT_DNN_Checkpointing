import sys
import os

dirname = os.path.dirname(__file__)
slimpath = '/home/droid/mhwork/cmu-tf-object-detector/slim/'
sys.path.append(slimpath)

from tag_layer import tag_all_tf_layers
import tensorflow as tf

if __name__ == "__main__":
    print("Testing overriding of tf slim layers")
    tag_all_tf_layers(True)

    # Init a small Keras model 
    import numpy as np
    a = tf.keras.layers.Input(shape=(32,))
    b = tf.keras.layers.Dense(32, name="testDense32")(a)
    model = tf.keras.models.Model(inputs=a, outputs=b)
    
    x = tf.constant(np.ones((1,32), dtype=np.float32))
    model.predict([x], steps=1)
    import pdb; pdb.set_trace()

