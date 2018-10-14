import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1
from pprint import pprint
import pandas as pd
import click
import threading

HEIGHT = 224
WIDTH = 224
CHANNELS = 3
checkpoint_file = "resnet_v1_50.ckpt"

import numpy as np

res50_path = 'ckpts/resnet_v1_50.ckpt'
res152_path = 'ckpts/resnet_v1_152.ckpt'
mobilenet_path = 'ckpts/mobilenet_v2_1.0_96.ckpt'

def _get_resnet_input():
    return np.random.randn(1, 224, 224, 3)

def _get_mobilenet_input():
    return np.random.randn(1, 96, 96, 3)


def get_input():
    input_img = np.random.randn(1, HEIGHT, WIDTH, CHANNELS)
    return input_img


def load_tf_sess(mem_frac=0.1, allow_growth=False):
    graph = tf.Graph()
    with graph.as_default():
        img_tensor = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, CHANNELS))
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(
                img_tensor, 1000, is_training=False
            )
        predictions = end_points["predictions"]
        saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = mem_frac
    sess = tf.Session(config=config, graph=graph)
    saver.restore(sess, checkpoint_file)
    return sess, img_tensor, predictions


# def prof(out):
#     profiles = {"start": [], "end": [], "duration": []}
#     for _ in range(1000):
#         start = time.perf_counter()
#         pred_prob = sess.run(predictions, feed_dict={img_tensor: input_img})
#         end = time.perf_counter()
#         profiles["start"].append(start)
#         profiles["end"].append(end)
#         profiles["duration"].append(end - start)
#     df = pd.DataFrame(profiles)
#     df.to_parquet(out + ".pq")


@click.command()
@click.option("--out")
def save(out):
    df.to_parquet(out + ".pq")


if __name__ == "__main__":
    #    threads = [
    #        threading.Thread(target=prof, args=(f"result/thread-{i}",)) for i in range(10)
    #    ]
    #    [t.start() for t in threads]
    #    [t.join() for t in threads]
    prof("smp")
