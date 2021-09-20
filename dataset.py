import tensorflow as tf
from collections import OrderedDict

class Dataset:
    def __init__(self, batch_size, dataset_path, dataset_info_path, shuffle_buffer_size=0):
        self._batch_size = batch_size
        self._shuffle_buffer_size = shuffle_buffer_size
        self._dataset_path = dataset_path
        with open(dataset_info_path, 'r') as dataset_info:
            self.examples_num = int(dataset_info.readline())
            self.scale_factor = int(dataset_info.readline())
            self.input_info = OrderedDict()
            for line in dataset_info.readlines():
                items = line.split(',')
                self.input_info[items[0]] = [int(dim) for dim in items[1:]]

    def _parse_tf_example(self, example_proto):
        features = dict([(key, tf.io.FixedLenFeature([], tf.string)) for key, _ in self.input_info.items()])
        parsed_features = tf.io.parse_single_example(example_proto, features=features)

        return [tf.reshape(tf.cast(tf.io.decode_raw(parsed_features[key], tf.uint8), tf.float32), value)
                for key, value in self.input_info.items()]
    
    def get_data(self):
        dataset = tf.data.TFRecordDataset(self._dataset_path)
        dataset = dataset.map(self._parse_tf_example,num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=self._shuffle_buffer_size)
        if self._shuffle_buffer_size != 0:
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size,reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size,drop_remainder=True)
        #dataset = tf.cast(dataset, tf.float32) / 255.0
        iterator = iter(dataset)       
        data_batch = iterator.get_next()
        keys = list(self.input_info.keys())
        data_batch = dict([(keys[i], data_batch[i]) for i in range(len(keys))])

        return data_batch, iterator