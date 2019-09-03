import tensorflow as tf
from utils.data import *

lang_data = load_data('./data/parallel/wmt14/en_de/spm')
print(lang_data['en'][0][0])

data = tf.data.Dataset.from_tensor_slices(lang_data['en'][0])