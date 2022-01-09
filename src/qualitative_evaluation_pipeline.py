# %%
from thesis_data_readers import VolvoIncidentsReader, RequestForPaymentLogReader, BPIC12LogReader
from tensorflow import keras
import tensorflow as tf
import pathlib
import os
import io
from thesis_generators.helper import constants
from thesis_predictors.helper import metrics
# %%
reader = RequestForPaymentLogReader().init_data()
data = reader.get_dataset()
reader._get_example_trace_subset()
# %%
model_dirs = [x for x in constants.PATH_MODEL.iterdir() if x.is_dir()]
# %%
loss_fn = metrics.SparseCrossEntropyLoss()
metric = metrics.SparseAccuracyMetric()

model = keras.models.load_model(model_dirs[2], custom_objects={'SparseCrossEntropyLoss': loss_fn, 'SparseAccuracyMetric': metric})
# %%
predictions = model.predict(data)
predictions.shape
# %%
predictions.argmax(axis=-1)
# %%
predictions[0].shape
# %%
predictions[0].argmax(axis=-1)


# %%
def construct_fake_input(example, reader):
    # reader.get_dataset()._structure[0]
    # reader.get_dataset()._structure[0][0].shape[1:]
    structure = reader.get_dataset()._structure[0]

    # shape = self.reader.data_container.shape[1:]
    shape_batch = (1, )

    # second_shape = (reader.data_container.shape[1], reader.data_container.shape[2]-1)
    return (
        tf.constant(example, dtype=tf.float32),
        tf.zeros(shape_batch + structure[1].shape[1:]),
        tf.zeros(shape_batch + structure[2].shape[1:]),
        tf.zeros(shape_batch + structure[3].shape[1:]),
    )


example = next(iter(data.take(1)))[0][0]
print(model.predict(construct_fake_input(example, reader))[0].argmax(-1))
print(predictions[0].argmax(axis=-1))
# %%

# %%
