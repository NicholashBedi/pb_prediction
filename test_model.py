import tensorflow as tf
import numpy as np
import format_data_for_ML
from generating_model import get_huber_loss_fn

saved_model = tf.keras.models.load_model('best_model.h5',
                    custom_objects={'custom_huber_loss': get_huber_loss_fn(
                                    delta=0.9)})
training_percent = 0.6
testing_percent = 0.2
model_name = "initial"
data_folder = ""
data_name = "combined_results_2019"
data = format_data_for_ML.get_data(data_folder = data_folder, data_name = data_name,
            training_percent = training_percent, testing_percent = testing_percent)
if (np.shape(data["testing"]["input"])[0] > 0):
    test_loss, test_acc = saved_model.evaluate(x=data["testing"]["input"],
                                        y=data["testing"]["target"])
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)
    predictions = saved_model.predict(x=data["testing"]["input"]).T * data["norm_target"] \
                                        + data["mean_target"]
    abs_error, mse = format_data_for_ML.print_data(data["testing"],
                                norm_input = data["norm_input"],
                                mean_input = data["mean_input"],
                                norm_target = data["norm_target"],
                                mean_target = data["mean_target"],
                                predictions = predictions[0])
    print("ae: {0:4.2f}| mse: {1:4.2f}".format(abs_error, mse))
