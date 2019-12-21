import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import format_data_for_ML
import athletes_PBS

model_name = "initial"
data_folder = ""
data_name = "combined_results_2019"

training_percent = 0.8
testing_percent = 0.1
# validation_percent = 1 - training_percent - testing_percent

data = format_data_for_ML.get_data(data_folder = data_folder, data_name = data_name,
            training_percent = training_percent, testing_percent = testing_percent)


reg_lamb = 0.001

model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation="relu", input_shape=(6,),
                        kernel_regularizer=tf.keras.regularizers.l2(reg_lamb),
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                                            mean=0.0, stddev=0.05, seed=None)),
    tf.keras.layers.Dense(12, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(reg_lamb),
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                                            mean=0.0, stddev=0.05, seed=None)),
    tf.keras.layers.Dense(12, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.l2(reg_lamb),
                          kernel_initializer=tf.keras.initializers.RandomNormal(
                                              mean=0.0, stddev=0.05, seed=None)),
    tf.keras.layers.Dense(1)
])

optim = tf.keras.optimizers.Adam(lr=0.0002)

def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss

model.compile(optimizer = optim,
              loss =get_huber_loss_fn(delta=0.05),
              metrics =['mean_absolute_error'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                    verbose=0, patience=100)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5',
                    monitor='val_loss',
                    mode='min', verbose=0,
                    save_best_only=True)

history = model.fit(x=data["training"]["input"],
        y=data["training"]["target"],
        epochs=5001,
        batch_size=32,
        verbose=1,
        validation_data=(data["validation"]["input"], data["validation"]["target"]),
        callbacks=[es, mc])
saved_model = tf.keras.models.load_model('best_model.h5',custom_objects={'custom_huber_loss': get_huber_loss_fn(delta=0.1)})
if (testing_percent > 0):
    test_loss, test_acc = saved_model.evaluate(x=data["testing"]["input"],
                                        y=data["testing"]["target"],
                                        batch_size=32)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)
    predictions = saved_model.predict(x=data["testing"]["input"]).T *data["norm_target"] + data["mean_target"]
    real_target = data["testing"]["target"]*data["norm_target"] + data["mean_target"]
    print("Target \t  Prediction \t  Difference")
    for i in range(0, len(data["testing"]["target"])):
        print(athletes_PBS.convert_to_string_from_seconds(real_target[i]) + "\t  "
            + athletes_PBS.convert_to_string_from_seconds(predictions[0][i]) + "\t  "
            + str(athletes_PBS.round(predictions[0][i] - real_target[i], 2)))

# Plot training & validation accuracy svalues
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# if (input('Press s to save model or anykey otherwise:') == "s"):
#     print("Saving model")
#     SAVE_MODEL = True
# else:
#     print("Discarding model")
#     SAVE_MODEL = False
# # serialize model to JSON
# if (SAVE_MODEL):
#     model_json = saved_model.to_json()
#     with open(base_folder + model_name + ".json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     saved_model.save_weights(base_folder + model_name + ".h5")
#     print("Saved model to disk")
