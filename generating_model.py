import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import format_data_for_ML

model_name = "initial"
data_folder = ""
data_name = "combined_results_2019"

training_percent = 0.8
testing_percent = 0.1
# validation_percent = 1 - training_percent - testing_percent

data = format_data_for_ML.get_data(data_folder = data_folder, data_name = data_name,
            training_percent = training_percent, testing_percent = testing_percent)


reg_lamb = 0.01

model = tf.keras.Sequential([
  tf.keras.layers.Dense(5, activation="relu", input_shape=(6,),
                        kernel_regularizer=tf.keras.regularizers.l2(reg_lamb)),
  tf.keras.layers.Dense(3, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(reg_lamb)),
  tf.keras.layers.Dense(1)
])

sgd = tf.keras.optimizers.RMSprop(lr=0.005)

model.compile(optimizer = sgd,
              loss ='mse',
              metrics =['mean_squared_error'])

# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
#                     verbose=1, patience=100)
# mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5',
#                     monitor='val_acc',
#                     mode='max', verbose=0,
#                     save_best_only=True)
history = model.fit(data["training"]["input"], data["training"]["target"],
        epochs=1001,
        batch_size=32,
        verbose=1,
        validation_data=(data["validation"]["input"], data["validation"]["target"]))


# Plot training & validation accuracy svalues
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
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

if (testing_percent > 0):
    test_loss, test_acc = saved_model.evaluate(data["testing"]["input"], data["testing"]["target"])
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


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
