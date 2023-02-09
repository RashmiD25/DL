
# # Image Segmentation
# 
# Semantic segmentation is the task of placing each pixel into a specific class.  In a sense it's a classification problem where we'll classify on a pixel basis rather than an entire image.  In this assignment the task will be classifying each pixel in a cardiac MRI image based on whether the pixel is a part of the left ventricle (LV) or not.

# ## Input Data Set
# 
# The data set we'll be utilizing is a series of cardiac images (specifically MRI short-axis (SAX) scans) that have been expertly labeled.  
# 

# we obtained guidance and partial code from a prior [Kaggle competition](https://www.kaggle.com/c/second-annual-data-science-bowl/details/deep-learning-tutorial) on how to extract the images properly.  At that point we took the images, converted them to TensorFlow records (TFRecords), and stored them to files.  [TFRecords](https://www.tensorflow.org/programmers_guide/reading_data) are a special file format provided by TensorFlow, which allow you to use built-in TensorFlow functions for data management including multi-threaded data reading and sophisticated pre-processing of the data such as randomizing and even augmenting the training data.
# 
# The images themselves are originally 256 x 256 grayscale [DICOM](https://en.wikipedia.org/wiki/DICOM) format, a common image format in medical imaging.  The label is a tensor of size 256 x 256 x 2.  The reason the last dimension is a 2 is that the pixel is in one of two classes so each pixel label has a vector of size 2 associated with it.  The training set is 234 images and the validation set (data NOT used for training but used to test the accuracy of the model) is 26 images.

# # Deep Learning with TensorFlow

# TensorFlow is an open source software library for machine intelligence.  The computations are expressed as data flow graphs which operate on tensors (hence the name).  If you can express your computation in this manner you can run your algorithm in the TensorFlow framework.
# 
# TensorFlow is portable in the sense that you can run on CPUs and GPUs and utilize workstations, servers, and even deploy models on mobile platforms.  At present TensorFlow offers the options of expressing your computation in either Python or C++, with varying support for other [languages](https://www.tensorflow.org/api_docs/) as well.  A typical usage of TensorFlow would be performing training and testing in Python and once you have finalized your model you might deploy with C++.
# 
# TensorFlow is designed and built for performance on both CPUs and GPUs.  Within a single TensorFlow execution you have lots of flexibility in that you can assign different tasks to CPUs and GPUs explicitly if necessary.  When running on GPUs TensorFlow utilizes a number of GPU libraries including [cuDNN](https://developer.nvidia.com/cudnn) allowing it to extract the most performance possible from the very newest GPUs available.

# Keras provides an easy mechanism to connect layers together.  There is a `Sequential` model which, as the name suggests, allows one to build a neural network from a series of layers, one after the other.  If your neural network structure is more complicated, you can utilize the `Functional` API, which allows more customization due to non-linear topology, shared layers, and layers with multiple inputs.  And you can even extend the `Functional` API to create custom layers of your own.  These different layer types can be mixed and matched to build whatever kind of network architecture you like.  Keras provides a lot of great built-in layers types for typical use cases, and allows the user to build complex layers and models as your needs dictate.  These layers are backed by TensorFlow under-the-covers so the user can concern themself with their model and let TensorFlow worry about performance.
# 

# # Sample Workflow
# 1. Prepare input data--Input data can be Numpy arrays but for very large datasets TensorFlow provides a specialized format called TFRecords.  
# 2. Build the Keras Model--Structure the architecture of your model by defining the neurons, loss function, and learning rate.
# 3. Train the model--Inject input data into the TensorFlow graph by using [model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).  Customize your batch size, number of epochs, learning rate, etc.
# 4. Evaluate the model--run inference (using the same model from training) on previously unseen data and evaluate the accuracy of your model based on a suitable metric.


# # Preparing the Data




import datetime
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Flatten, Dense, Reshape, Conv2D, MaxPool2D, Conv2DTranspose)
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'Greys_r'


# ## Making a TensorFlow Dataset

# loading the training and test sets from TFRecords
raw_training_dataset = tf.data.TFRecordDataset('data/train_images.tfrecords')
raw_val_dataset      = tf.data.TFRecordDataset('data/val_images.tfrecords')


# In order to parse the data, we'll need to provide it's schema. For each feature, we'll define its [class](https://www.tensorflow.org/api_docs/python/tf/io#classes_2) and [data type](https://www.tensorflow.org/api_docs/python/tf/dtypes#other_members). Since our data dimensions are the same for each image, we'll use the [`tf.io.FixedLenFeature`](https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature) class.

# dictionary describing the fields stored in TFRecord, and used to extract the date from the TFRecords
image_feature_description = {
    'height':    tf.io.FixedLenFeature([], tf.int64),
    'width':     tf.io.FixedLenFeature([], tf.int64),
    'depth':     tf.io.FixedLenFeature([], tf.int64),
    'name' :     tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label_raw': tf.io.FixedLenFeature([], tf.string),
}


# The `image_feature_description` can be used to parse each tfrecord with [`tf.io.parse_single_example`](https://www.tensorflow.org/api_docs/python/tf/io/parse_single_example). Since the records are already in a tf.dataset, we can [map](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) the parsing to each record.

# helper function to extract an image from the dictionary
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_training_dataset = raw_training_dataset.map(_parse_image_function)
parsed_val_dataset      = raw_val_dataset.map(_parse_image_function)


# To verify, let's print the number of elements in our parsed dataset. There should be 234 training images and 26 validation images.

print(len(list(parsed_training_dataset)))
print(len(list(parsed_val_dataset)))

# Neural networks work better with floats instead of integers, so we'll scale down the images into a 0 to 1 range. We won't do this with the label image, as we'll use the integers there to represent our prediction classes.

# function to read and decode an example from the parsed dataset
@tf.function
def read_and_decode(example):
    image_raw = tf.io.decode_raw(example['image_raw'], tf.int64)
    image_raw.set_shape([65536])
    image = tf.reshape(image_raw, [256, 256, 1])

    image = tf.cast(image, tf.float32) * (1. / 1024)

    label_raw = tf.io.decode_raw(example['label_raw'], tf.uint8)
    label_raw.set_shape([65536])
    label = tf.reshape(label_raw, [256, 256, 1])

    return image, label


# We can map this decoding function to each image in our dataset like we did with the parsing function before.

# get datasets read and decoded, and into a state usable by TensorFlow
tf_autotune = tf.data.experimental.AUTOTUNE
train = parsed_training_dataset.map(
    read_and_decode, num_parallel_calls=tf_autotune)
val = parsed_val_dataset.map(read_and_decode)
train.element_spec


# setup the buffer size and batch size for data reading and training
BUFFER_SIZE = 10
BATCH_SIZE = 1


# setup the train and test data by shuffling, prefetching, etc
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf_autotune)
test_dataset  = val.batch(BATCH_SIZE)
train_dataset


# ## Visualizing the Dataset

# helper function to display an image, it's label and the prediction
def display(display_list):
    plt.figure(figsize=(10, 10))
    title = ['Input Image', 'Label', 'Predicted Label']

    for i in range(len(display_list)):
        display_resized = tf.reshape(display_list[i], [256, 256])
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_resized)
        plt.axis('off')
    plt.show()


# Bringing it all together, let's take two batches from our dataset and visualize our input images and segmentation labels.

# display an image and label from the training set
for image, label in train.take(2):
    sample_image, sample_label = image, label
    display([sample_image, sample_label])


# If we run the same thing again, we'll get two new sets of images from our dataset.

# display an image and label from the test set
for image, label in val.take(2):
    sample_image, sample_label = image, label
    display([sample_image, sample_label])


# # Task 1 -- One Hidden Layer

# ## Build the Keras Model
# 
# The first task we'll consider will be to create, train and evaluate a fully-connected neural network with one hidden layer:
# 
# * The input to the neural network will be the value of each pixel, i.e., a size 256 x 256 x 1 (or 65,536) array. The `X 1` is the number of color channels. Since the images are black and white, we'll only use 1, but if it was color, we might use `X 3` for the red, green, and blue in [RGB](https://www.w3schools.com/colors/colors_rgb.asp). The following Dense layer expects a vector, not a matrix, so we'll [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the incoming image.
# * The hidden [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layer will have a size that you can adjust to any positive integer.
# * the output will have 256 x 256 x 2 values, i.e., each input pixel can be in either one of two classes so the output value associated with each pixel will be the probability that the pixel is in that particular class.  In our case the two classes are LV or not. Then, we'll [Reshape](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape) it into a 256 x 256 x 2 matrix so we can see the result as an image.
# 
# We'll compute the loss via a TensorFlow function called [`sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits) which simply combines the softmax with the cross entropy calculation into one function call.

tf.keras.backend.clear_session()

# set up the model architecture
model = tf.keras.models.Sequential([
    Flatten(input_shape=[256, 256, 1]),
    Dense(64, activation='relu'),
    Dense(256*256*2, activation='softmax'),
    Reshape((256, 256, 2))
])

# specify how to train the model with algorithm, the loss function and metrics
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


# print out the summary of the model
model.summary()

# plot the model including the sizes of the model
tf.keras.utils.plot_model(model, show_shapes=True)


# function to take a prediction from the model and output an image for display
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# helper function to show the image, the label and the prediction
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, label in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], label[0], create_mask(pred_mask)])
    else:
        prediction = create_mask(model.predict(sample_image[tf.newaxis, ...]))
        display([sample_image, sample_label, prediction])


# show a predection, as an example
show_predictions(test_dataset)



# define a callback that shows image predictions on the test set
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# setup a tensorboard callback
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# ## Train the Model

# setup and run the model
EPOCHS = 20
STEPS_PER_EPOCH = len(list(parsed_training_dataset))
VALIDATION_STEPS = 26

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback])


# output model statistics
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


# ## Evaluate the Model

model.evaluate(test_dataset)


# ### TensorBoard

get_ipython().run_line_magic('load_ext', 'tensorboard')

get_ipython().run_line_magic('tensorboard', '--logdir logs')


show_predictions(test_dataset, 5)


# # Task 2 -- Convolutional Neural Network (CNN)


tf.keras.backend.clear_session()

layers = [
    Conv2D(input_shape=[256, 256, 1],
           filters=100,
           kernel_size=5,
           strides=2,
           padding="same",
           activation=tf.nn.relu,
           name="Conv1"),
    MaxPool2D(pool_size=2, strides=2, padding="same"),
    Conv2D(filters=200,
           kernel_size=5,
           strides=2,
           padding="same",
           activation=tf.nn.relu),
    MaxPool2D(pool_size=2, strides=2, padding="same"),
    Conv2D(filters=300,
           kernel_size=3,
           strides=1,
           padding="same",
           activation=tf.nn.relu),
    Conv2D(filters=300,
           kernel_size=3,
           strides=1,
           padding="same",
           activation=tf.nn.relu),
    Conv2D(filters=2,
           kernel_size=1,
           strides=1,
           padding="same",
           activation=tf.nn.relu),
    Conv2DTranspose(filters=2, kernel_size=31, strides=16, padding="same")
]

model = tf.keras.models.Sequential(layers)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])



# print out the summary of the model
model.summary()

# plot the model including the sizes of the model
tf.keras.utils.plot_model(model, show_shapes=True)

# show a predection, as an example
show_predictions(test_dataset)


# Initialize new directories for new task
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# setup and run the model
EPOCHS = 20
STEPS_PER_EPOCH = len(list(parsed_training_dataset))
VALIDATION_STEPS = 26

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback(), tensorboard_callback])


#with 1 epoch of training we achieved an accuracy of 56.7%:
# 
# ```
# OUTPUT: 2017-01-27 17:41:52.015709: precision = 0.567
# ```
# 
# When increasing the number of epochs to 10, we obtained a much higher accuracy of this:
# 
# ```
# OUTPUT: 2017-01-27 17:47:59.604529: precision = 0.983
# ```
# 
# As you can see when we increase the training epochs we see a significant increase in accuracy. 


# output model statistics
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


model.evaluate(test_dataset)


# show predictions from the test data set that has NOT been used for training.
show_predictions(test_dataset, 5)


# # Accuracy

# # Task 3 -- CNN with Dice Metric Loss
# 
# One metric we can use to more accurately determine how well our network is segmenting LV is called the Dice metric or Sorensen-Dice coefficient, among other names.  This is a metric to compare the similarity of two samples.  In our case we'll use it to compare the two areas of interest, i.e., the area of the expertly-labelled contour and the area of our predicted contour.  The formula for computing the Dice metric is:
# 
# $$ \frac{2A_{nl}}{A_{n} + A_{l}} $$
# 
# where $A_n$ is the area of the contour predicted by our neural network, $A_l$ is the area of the contour from the expertly-segmented label and $A_{nl}$ is the intersection of the two, i.e., the area of the contour that is predicted correctly by the network.  1.0 means perfect score.
# 
# This metric will more accurately inform us of how well our network is segmenting the LV because the class imbalance problem is negated.  Since we're trying to determine how much area is contained in a particular contour, we can simply count the pixels to give us the area.


def dice_coef(y_true, y_pred, smooth=1):
    indices = K.argmax(y_pred, 3)
    indices = K.reshape(indices, [-1, 256, 256, 1])

    true_cast = y_true
    indices_cast = K.cast(indices, dtype='float32')

    axis = [1, 2, 3]
    intersection = K.sum(true_cast * indices_cast, axis=axis)
    union = K.sum(true_cast, axis=axis) + K.sum(indices_cast, axis=axis)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

    return dice

# Initialize new directories for new task
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


tf.keras.backend.clear_session() 
model = tf.keras.models.Sequential(layers)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[dice_coef,'accuracy'])


# setup and run the model
EPOCHS = 30
STEPS_PER_EPOCH = len(list(parsed_training_dataset))

model_history = model.fit(train_dataset, epochs=EPOCHS,
                         steps_per_epoch=STEPS_PER_EPOCH,
                         validation_data=test_dataset,
                         callbacks=[DisplayCallback()])


# output model statistics
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']
dice = model_history.history['dice_coef']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.plot(epochs, dice, 'go', label='Dice Coefficient')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


# In a prior run we obtained 
# 
# ```
# OUTPUT: 2017-01-27 18:44:04.103153: Dice metric = 0.9176
# ```
# 
# for 1 epoch.  If you try with 30 epochs you might get around 99.9% accuracy.
# 
# ```
# OUTPUT: 2017-01-27 18:56:45.501209: Dice metric = 0.9583
# ```
# 
# With a more realistic accuracy metric, you can see that there is some room for improvement in the neural network.

# # Parameter Search
# 
# At this point we've created a neural network that we think has the right structure to do a reasonably good job and we've used an accuracy metric that correctly tells us how well our network is learning the segmentation task.  But up to this point our evaluation accuracy hasn't been as high as we'd like.  The next thing to consider is that we should try to search the parameter space a bit more.  Up to now we've changed the number of epochs but that's all we've adjusted.  There are a few more parameters we can test that could push our accuracy score higher.
# 
# When it comes to image segmentation, there are many types of [metrics](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) and [loss functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses). For this demo, [Categorical Crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) was effective as it can apply near infinite loss for innacurate predictions from a model, but it's worth exploring when TensorFlow has to offer.
# 
# We could also alter the structure of our model such as:
#  - changing the number of layers
#    - changing the proportion of Conv2D, MaxPool2D and Conv2DTranspose layers
#  - changing the number and size of the filters
#  - chanfing the stride of the filters


layers = [
    Conv2D(input_shape=[256, 256, 1],
           filters=300,
           kernel_size=5,
           strides=2,
           padding="same",
           activation=tf.nn.relu,
           name="Conv1"),
    MaxPool2D(pool_size=2, strides=1, padding="same"),
    Conv2DTranspose(filters=200, kernel_size=5, strides=2, padding="same")
]

# Initialize new directories for new task
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential(layers)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[dice_coef,'accuracy'])




EPOCHS = 30
STEPS_PER_EPOCH = len(list(parsed_training_dataset))

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])



