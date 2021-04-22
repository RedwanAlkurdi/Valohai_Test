import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
import shutil


INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
TRAIN_IMAGES_DIR = os.path.join(INPUTS_DIR, "training-set-images")
TEST_IMAGES_DIR = os.path.join(INPUTS_DIR, "test-set-images")



# Get the Horse or Human dataset
path_horse_or_human = TRAIN_IMAGES_DIR + "/train.zip"
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = TEST_IMAGES_DIR + "/validation.zip"



OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
#shutil.rmtree('/tmp')
local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(os.path.join(getcwd(), "training"))
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(os.path.join(getcwd(), "validation"))
zip_ref.close()


print("Done")


# our example directories and files
train_dir = os.path.join(getcwd(), "training") + '/training/train'
validation_dir = os.path.join(getcwd(), "validation") + "/validation/validation/"

train_horses_dir = os.path.join(train_dir, "horses")
train_humans_dir = os.path.join(train_dir, "humans")
validation_horses_dir = os.path.join(validation_dir, "horses")
validation_humans_dir = os.path.join(validation_dir, "humans")

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)



INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
WEIGHTS = os.path.join(INPUTS_DIR, "WEIGHTS")

path_inception = WEIGHTS + "/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

#  the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

#  an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception

pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                               include_top = False,
                               weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable=False


# Print the model summary
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True




# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
#  a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
#  a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
#  a final sigmoid layer for classification
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = Adam(learning_rate=3e-4),
              loss = 'binary_crossentropy',
              metrics = ['acc',"binary_accuracy"])

model.summary()


#  data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True
)

# the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)



# Flow training images in batches of X using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=60,
    class_mode='binary'
)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=60,
    class_mode='binary')


callbacks = myCallback()
history = model.fit(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=[callbacks])



# Save model and weights.
outputs_dir = os.getenv('VH_OUTPUTS_DIR', './')
output_file = os.path.realpath(os.path.join(outputs_dir, 'my_model.h5'))
if not os.path.isdir(outputs_dir):
    os.makedirs(outputs_dir)

print('Saving trained model to %s' % output_file)
model.save(output_file)

"""acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()"""
