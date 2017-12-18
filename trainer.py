from __future__ import print_function, division
from theano.sandbox import cuda

cuda.use('gpu0')
path = "data/sign_mix_expedition3/"

import utils; reload(utils)
from utils import *

batch_size=32

# This is for extracting out the convolutional part of vgg16
vgg = Vgg16()
model=vgg.model
last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
conv_layers = model.layers[:last_conv_idx+1]

for layer in conv_layers:
    layer.trainable = False

def get_bn_da_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(24, activation='softmax')
        ]
p=0.8
bn_fc_layers = get_bn_da_layers(p)

hand_sign_model = Sequential(conv_layers + bn_fc_layers)
hand_sign_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Generate batches for all our data
batches = get_batches(path+'train', batch_size=batch_size)
val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
test_batches = get_batches(path+'test', batch_size=batch_size, shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)

gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.15, 
                                 channel_shift_range=20, width_shift_range=0.15)
da_batches = get_batches(path+'train', gen_t, batch_size=batch_size, shuffle=True)

da_trn_labels = np.concatenate([trn_labels]*4)

da_batches.nb_sample

hand_sign_model.fit_generator(da_batches, samples_per_epoch=da_batches.nb_sample*4, nb_epoch=10,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

hand_sign_model.save_weights(path+'models/da_conv8_1_at80.h5')

# Do only when weights have been saved already
# hand_sign_model.load_weights(path+'models/da_conv8_1_at71.h5')

###TEST TEST TEST###

final_labels = {}
for x,y in batches.class_indices.items():
    final_labels[y]=x

test_batches = get_batches(path+'test', batch_size=batch_size, shuffle=False)

preds = hand_sign_model.predict_generator(test_batches, test_batches.nb_sample)

for i, x in enumerate(preds[0]):
    print(final_labels[i], x)

hand_sign_model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=1,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)
hand_sign_model.save_weights(path+'models/da_conv8_1.h5')

final_labels = {}
for x,y in batches.class_indices.items():
    final_labels[y]=x
final_labels

