from __future__ import print_function, division
from theano.sandbox import cuda
cuda.use('gpu0')
path = "data/sign_mix_expedition3/"

import utils; reload(utils)
from utils import *

import cv2
import os
import time
from skimage.transform import resize

batch_size=32
vgg = Vgg16()
model=vgg.model
last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
conv_layers = model.layers[:last_conv_idx+1]

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
hand_sign_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
batches = get_batches(path+'train', batch_size=batch_size)
hand_sign_model.load_weights(path+'models/da_conv8_1_at80.h5')

final_labels = {}
for x,y in batches.class_indices.items():
    final_labels[y]=x

video_capture = cv2.VideoCapture(0)
fps = 0
start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fps += 1

    # Draw rectangle around face
    x = 150
    y = 50
    w = 300
    h = 350
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

    # Crop + process captured frame
    hand_ini = frame[60:390, 160:440]
    cv2.imwrite(path+'imgs/imgs/img-'+str(fps)+'.jpg', hand_ini)
    #hand = np.expand_dims(resize(hand_ini, (224, 224), mode='reflect'), axis=0).transpose(0, 3, 1, 2)
    # hand = resize(hand, (224, 224), mode='reflect').transpose(2,0,1)
    # Make prediction
    #my_predict = hand_sign_model.predict(hand,
    #                              batch_size=1,
    #                              verbose=0)
    test_batches = get_batches(path+'imgs', batch_size=1, shuffle=False)
    my_predict = hand_sign_model.predict_generator(test_batches, test_batches.nb_sample)
    # Predict letter
    top_prd = np.argmax(my_predict)
    for i, x in enumerate(my_predict[0]):
        print(final_labels[i], x)
    # Only display predictions with probabilities greater than 0.5
    if np.max(my_predict) >= 0.10:

        prediction_result = final_labels[top_prd]
        preds_list = np.argsort(my_predict)[0]
        pred_2 = final_labels[preds_list[-2]]
        pred_3 = final_labels[preds_list[-3]]

        width = int(video_capture.get(3) + 0.5)
        height = int(video_capture.get(4) + 0.5)

        # Annotate image with most probable prediction
        cv2.putText(frame, text=prediction_result,org=(200,450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(0, 0, 0),
                    thickness=6, lineType=cv2.CV_AA)
        # Annotate image with second probable prediction (displayed on bottom right)
        cv2.putText(frame, text=pred_2,org=(275,450),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=4, color=(0, 0, 0),
                    thickness=6, lineType=cv2.CV_AA)
        # Annotate image with third probable prediction (displayed on bottom right)
        cv2.putText(frame, text=pred_3,org=(350,450),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=4, color=(0, 0, 0),
                    thickness=6, lineType=cv2.CV_AA)

    # Display the resulting frame
    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video',cv2.WND_PROP_FULLSCREEN,cv2.cv.CV_WINDOW_FULLSCREEN)
    cv2.imshow('Video', frame)
    os.remove(path+'imgs/imgs/img-'+str(fps)+'.jpg')

    # Press 'q' to exit live loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate frames per second
end = time.time()
FPS = fps/(end-start)
print("[INFO] approx. FPS: {:.2f}".format(FPS))

# Release the capture
video_capture.release()
cv2.destroyAllWindows()