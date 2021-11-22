import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# Dogs and Cats images should be placed to "pets/dogs" and "pets/cats" respectively
CATEGORIES = ['dogs', 'cats']
IMAGE_SIZE = 150


def run(data_dir_path):
    x_path = data_dir_path + os.path.sep + 'X.pickle'
    y_path = data_dir_path + os.path.sep + 'Y.pickle'
    x = []
    y = []

    try:
        pickle_in = open(x_path, 'rb')
        x = pickle.load(pickle_in)
        pickle_in.close()

        pickle_in = open(y_path, 'rb')
        y = pickle.load(pickle_in)
        pickle_in.close()
    except Exception as e:
        print(e)
        training_data = _create_training_data(data_dir_path)
        print(len(training_data))

        for features, label in training_data:
            x.append(features)
            y.append(label)

        x = np.array(x).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        pickle_out = open(x_path, 'wb')
        pickle.dump(x, pickle_out)
        pickle_out.close()

        y = np.array(y)
        pickle_out = open(y_path, 'wb')
        pickle.dump(y, pickle_out)
        pickle_out.close()


def _create_training_data(data_dir_path):
    training_data = []
    for categoty in CATEGORIES:
        categoty_path = data_dir_path + os.path.sep + categoty
        class_num = CATEGORIES.index(categoty)
        for img in os.listdir(categoty_path):
            try:
                path_to_image = categoty_path + os.path.sep + img
                img_array = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
                resized_images = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([resized_images, class_num])
            except Exception as e:
                # print("Error -> " + str(e))
                pass
    random.shuffle(training_data)
    return training_data
