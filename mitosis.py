import os
import csv
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

div = 1
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

def create_model():
    nn = Sequential()
    nn.add(Convolution2D(50, 5, 5, border_mode='valid', input_shape=(99, 99, 3), activation='relu'))
    # 99 - 4 99 -4 7*7*2 = 95, 95, 98
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    # 99 - 6 99 -6 7*7*2 = 47, 47, 98
    nn.add(Convolution2D(50, 1, 1, border_mode='valid', activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    # 46, 46, 49
    nn.add(Dropout(0.2))
    nn.add(Convolution2D(100, 5, 5, border_mode='valid', activation='relu'))
    #  40, 40, 100
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    # 20, 20, 100
    nn.add(Convolution2D(200, 1, 1, border_mode='valid', activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    # 10, 10, 100
    nn.add(Convolution2D(1000, 4, 4, border_mode='valid', activation='relu'))
    nn.add(Convolution2D(2, 1, 1, border_mode='valid', activation='sigmoid'))
    nn.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    return nn

def create_image(file_name, name, root_dir, sz=99):
    with open(file_name + ".csv") as f:
        csv_reader = csv.reader(f)
        image = imread(file_name + ".png")
        rotatable_sz = int(np.sqrt(2) * sz)
        hlf = rotatable_sz // 2
        st = (rotatable_sz - sz) // 2
        image = np.pad(image, [(hlf, hlf), (hlf, hlf), (0, 0)], 'reflect')
        for line in csv_reader:
            y_vals = sorted([int(y) for y in line[1::2]])
            x_vals = sorted([int(x) for x in line[::2]])
            left = y_vals[0] + (y_vals[-1] - y_vals[0]) // 2
            top = x_vals[0] + (x_vals[-1] - x_vals[0]) // 2
            img = image[left:left + rotatable_sz, top:top + rotatable_sz, ::-1]
            center = (img.shape[0] // 2, img.shape[1] // 2)
            for angle in range(0, 360, 10):
                M = cv2.getRotationMatrix2D(center, angle, scale=1)
                rot_i = cv2.warpAffine(img, M, img.shape[:-1])
                cv2.imwrite('{}\\cropped\\{:05d}.png'.format(root_dir, name), rot_i[st:st + sz, st:st + sz, :])
                name += 1
    return name


def file_names(root_dir):
    return ("{0}\\{1}0{3}_v2\\{1}0{3}_0{2}".format(root_dir, let, img, fold)
            for let in ('A', 'H') for fold in range(0, 5) for img in range(0, 10))


def create_new_set(root_dir):
    name = 0
    for file_name in file_names(root_dir):
        try:
            name = create_image(file_name, name, root_dir)
        except Exception as e:
            print(e)
            continue


def create_no_mitosis(file_name, counter, root_dir, model):
    image = imread(file_name + ".png") / div
    image_data = create_final_image(image, img_border)
    mask_shape = image_data.shape[:-1]
    image_data.shape = -1, image_data.shape[-1]
    mask = model.predict_proba(image_data)[:, 1]
    mask.shape = mask_shape
    mask = cv2.GaussianBlur(mask, (5, 5), 1) > .95
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gt_mask = create_mask(mask_shape, file_name)
    mask[gt_mask == 1] = 0
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    xy = output[3][1:, :].astype(int)
    hlf = 99 // 2
    image = np.pad(image, [(hlf, hlf), (hlf, hlf), (0, 0)], 'reflect')
    for corner in xy:
        cv2.imwrite('{}\\no_mitosis\\{:05d}.png'.format(root_dir, counter),
                    image[corner[1]:corner[1] + 99, corner[1]:corner[1] + 99, ::-1])
        counter += 1
    return counter


def create_all_no_mitosis(root_dir):
    counter = 0
    model = joblib.load('model_l1_bal_3.pkl')
    for file_name in file_names(root_dir):
        try:
            counter = create_no_mitosis(file_name, counter, root_dir, model)
        except Exception as e:
            print(e)
            continue


def create_mask(shape, filename):
    """ Loads the csv file, an creates a binary mask of the same size as the
    image, given by shape.

    :param shape: Size of mask
    :param filename: Filename of the csv file to be loaded.
    :return:
    """
    mask = np.zeros(shape).astype('uint8')
    with open(filename + ".csv") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            for i in range(0, len(line), 2):
                mask[int(line[i + 1]), int(line[i])] = 1
    return mask


def shift_overlay(in_image, s_x, s_y):
    """ Creates a version of the given input image, that is shifte by s_x in x direction
    and by s_y in y direction. The region that does not contain any image information is
    filled up with zeros.

    :param in_image: Image (m x n x c) to be shifted
    :param s_x: Shift in x
    :param s_y: Shift in y
    :return: Shifted image.
    """
    out = np.zeros_like(in_image)
    left_x_start, left_x_end = (None, s_x) if s_x < 0 else (s_x, None)
    right_x_start, right_x_end = (-s_x, None) if s_x <= 0 else (None, -s_x)

    left_y_start, left_y_end = (None, s_y) if s_y < 0 else (s_y, None)
    right_y_start, right_y_end = (-s_y, None) if s_y <= 0 else (None, -s_y)

    out[left_y_start:left_y_end, left_x_start:left_x_end, :] = \
        in_image[right_y_start:right_y_end, right_x_start:right_x_end, :]
    return out.astype("float32")


def create_final_image(img, border):
    img = img.astype('float32')  # cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2HSV)
    img -= np.mean(np.mean(img, axis=0), axis=0)
    return np.concatenate([shift_overlay(img, s_x, s_y) for s_x in range(-border, border + 1)
                           for s_y in range(-border, border + 1)], axis=2)


def load_test_and_train(root_dir, border, model=None):
    bad_test, bad_train, good_test, good_train = [None] * 4
    for fold in range(0, 5):
        for img in range(0, 10):
            file_name = "{0}\\{1}0{3}_v2\\{1}0{3}_0{2}".format(root_dir, "A", img, fold)
            print("Reading ", file_name)
            try:
                image = imread(file_name + ".png") / div
            except:
                continue
            shape = image.shape[:-1]

            image = create_final_image(image, border)
            border_mask = np.ones(shape, dtype='uint8')
            border_mask[border:-border, border:-border] = 0
            cancer_mask = create_mask(shape, file_name)
            final_cancer_mask = cancer_mask == 1  # cv2.dilate(cancer_mask, np.ones((5, 5))) == 1
            no_cancer_mask = cv2.dilate(cancer_mask, np.ones((sz + 2, sz + 2))) == 1
            if model:
                old_shape = image.shape
                image.shape = -1, image.shape[-1]
                mask_false_predict = model.predict(image) == 0
                image.shape = old_shape
                mask_false_predict.shape = shape
                border_mask += mask_false_predict

            no_cancer_pixel = image[(no_cancer_mask + border_mask) == 0]
            cancer_pixel = image[final_cancer_mask]

            print(100 * no_cancer_pixel.size / image.size)
            n_samples = np.min([30000, len(no_cancer_pixel)])
            idx = np.random.choice(range(len(no_cancer_pixel)), n_samples, replace=False)
            no_cancer_pixel = no_cancer_pixel[idx, :]

            if img == 0 and fold < 0:
                bad_test = cancer_pixel if bad_test is None else np.concatenate((bad_test, cancer_pixel))
                good_test = no_cancer_pixel if good_test is None else np.concatenate((good_test, no_cancer_pixel))
            else:
                bad_train = cancer_pixel if bad_train is None else np.concatenate((bad_train, cancer_pixel))
                good_train = no_cancer_pixel if good_train is None else np.concatenate((good_train, no_cancer_pixel))
    return bad_test, bad_train, good_test, good_train


def train_the_model(rootdir, border, model=None):
    bad_test, bad_train, good_test, good_train = load_test_and_train(rootdir, border, model)

    def create_sets_and_labels(bad, good):
        out = np.concatenate((bad, good))
        n_bad = len(bad)
        lbls = np.concatenate((np.ones(n_bad), np.zeros(len(good))))
        return out, lbls, n_bad

    training, labels, n_cancer = create_sets_and_labels(bad_train, good_train)
    # test, test_labels, n_c_test = create_sets_and_labels(bad_test, good_test)
    print(training.shape)

    def train(func, name, **kwargs):
        clf = func(**kwargs)
        # scores = cross_val_score(clf, training, labels, cv=5)
        # print(name, np.mean(scores))
        clf.fit(training, labels)
        try:
            y_score = clf.predict_proba(training)

            print("On train cancer ", np.sum((clf.predict(training) == labels)[:n_cancer]) / n_cancer)
            print("On train not cancer ",
                  np.sum((clf.predict(training) == labels)[n_cancer:]) / (len(labels) - n_cancer))
            # print("On test ", np.sum((clf.predict(test) == test_labels)[:n_c_test]) / n_c_test)
            # print("On test not ", np.sum((clf.predict(test) == test_labels)[n_c_test:]) / (len(test_labels) - n_c_test))

            precision, recall, _ = precision_recall_curve(labels, y_score, pos_label=1)
            plt.plot(recall, precision, label=name)
        except:
            pass
        return clf

    # scaler = StandardScaler()
    # scaler = scaler.fit(training)
    # joblib.dump(scaler, 'scaler.pkl')
    # training = scaler.transform(training)
    # model_ = train(linear_model.LogisticRegression, "LOG REG l1 ", penalty='l2', class_weight='balanced')
    # joblib.dump(model_, 'model_l1_bal_{}.pkl'.format(3 if model is None else 4))

    model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(100, 100, 50, 2))
    joblib.dump(model_nn, 'model_nn6.pkl'.format(border))

    # plt.legend(loc="lower left")
    # plt.show()


if __name__ == '__main__':

    # x,y coordinates
    np.random.seed(43)
    sz = 7
    img_border = sz // 2
    image_dir = 'C:\\Users\\Shawe\\Downloads\\Images'
    #create_all_no_mitosis(image_dir)
    train_model = 10
    if train_model == 0:
        model = joblib.load('model_l1_bal_3.pkl')
        train_the_model(image_dir, img_border, model)
        # model = joblib.load('model_l1_bal_3.pkl')
        # train_the_model(image_dir, img_border, model)
    elif train_model == 1:
        model = joblib.load('model_l1_bal_3.pkl')
        file_name = "{0}\\{1}0{3}_v2\\{1}0{3}_0{2}".format(image_dir, "A", 0, 3)
        image_data = imread(file_name + ".png") / div
        image_data = create_final_image(image_data, img_border)
        mask_shape = image_data.shape[:-1]
        image_data.shape = -1, image_data.shape[-1]
        # image_data = pca.transform(image_data)
        mask = model.predict_proba(image_data)[:, 1]
        # mask = model.predict(image_data)
        print(np.sum(mask))
        gt_mask = create_mask(mask_shape, file_name)
        mask.shape = mask_shape
        mask = cv2.GaussianBlur(mask, (5, 5), 1) > .95
        mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Get the results

        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(gt_mask)
        plt.show()
    else:
        print("HELLO")
        # Biggest side == 95
        # Take 99 as side length
        # 98 images
        # 646 mitosis
        #create_new_set(image_dir)
        #data = np.array([np.random.rand(99,99,3) for i in range(10000)],dtype=np.float32)

        #lables = np.random.rand(10000) > .5
        #lables  = np.concatenate((lables,~lables))
        #lables.shape = 10000,1,1,2


        mm = create_model()
        #mm.fit(data, lables, nb_epoch=10, batch_size=200, verbose=2)
        mm.summary()
        # train(linear_model.LogisticRegression, "LOG REG l1 ", penalty='l1', class_weight='balanced')
# train(linear_model.LogisticRegressionCV, "LOG REG l2 CV", penalty='l2', class_weight='balanced')
# train(linear_model.LogisticRegressionCV, "LOG REG l1 CV", penalty='l1', class_weight='balanced', solver='liblinear')
# train(svm.LinearSVC, "LinearSVC l1 ", penalty='l1', loss='squared_hinge', dual=False, class_weight='balanced')
# train(svm.LinearSVC, "LinearSVC l2 ", penalty='l2', dual=False, class_weight='balanced')
# train(linear_model.Perceptron, "Perceptron l2 ", penalty='l2', n_iter=30, class_weight='balanced')
# train(linear_model.Perceptron, "Perceptron l1 ", penalty='l1', n_iter=30, class_weight='balanced')
# train(GradientBoostingClassifier, "GradientBoostingClassifier ", n_estimators=100, random_state=0)
