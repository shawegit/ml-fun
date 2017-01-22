import os
import csv
from collections import defaultdict
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import linear_model, datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

div = 1


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


def load_test_and_train(root_dir, border):
    bad_test, bad_train, good_test, good_train = [None] * 4
    for fold in range(1, 5):
        for img in range(0, 10):
            file_name = "{0}\\{1}0{3}_v2\\{1}0{3}_0{2}".format(root_dir, "A", img, fold)
            print("Reading ", file_name)
            image_data = imread(file_name + ".png") / div
            shape = image_data.shape[:-1]

            image_data = create_final_image(image_data, border)
            o_mask = np.ones(shape).astype('uint8')
            o_mask[sz:-sz, sz:-sz] = 0
            mask = create_mask(shape, file_name)
            new_mask = cv2.erode(mask, np.ones((3, 3))) == 1
            new_mask2 = cv2.dilate(mask, np.ones((3, 3))) == 1
            bad_pix = image_data[new_mask]
            good_ones = image_data[(new_mask2 + o_mask) == 0]
            idx = np.random.choice(range(len(good_ones)), np.sum(new_mask) * 15, replace=False)
            good_ones = good_ones[idx, :]

            if img == 0 and fold < 0:
                bad_test = bad_pix if bad_test is None else np.concatenate((bad_test, bad_pix))
                good_test = good_ones if good_test is None else np.concatenate((good_test, good_ones))
            else:
                bad_train = bad_pix if bad_train is None else np.concatenate((bad_train, bad_pix))
                good_train = good_ones if good_train is None else np.concatenate((good_train, good_ones))
    return bad_test, bad_train, good_test, good_train


def train_the_model(rootdir, border):
    bad_test, bad_train, good_test, good_train = load_test_and_train(rootdir, border)

    # for i in range(27, 28):
    # my_pca = PCA(n_components=bad_test.shape[-1])
    # my_pca.fit(np.concatenate((good_train, bad_train)))
    # joblib.dump(my_pca, 'pca.pkl')

    def create_sets_and_labels(bad, good):
        # out = my_pca.transform(np.concatenate((bad, good)))
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

    szsz = training.shape[-1]
    model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(2*szsz, szsz, szsz // 2, 30))
    joblib.dump(model_nn, 'model_nn7.pkl'.format(border))

    model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(2*szsz, szsz, szsz // 2, 20))
    joblib.dump(model_nn, 'model_nn7.pkl'.format(border))

    model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(2*szsz, szsz, szsz // 2, 40))
    joblib.dump(model_nn, 'model_nn7.pkl'.format(border))

    model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(szsz // 2, szsz // 4, 30, 30))
    joblib.dump(model_nn, 'model_nn7.pkl'.format(border))



    # model_nn = train(MLPClassifier, "Neural Net ", hidden_layer_sizes=(50, 60, 50, 30))
    # joblib.dump(model_nn, 'model_nn6.pkl'.format(border))
    # model = train(linear_model.LogisticRegression, "LOG REG l1 ", penalty='l1', class_weight='balanced')
    # joblib.dump(model, 'model_l1_{}.pkl'.format(border))

    # plt.legend(loc="lower left")
    # plt.show()


if __name__ == '__main__':

    # x,y coordinates
    np.random.seed(43)
    sz = 7
    img_border = sz // 2
    image_dir = 'C:\\Users\\Obivion\\Downloads\\Images'
    train_model = 1
    if train_model:
        train_the_model(image_dir, img_border)
    else:
        pca = joblib.load('pca.pkl')
        model = joblib.load('model_l1_bal.pkl')
        file_name = "{0}\\{1}0{3}_v2\\{1}0{3}_0{2}".format(image_dir, "A", 0, 3)
        image_data = imread(file_name + ".png") / div
        image_data = create_final_image(image_data, img_border)
        mask_shape = image_data.shape[:-1]
        image_data.shape = -1, image_data.shape[-1]
        # image_data = pca.transform(image_data)
        hh = model.predict_proba(image_data)
        mask = (hh > .95)[:, 1]  # model.predict(image_data)
        #mask = model.predict(image_data)
        mask.shape = mask_shape
        plt.subplot(121)
        plt.imshow(mask)
        gt_mask = create_mask(mask_shape, file_name)
        plt.subplot(122)
        plt.imshow(gt_mask)
        plt.show()
        # train(linear_model.LogisticRegression, "LOG REG l1 ", penalty='l1', class_weight='balanced')
# train(linear_model.LogisticRegressionCV, "LOG REG l2 CV", penalty='l2', class_weight='balanced')
# train(linear_model.LogisticRegressionCV, "LOG REG l1 CV", penalty='l1', class_weight='balanced', solver='liblinear')
# train(svm.LinearSVC, "LinearSVC l1 ", penalty='l1', loss='squared_hinge', dual=False, class_weight='balanced')
# train(svm.LinearSVC, "LinearSVC l2 ", penalty='l2', dual=False, class_weight='balanced')
# train(linear_model.Perceptron, "Perceptron l2 ", penalty='l2', n_iter=30, class_weight='balanced')
# train(linear_model.Perceptron, "Perceptron l1 ", penalty='l1', n_iter=30, class_weight='balanced')
# train(GradientBoostingClassifier, "GradientBoostingClassifier ", n_estimators=100, random_state=0)
