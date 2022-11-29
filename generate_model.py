from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
from prepare_data import Data
import numpy as np


class Model:
    def __init__(self):
        print('Reading training files....')
        data_model = Data()
        training_data, classification_labels = data_model.generate_sfts_data(
            for_dataset='TRAIN')

        print('Preparing data....')
        # make all sfts of equal dimension / size
        max_sft_len = len(max(training_data, key=len))
        for i, d in enumerate(training_data):
            training_data[i] = np.pad(d, (0, max_sft_len - len(d)), 'median')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            training_data, classification_labels, test_size=0.20)

    def generate_model(self, clf):
        print('Creating model....')
        clf.fit(self.X_train, self.y_train)

        print('Predicting on test data....')
        y_pred = clf.predict(self.X_test)
        print('Accuracy: ', metrics.accuracy_score(self.y_test, y_pred))

        return clf

    def save_model(self, clf, filename):
        print('Saving the generated model....')
        with open('./' + filename, 'wb') as f:
            pickle.dump(clf, f)


if __name__ == '__main__':
    model = Model()
    clf_svm = svm.SVC(kernel='rbf')
    clf_dt = DecisionTreeClassifier()

    print('Generating model for svm....')
    m_svm = model.generate_model(clf_svm)
    model.save_model(m_svm, 'trained-model-svm.pkl')

    print('Generating model for dt....')
    m_dt = model.generate_model(clf_dt)
    model.save_model(m_dt, 'trained-model-dt.pkl')
