from __future__ import print_function
import pandas as pd
import numpy as np
import math
import os
import sys
from sklearn.metrics import confusion_matrix
import progressbar

widgets = [' (', progressbar.Percentage(), ') ',' [', progressbar.Timer(), '] ', progressbar.Bar(),' [', progressbar.ETA(), '] ',]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class NaiveBayesModel:
    def __init__(self, train_data, train_label, n_vocabs):
        self.prob_class = []
        self.n_k = {}
        self.n = {}
        self.train_data = train_data
        self.train_label = train_label
        self.all_classes = np.sort(np.unique(self.train_label['label']))
        self.n_examples = len(np.unique(self.train_data['docIdx']))
        self.n_vocabs = n_vocabs

    def docs_in_class(self, class_num):
        n_docs_in_class = self.train_label['label'] == class_num
        return np.sum(n_docs_in_class)

    def prob_BE(self, class_num, word):
        word_nk = 0 if word not in self.n_k[class_num] else self.n_k[class_num][word]
        return (word_nk + 1) / (self.n[class_num] + self.n_vocabs)

    def prob_MLE(self, class_num, word):
        word_nk = 0 if word not in self.n_k[class_num] else self.n_k[class_num][word]
        return word_nk / self.n[class_num]

    def compute_nk(self, class_num):
        docs_in_class = self.train_label['label'][self.train_label['label'] == class_num].index + 1
        data_in_class = self.train_data[self.train_data['docIdx'].isin(docs_in_class)]
        self.n[class_num] = np.sum(data_in_class['count'])
        self.n_k[class_num] = {}
        unique_words_in_class = np.unique(data_in_class['wordIdx'])

        for word in unique_words_in_class:
            self.n_k[class_num][word] = np.sum(data_in_class[data_in_class['wordIdx'] == word]['count'])

    def fit(self):
        self.prob_class = [self.docs_in_class(cls) / self.n_examples for cls in self.all_classes]
        with progressbar.ProgressBar(max_value=len(self.all_classes), widgets=widgets) as bar:
            for cls in self.all_classes:
                self.compute_nk(cls)
                bar.update(cls - 1)

    def print_prob_class(self):
        for idx, prob in enumerate(self.prob_class):
            print(r'Pr(Omega = {:2d}) = {:.4f}'.format(idx + 1, prob))

    def predict_doc(self, doc, mode = 'BE'):
        decision_cls = []
        for cls in self.all_classes:
            prob = math.log(self.prob_class[cls - 1])
            for word, count in zip(doc['wordIdx'], doc['count']):
                prob_word = self.prob_BE(class_num=cls, word=word) if mode == 'BE' else self.prob_MLE(class_num=cls, word=word)
                if prob_word == 0:
                    prob = -math.inf
                    break
                prob = prob + count * math.log(prob_word)
            decision_cls.append(prob)
        winner_cls = np.argmax(decision_cls) + 1
        return winner_cls

    def evaluate(self, data, labels, mode='BE'):
        docs = np.unique(data['docIdx'])
        self.y_true = np.array(labels['label'])
        self.y_pred = np.array([])
        with progressbar.ProgressBar(max_value=len(docs), widgets=widgets) as bar:
            for idx, doc in enumerate(docs):
                pred_label = self.predict_doc(data[data['docIdx'] == doc], mode)
                self.y_pred = np.append(self.y_pred, pred_label)
                bar.update(idx)

        correct = np.sum(self.y_true == self.y_pred)
        return correct / len(docs)

    def report_evaluation(self, data, label, mode='BE', file_name='out'):
        print('*' * 100)
        print('*' * 35 , 'Evaluation for {}'.format(file_name), '*' * 40)
        print('*' * 100)

        overall_acc = self.evaluate(data, label, mode)
        print('Overall Accuracy {:.4f}'.format(overall_acc))

        self.cm = confusion_matrix(self.y_true, self.y_pred)
        acc_per_cls = self.cm.diagonal() / self.cm.sum(axis=1)

        print('Class Accuracy:')
        for cls in self.all_classes:
            print('Group {:2d} : {:.4f}'.format(cls, acc_per_cls[cls - 1]))

        np.set_printoptions(linewidth=np.inf)
        print('Confusion Matrix:')
        print(self.cm)

def main(args):
    root_address = '.'
    data_cols = ['docIdx', 'wordIdx', 'count']
    label_cols = ['label']
    train_data = pd.read_csv(os.path.join(root_address, args[0]), header=None)
    train_label = pd.read_csv(os.path.join(root_address, args [1]), header=None)
    test_data = pd.read_csv(os.path.join(root_address, args [2]), header=None)
    test_label = pd.read_csv(os.path.join(root_address, args [3]), header=None)
    vocabs = pd.read_csv(os.path.join(root_address, args [4]), header=None)

    n_vocabs = len(vocabs)

    train_data.columns = data_cols
    train_label.columns = label_cols
    test_data.columns = data_cols
    test_label.columns = label_cols

    print('*' * 100)
    print('*' * 41, 'Training process', '*' * 41)
    print('*' * 100)

    NBM = NaiveBayesModel(train_data, train_label, n_vocabs)
    NBM.fit()
    NBM.print_prob_class()

    NBM.report_evaluation(train_data, train_label, mode='BE', file_name='BE_train')
    NBM.report_evaluation(train_data, train_label, mode='MLE', file_name='MLE_train')
    NBM.report_evaluation(test_data, test_label, mode='BE', file_name='BE_test')
    NBM.report_evaluation(test_data, test_label, mode='MLE', file_name='MLE_test')

if __name__ == '__main__':
    if len(sys.argv) < 6:
        eprint ('Command line: lab1.py train_data.csv train_label.csv test_data.csv test_label.csv vocabulary.txt')
        exit(0)
    main(sys.argv[1:])