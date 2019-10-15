from __future__ import print_function
import itertools
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import sys

class99 = True

kaggle_name = 'Major Tom'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()

    cm_diagonal = np.zeros(np.shape(cm))
    for i in range(np.shape(cm)[0]):
        cm_diagonal[i, i] = cm[i, i]

    cm_off_diagonal = np.zeros(np.shape(cm))
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if i != j:
                cm_off_diagonal[i, j] = cm[i, j]

    ma_cm_diagonal = ma.masked_array(cm_diagonal, cm_diagonal<0.03)
    ma_cm_off_diagonal = ma.masked_array(cm_off_diagonal, cm_off_diagonal<0.005)

    im = ax.imshow(ma_cm_off_diagonal, interpolation='nearest', cmap=plt.cm.Reds,
                   vmin=0., vmax=np.amax(ma_cm_off_diagonal))
    im1 = ax.imshow(ma_cm_diagonal, interpolation='nearest', cmap=plt.cm.Blues,
                    vmin=0., vmax=1.)

    if class99:
        ax.set_title(kaggle_name)
    else:
        ax.set_title('{} - No Class 99'.format(kaggle_name))
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, size=12)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, size=12)

    fmt = '.2f' if normalize else 'd'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            thresh = cm.max() / 2.
        else:
            thresh = np.amax(ma_cm_off_diagonal) / 2.
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', size=15)
    ax.set_xlabel('Predicted label', size=15)
    plt.tight_layout()
    if class99:
        fig.savefig("{}_20191015.pdf".format(title))
        fig.savefig("{}_20191015.png".format(title))
    else:
        fig.savefig("{}_noclass99_20191015.pdf".format(title))
        fig.savefig("{}_noclass99_20191015.png".format(title))


def main():

    # read in data
    submission_file = sys.argv[1]
    filenum = submission_file.split('.csv')[0]
    truth_file = "plasticc_test_truthtable.csv"
    classname_file = "classnames.txt"
    z_file = "redshifts.tsv"
    classlist = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
    #classlist2 = [90, 67, 52, 42, 62, 15, 95, 64, 88, 92, 65, 16, 53, 6, 99]
    classdict = {'90': 100, '67': 101, '52': 102, '42': 103, '62': 104, '95': 105, '15': 106,
                  '64': 107, '88': 108, '92': 109, '65': 110 , '16': 111, '53': 112,
                  '6': 113, '99': 114}

    classlist4 = [0, 3, 4, 2, 1, 6, 8, 5, 9, 10, 11, 12, 13, 14, 7]
    if len(sys.argv)<4:
        zrange = [-np.inf,np.inf]
    else:
        zrange = [np.float(sys.argv[2]),np.float(sys.argv[3])]

    classnames = pd.read_table(classname_file,sep='\s+',header=0)

    #classnames = classnames.sort_values(by='TARGET')
    cl = classnames.reindex(classlist4, axis='index')
    #print(classnames)
    classnames = cl['MODEL_NAME']

    ztable = pd.read_csv(z_file,sep='\t',header=0)
    ztable.rename(columns={'SNID':'object_id'},inplace=True)
    ztable = ztable[(ztable['SIM_REDSHIFT_HOST']>zrange[0]) & (ztable['SIM_REDSHIFT_HOST']<=zrange[1])]
    print("zrange = {}, {}".format(np.min(ztable['SIM_REDSHIFT_HOST']),np.max(ztable['SIM_REDSHIFT_HOST'])))

    print("Reading in submission")
    probs = pd.read_csv(submission_file, sep=',',header=0)
    probs.rename(columns={x: x.split('_')[1] for x in list(probs) if x != 'object_id'},inplace=True)
    print("Done reading in submission:",submission_file)
    truth = pd.read_csv(truth_file,sep=',',header=0)

    truth = truth.join(ztable.set_index('object_id'),on='object_id',how='inner')
    probs = probs.join(ztable.set_index('object_id'),on='object_id',how='inner')
    probs.sort_values(by='object_id',inplace=True)

    if not class99:
        probs = probs.drop(columns=['object_id', 'SIM_REDSHIFT_HOST', 'HOSTGAL_SPECZ',
                                    'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR'])
        probs['99'] = 0
        norm = np.sum(probs, axis=1)
        probs = probs.div(norm, axis=0)

    truth.sort_values(by='object_id',inplace=True)

    classids = [x for x in list(probs) if x[0].isdigit()]
    y_pred = probs[classids].idxmax(axis=1).values.astype(int) ## classids
    y_true = truth['target'].values

    for key, value in classdict.items():
        wh_y_true, = np.where(y_true == int(key))
        y_true[wh_y_true] = value
        #print(np.shape(wh_y_true))

        wh_y_pred, = np.where(y_pred == int(key))
        y_pred[wh_y_pred] = value
        #print(np.shape(wh_y_pred))

    #print(y_true[0:10], y_pred[0:10])

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred) #, labels=np.arange(100, 115, 1))
    
    np.set_printoptions(precision=2)

    #classidx = np.searchsorted(classlist,np.unique([y_pred,y_true]))
    #classidx = np.sort(classidx)


    #classes = classnames.reset_index(drop=True)[classlist4]#classidx]

    plot_confusion_matrix(cnf_matrix, classes=classnames, normalize=True,
                          title='Normalized_confusion_matrix-z{}_{}-{}'.format(zrange[0],zrange[1],filenum))


if __name__ == "__main__":
    main()
