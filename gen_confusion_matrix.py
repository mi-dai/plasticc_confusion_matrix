from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import sys

def plot_confusion_matrix(cm, classes, newinds=[],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if newinds==[]:
        newinds= np.arange(len(classes))

    print(newinds, 'newinds', len(newinds), 'length', len(classes), 'class length')
    newcm = [[0] * len(classes) for i in range(len(classes))]

    for ci,i in enumerate(newinds):
        for cj,j in enumerate(newinds):
            newcm[ci][cj]=cm[i][j]

    newcm = np.array(newcm)
#    print(newcm[0][0], cm[newinds[0]][newinds[0]])
    print(np.shape(newcm), 'newcm shape')

    print(np.shape(cm), 'cm shape')

    cm= newcm


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)





    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(classes,'original')
    print(classes[newinds],'newinds')
    #plt.xticks(tick_marks, classes[newinds], rotation=45)
    #plt.yticks(tick_marks, classes[newinds])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main():

    # read in data
    submission_file = sys.argv[1]
    filenum = submission_file.split('.csv')[0]
    truth_file = "truth/plasticc_test_truthtable.csv"
    classname_file = "truth/classnames.txt"
    z_file = "truth/redshifts.tsv"
    classlist = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
    
    newclasslist = [15,16,53,92,65,42,62,6,90,88,64,52,67,95,99]

    classlist = newclasslist
    newinds = [1,2,5,12,8,3,6,0,11,10,7,4,9,13,14]

    if len(sys.argv)<4:
        zrange = [-np.inf,np.inf]
    else:
        zrange = [np.float(sys.argv[2]),np.float(sys.argv[3])]

    classnames = pd.read_table(classname_file,sep='\s+',header=0)
    print(classnames)
    classnames = classnames.sort_values(by='TARGET')
    print(classnames, 'sorted')
    classnames = classnames['MODEL_NAME']
    print(classnames,'modelname')

    print(newinds)
    classnames = classnames[newinds]
    print(classnames, 'check')
    ztable = pd.read_csv(z_file,sep='\t',header=0)
    ztable.rename(columns={'SNID':'object_id'},inplace=True)
    ztable = ztable[(ztable['SIM_REDSHIFT_HOST']>zrange[0]) & (ztable['SIM_REDSHIFT_HOST']<=zrange[1])]
    print("zrange = {}, {}".format(np.min(ztable['SIM_REDSHIFT_HOST']),np.max(ztable['SIM_REDSHIFT_HOST'])))

    print("Reading in submission")
    probs = pd.read_csv(submission_file,sep=',',header=0)
    probs.rename(columns={x: x.split('_')[1] for x in list(probs) if x != 'object_id'},inplace=True)
    print("Done reading in submission:",submission_file)
    truth = pd.read_csv(truth_file,sep=',',header=0)

    truth = truth.join(ztable.set_index('object_id'),on='object_id',how='inner')
    probs = probs.join(ztable.set_index('object_id'),on='object_id',how='inner')
    probs.sort_values(by='object_id',inplace=True)
    truth.sort_values(by='object_id',inplace=True)

    classids = [x for x in list(probs) if x[0].isdigit()]
    print(classids,'classids')
#    newclassids = [x for x in classids if

    y_pred = probs[classids].idxmax(axis=1).values.astype(int)
#    print(y_pred[0:10])
    y_true = truth['target'].values
 #   print(y_true[0:10])

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    classidx = np.searchsorted(classlist,np.unique([y_pred,y_true]))
    classidx = np.sort(classidx)

    classes = classnames.reset_index(drop=True)[classidx]

    # # Plot non-normalized confusion matrix
    # fig = plt.figure(figsize=(10,10))
    # plot_confusion_matrix(cnf_matrix, classes=classes,
    #                       title='Confusion matrix, without normalization')
    # plt.show()

    # Plot normalized confusion matrix
    fig = plt.figure(figsize=(10,10))

    inds = newinds
    print(cnf_matrix[0,:])
    plot_confusion_matrix(cnf_matrix, classes=classes, newinds=inds,normalize=True,
                          title='Normalized confusion matrix-z{}_{}-{}'.format(zrange[0],zrange[1],filenum))
    fig.savefig("{}_z{}_{}.pdf".format(submission_file,zrange[0],zrange[1]))
    
    f = open("{}_z{}_{}.txt".format(submission_file,zrange[0],zrange[1]),'w')
    np.savetxt(f, np.transpose(cnf_matrix), delimiter=' ')
    f.close()               

if __name__ == "__main__":
    main()
