from __future__ import print_function
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import sys

def main():

    # read in data
    submission_file = sys.argv[1]
    outprefix = sys.argv[2]
    filenum = submission_file.split('.csv')[0]
    truth_file = "plasticc_test_truthtable.csv"
    classname_file = "classnames.txt"
    z_file = "redshifts.tsv"
    classlist = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
    if len(sys.argv)<5:
        zrange = [-np.inf,np.inf]
    else:
        zrange = [np.float(sys.argv[3]),np.float(sys.argv[4])]

    classnames = pd.read_table(classname_file,sep='\s+',header=0)
    classnames = classnames.sort_values(by='TARGET')
    classnames = classnames['MODEL_NAME']

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
    y_pred = probs[classids].idxmax(axis=1).values.astype(int)
    print(y_pred[0:10])
    y_true = truth['target'].values
    print(y_true[0:10])

    probs[list(['object_id'])+classids].to_csv(outprefix+'_probs.csv.gz',index=False,compression='gzip')
    truth[['object_id','target']].to_csv(outprefix+'_truth.csv.gz',index=False,compression='gzip') 

    # # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)

    # classidx = np.searchsorted(classlist,np.unique([y_pred,y_true]))
    # classidx = np.sort(classidx)

    # classes = classnames.reset_index(drop=True)[classidx]

    # # # Plot non-normalized confusion matrix
    # # fig = plt.figure(figsize=(10,10))
    # # plot_confusion_matrix(cnf_matrix, classes=classes,
    # #                       title='Confusion matrix, without normalization')
    # # plt.show()

    # # Plot normalized confusion matrix
    # fig = plt.figure(figsize=(10,10))

    # plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
    #                       title='Normalized confusion matrix-z{}_{}-{}'.format(zrange[0],zrange[1],filenum))
    # fig.savefig("{}_z{}_{}.pdf".format(submission_file,zrange[0],zrange[1]))


if __name__ == "__main__":
    main()