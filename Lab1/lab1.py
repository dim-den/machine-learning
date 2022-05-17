import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib
import numpy
import scipy
import IPython
import sklearn
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def libversion():
    print("версияPython: {}".format(sys.version))
    print("версияpandas: {}".format(pandas.__version__))
    print("версияmatplotlib: {}".format(matplotlib.__version__))
    print("версияNumPy: {}".format(numpy.__version__))
    print("версияSciPy: {}".format(scipy.__version__))
    print("версияIPython: {}".format(IPython.__version__))
    print("версияscikit-learn: {}".format(sklearn.__version__))

def main():
    iris_dataset = load_iris()
    print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
    print(iris_dataset['DESCR'][:193] + "\n...")
    print("Названияответов: {}".format(iris_dataset['target_names']))
    print("Названияпризнаков:\n{}".format(iris_dataset['feature_names']))
    print("Формамассива data: {}".format(iris_dataset['data'].shape))
    print("Первыепятьстрокмассива data:\n{}".format(iris_dataset['data'][:5]))
    print("Типмассива target: {}".format(type(iris_dataset['target'])))
    print("Формамассива target: {}".format(iris_dataset['target'].shape))
    print("Ответы:\n{}".format(iris_dataset['target']))

    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    print("формамассива X_train: {}".format(X_train.shape))
    print("формамассива y_train: {}".format(y_train.shape))
    print("формамассива X_test: {}".format(X_test.shape))
    print("формамассива y_test: {}".format(y_test.shape))

    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                            alpha=.8, cmap=mglearn.cm3)
    plt.show()


if __name__ == '__main__':
    main()