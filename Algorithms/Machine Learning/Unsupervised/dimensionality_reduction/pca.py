"""Each feature in our data corresponds to a dimension in our problem space. Minimizing the
       number of features to make our problem space simpler is called dimensionality reduction.

       It can be done in one of the following two ways:
            Feature selection: Selecting a set of features that are important in the context of
                                the problem we are trying to solve
            Feature aggregation: Combining two or more features to reduce dimensions
                                    using one of the following algorithms:
                           PCA: A linear unsupervised ML algorithm
                           Linear discriminant analysis (LDA): A linear supervised ML algorithm
                           Kernel principal component analysis: A nonlinear algorithm
""" 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal');
# plt.show()

pca = PCA(n_components=2)
pca.fit(X)

X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# # plot data
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal');



# PCA for visualization 

digits = load_digits()

pca = PCA(2)
projected = pca.fit_transform(digits.data)

print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_stern', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()