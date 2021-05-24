import numpy as np


def pca(data, variance_threshold=.98):
    covariance_matrix = data @ data.T
    # print(covariance_matrix)
    eigenstuff = np.linalg.eig(covariance_matrix)
    eigenstuff_but_enhanced = sorted(list(zip(*eigenstuff)), key=lambda x: x[0], reverse=True)
    all_variance = sum(map(lambda x: x[0], eigenstuff_but_enhanced))
    eigenstuff_but_enhancedest = [x for i, x in enumerate(eigenstuff_but_enhanced)
                                  if sum(map(lambda x: x[0], eigenstuff_but_enhanced[:i])) + eigenstuff_but_enhanced[i][0] < all_variance*variance_threshold]

    formtrans = np.asarray(list(map(lambda x: x[1], eigenstuff_but_enhancedest)))
    return formtrans @ data.T


if __name__ == '__main__':
    print(pca(np.random.rand(10, 4)))
