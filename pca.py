import numpy as np


def pca(data, variance_threshold=.95):
    if variance_threshold == 1:
        variance_threshold += 1
    covariance_matrix = data.T @ data
    eigenstuff = np.linalg.eig(covariance_matrix)
    eigenstuff_but_enhanced = sorted(list(zip(*eigenstuff)), key=lambda x: x[0], reverse=True)
    all_variance = sum(map(lambda x: x[0], eigenstuff_but_enhanced))
    eigenstuff_but_enhancedest = [x for i, x in enumerate(eigenstuff_but_enhanced)
                                  if sum(map(lambda x: x[0], eigenstuff_but_enhanced[:i+1])) < all_variance*variance_threshold]

    formtrans = np.asarray(list(map(lambda x: x[1], eigenstuff_but_enhancedest)))
    print(data.shape, formtrans.shape)
    return data @ formtrans.T


if __name__ == '__main__':
    print(pca(np.random.rand(15, 4)))
