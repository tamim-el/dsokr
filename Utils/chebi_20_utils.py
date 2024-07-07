from sklearn.metrics.pairwise import rbf_kernel

# Defining Gaussian kernel
def gaussian_kernel(gamma):
    def Compute_Gram(X, Y=None):
        if Y is None:
            Y = X.copy()
        return rbf_kernel(X, Y, gamma=gamma)

    return Compute_Gram