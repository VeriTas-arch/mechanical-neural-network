import numpy as np
 
if __name__=='__main__':
    a = np.array([[1, 2, 3],[3,4,5]])
    b = np.array([[4, 5, 6],[7,8,9]])
    c = np.concatenate((a, b), axis=None).reshape(3,4)
    print(c)

    d = np.array([[1, 2, 3], [4, 5, 6]])
    np.random.permutation(d)
    print(d)