import numpy as np
 
if __name__=='__main__':
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.concatenate((a, b))
    print(c)

    d = np.array([[1, 2, 3], [4, 5, 6]])
    np.random.permutation(d)
    print(d)