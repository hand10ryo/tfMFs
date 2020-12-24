import numpy as np
import tensorflow as tf
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


class tfCMF:
    def __init__(self, alpha=0.7, d_hidden=5, lamda=1,
                 left=lambda x: x ** 2 / 2, right=tf.math.softplus):
        """ This is a class of CMF to decompose X(n × m) ~ U(n × d) @  V (d × m) and Y(n × l) ~ U(n × d) @ Z(d × l) simultaneously.
        This paper ( http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf ) shows detail.

        Args:
            alpha (float, optional): hyper-parameter of defining loss rate. Defaults to 0.7.
            d_hidden (int, optional): num of decomposing dimention. Defaults to 5.
            lamda (float, optional): hyper-parameter of l2 reguralization. Defaults to 1.
            left (function, optional): link function of X. Defaults to lambdax:x**2/2.
            right (function, optional): link function of Y. Defaults to tf.math.softplus.

        x^2/2 is link function for mse. (lambda x:x**2/2)
        exp(log(x) + 1) is link function for cross entropy. (tf.math.softplus)
        exp is link function for poisson loss. (tf.math.exp)

        Expmple
        >>> import tfCMF
        >>> import numpy as np
        >>> n,m,d = 1000,2000,5
        >>> X_train = np.random.randint(0,2,[n,m]) # main matrix
        >>> X_valid = X_train
        >>> mask_train = np.array([[((i%2 == 0) & (j%2 == 1)) | ((i%2 == 1) & (j%2 == 0)) for i in range(m)]for j in range(n)])
        >>> mask_valid = 1 - mask_train
        >>> Y = np.random.random([n,d]) # sideinfo matrix
        >>> cmf = tfCMF.tfCMF(
                alpha=0.7, d_hidden=5, lamda= 1,
                left=lambda x:x**2/2, right=tfCMF.tf.math.softplus
            )
        >>> U, V, Z, train_rec, valid_rec = cmf.train(
                X_train, X_valid, Y,
                mask_train = mask_train, mask_valid = mask_valid,
                optim_steps=100, early_stopping=5,lr=0.005
            )
        """

        self.alpha = alpha
        self.lamda = lamda
        self.d = d_hidden
        self.left = left
        self.right = right

    def svd_init(self, M):
        U, S, V = randomized_svd(M, self.d)

        U_padded = np.zeros((U.shape[0], self.d))
        U_padded[:, :U.shape[1]] = U
        U = U_padded

        V_padded = np.zeros((self.d, V.shape[1]))
        V_padded[:V.shape[0], :] = V
        V = V_padded

        S_padded = np.zeros(self.d)
        S_padded[:S.shape[0]] = S
        S = S_padded

        S = np.diag(np.sqrt(S))
        A = np.dot(U, S)
        B = np.dot(S, V)

        return A, B

    def random_init(self, M):
        A = np.random.rand(M.shape[0], self.d)
        B = np.random.rand(self.d, M.shape[1])

        return A, B

    def init_matrices(self, X, Y, init="svd"):
        if init == "svd":
            U1, V = self.svd_init(X)
            U2, Z = self.svd_init(Y)

        else:
            U1, V = self.random_init(X)
            U2, Z = self.random_init(Y)

        U = self.alpha * U1 + (1 - self.alpha) * U2

        return U, V, Z

    def train(self, X_train, X_valid, Y, mask_train=None, mask_valid=None,
              optim_steps=20000, early_stopping=5, lr=0.005, optimizer="Adam", init="svd"):
        """ Method of training.

        Args:
            X_train (matrix): main matrix for train. The shape is (n × m).
            X_valid (matrix): main matrix for validation. The shape is (n × m).
            Y (matrix): side information matrix. The shape is (n × l)
            mask_train (matrix, optional): mask matrix for train. (If element is False, loss is set to be 0.) The shape is (n × m). Defaults to None.
            mask_valid (matrix, optional): mask matrix for valid. The shape is (n × m). Defaults to None.
            optim_steps (int, optional): num of epoch. Defaults to 20000.
            early_stopping (int, optional): num of step for early_stopping. Defaults to 5.
            lr (float, optional): learning rate. Defaults to 0.005.

        Returns:
            U.numpy() (ndarray): numpy array of U.
            V.numpy() (ndarray): numpy array of V.
            Z.numpy() (ndarray): numpy array of Z.
            train_loss_record (list) : record of train loss.
            valid_loss_record (list): record of valid loss.
        """

        U, V, Z = self.init_matrices(X_train, Y, init=init)

        U = tf.keras.backend.variable(U, dtype=tf.float32, name="U")
        V = tf.keras.backend.variable(V, dtype=tf.float32, name="V")
        Z = tf.keras.backend.variable(Z, dtype=tf.float32, name="Z")

        X_train = tf.keras.backend.constant(X_train, dtype=tf.float32)
        X_valid = tf.keras.backend.constant(X_valid, dtype=tf.float32)
        Y = tf.keras.backend.constant(Y, dtype=tf.float32)

        if mask_train is None:
            M_train = tf.keras.backend.constant(
                np.ones(X_train.shape).astype(int))
        elif mask_train == "nonzero":
            M_train = tf.keras.backend.constant((X_train != 0).astype(int))
        else:
            M_train = tf.keras.backend.constant(mask_train)

        if mask_valid is None:
            M_valid = tf.keras.backend.constant(
                np.ones(X_valid.shape).astype(int))
        elif mask_valid == "nonzero":
            M_valid = tf.keras.backend.constant((X_valid != 0).astype(int))
        else:
            M_valid = tf.keras.backend.constant(mask_valid)

        def loss(X=X_train, Y=Y, M=M_train, alpha=self.alpha, lamda=self.lamda):
            X_ = M * tf.matmul(U, V)
            loss_X = tf.math.reduce_sum(self.left(X_) - X * X_)

            Y_ = tf.matmul(U, Z)
            loss_Y = tf.math.reduce_sum(self.right(Y_) - Y * Y_)

            norm = tf.math.reduce_euclidean_norm(U) +\
                tf.math.reduce_euclidean_norm(V) +\
                tf.math.reduce_euclidean_norm(Z)

            loss_all = alpha * loss_X + (1 - alpha) * loss_Y + lamda * norm

            return loss_all

        if optimizer == "Adam":
            opt = tf.optimizers.Adam(learning_rate=lr)

        if optimizer == "SGD":
            opt = tf.optimizers.SGD(learning_rate=lr)
        stop_time = 0
        times, old_loss = 0, 100000
        train_loss_record = []
        valid_loss_record = []

        for times in tqdm(range(optim_steps)):
            loss_train = loss(X=X_train, Y=Y, M=M_train).numpy()
            loss_valid = loss(X=X_valid, Y=Y, M=M_valid).numpy()
            train_loss_record.append(loss_train)
            valid_loss_record.append(loss_valid)

            if loss_valid > old_loss:
                if stop_time == early_stopping:
                    break
                else:
                    stop_time += 1
            else:
                stop_time = 0

            old_loss = loss_valid

            opt.minimize(loss=loss, var_list=[U, V, Z])

        return U.numpy(), V.numpy(), Z.numpy(), train_loss_record, valid_loss_record
