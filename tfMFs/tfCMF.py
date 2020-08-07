import numpy as np
import tensorflow as tf
from sklearn.utils.extmath import randomized_svd


class tfCMF:
    def __init__(self, X, alpha=0.7, d_hidden=5, lamda=1, left=lambda x: x ** 2 / 2, right=tf.math.softplus):
        """ This is a class to decompose R(n × m) ~ U(n × d) @  V (d × m) and X(n × l) ~ U(n × d) @ Z(d × l)

        Args:
            X (matrix): side information matrix. The shape is (n × l)
            alpha (float, optional): hyper-parameter of defining loss rate. Defaults to 0.7.
            d_hidden (int, optional): num of decomposing dimention. Defaults to 5.
            lamda (int, optional): hyper-parameter of reguralization. Defaults to 1.
            left (function, optional): link function of R. Defaults to lambdax:x**2/2.
            right (function, optional): link function of X. Defaults to tf.math.softplus.

        x^2/2 is link of mse. (lambda x:x**2/2)
        exp(log(x) + 1) is link of cross entropy. (tf.math.softplus)

        Expmple
        >>> import tfCMF
        >>> import numpy as np
        >>> n,m,d = 1000,2000,5
        >>> train_data = np.random.randint(0,2,[n,m]) #binary matrix
        >>> test_data = train_data
        >>> mask_train = np.array([[((i%2 == 0) & (j%2 == 1)) | ((i%2 == 1) & (j%2 == 0)) for i in range(m)]for j in range(n)])
        >>> mask_test = 1 - mask_train
        >>> X = np.random.random([n,d])
        >>> cmf = tfCMF.tfSingleCMF(
                X, alpha=0.7, d_hidden=5, lamda= 1,
                left=tfCMF.tf.math.softplus, right=lambda x:x**2/2
            )
        >>> U,V, Z, train_rec, test_rec = cmf.train(
                train_data, test_data,
                mask_train = mask_train, mask_test = mask_test,
                optim_steps=100, verbose=10, early_stopping=5,lr=0.005
            )
        """

        self.X = X
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

    def init_matrices(self, R, X):

        U1, V = self.svd_init(R)
        U2, Z = self.svd_init(X)
        U = self.alpha * U1 + (1 - self.alpha) * U2

        return U, V, Z

    def train(self, train_data, test_data, mask_train=None, mask_test=None, optim_steps=20000, verbose=100, early_stopping=5, lr=0.005):
        """[summary]

        Args:
            train_data (matrix): R matrix for train. The shape is (n × m).
            test_data (matrix): R matrix for test. The shape is (n × m).
            mask_train (matrix, optional): mask matrix for train. (If element is False, loss is set to be 0.) The shape is (n × m). Defaults to None.
            mask_test (matrix, optional): mask matrix for test. The shape is (n × m). Defaults to None.
            optim_steps (int, optional): num of epoch. Defaults to 20000.
            verbose (int, optional): num of step for log is displayed. Defaults to 100.
            early_stopping (int, optional): num of step for early_stopping. Defaults to 5.
            lr (float, optional): learning rate. Defaults to 0.005.

        Returns:
            U.numpy() (ndarray): numpy array of U.
            V.numpy() (ndarray): numpy array of V.
            Z.numpy() (ndarray): numpy array of Z.
            train_loss_record (list) : record of train loss.
            test_loss_record (list): record of test loss.
        """

        U, V, Z = self.init_matrices(train_data, self.X)

        U = tf.keras.backend.variable(U, dtype=tf.float32, name="U")
        V = tf.keras.backend.variable(V, dtype=tf.float32, name="V")
        Z = tf.keras.backend.variable(Z, dtype=tf.float32, name="Z")

        R_train = tf.keras.backend.constant(train_data, dtype=tf.float32)
        R_test = tf.keras.backend.constant(test_data, dtype=tf.float32)
        X = tf.keras.backend.constant(self.X, dtype=tf.float32)

        if mask_train is None:
            M_train = tf.keras.backend.constant((train_data != 0).astype(int))
        else:
            M_train = tf.keras.backend.constant(mask_train)

        if mask_test is None:
            M_test = tf.keras.backend.constant((test_data != 0).astype(int))
        else:
            M_test = tf.keras.backend.constant(mask_test)

        def loss(R=R_train, M=M_train, alpha=self.alpha, lamda=self.lamda):
            R_ = M * tf.matmul(U, V)
            loss_R = tf.math.reduce_sum(self.left(R_) - R * R_)

            X_ = tf.matmul(U, Z)
            loss_X = tf.math.reduce_sum(self.right(X_) - X * X_)

            norm = tf.math.reduce_euclidean_norm(U) +\
                tf.math.reduce_euclidean_norm(V) +\
                tf.math.reduce_euclidean_norm(Z)

            loss_all = alpha * loss_R + (1 - alpha) * loss_X + lamda * norm

            return loss_all

        opt = tf.optimizers.Adam(learning_rate=lr)
        stop_time = 0
        times, old_loss = 0, 100000
        train_loss_record = []
        test_loss_record = []

        for times in range(optim_steps):
            loss_train = loss(R=R_train, M=M_train).numpy()
            loss_test = loss(R=R_test, M=M_test).numpy()
            train_loss_record.append(loss_train)
            test_loss_record.append(loss_test)

            if loss_test > old_loss:
                if stop_time == early_stopping:
                    print("[Info] At last ime-step {}, test data rmse loss is {}".format(times, loss_test**0.5))
                    break
                else:
                    stop_time += 1
            else:
                stop_time = 0

            old_loss = loss_test
            if verbose > 0:
                if times % verbose == 0:
                    print("[Info] At time-step {}, test data rmse loss is {}".format(times, loss_test**0.5))

            opt.minimize(loss=loss, var_list=[U, V, Z])

        return U.numpy(), V.numpy(), Z.numpy(), train_loss_record, test_loss_record
