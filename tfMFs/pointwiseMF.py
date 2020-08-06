import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd


class PointwiseMF(tf.keras.layers.Layer):
    """ This class is Custom Layer for PointwiseMF
    """
    def __init__(self, num_users, num_items, dim, U_init, V_init, **kwargs):
        """ initialize layer

        Args:
            num_users (int): num of users
            num_items (int): num of items
            dim (int): dimention of latent space
            U_init (matrix): matrix for initializing user embeddings
            V_init (matrix): matrix for initializing user embeddings
        """
        super(PointwiseMF, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.U_init = U_init
        self.V_init = V_init
        self.build()

    def build(self):
        """ build variables """
        self.user_embeddings = tf.Variable(
            self.U_init,
            name='user_embeddings',
            shape=(self.num_users, self.dim),
            dtype=tf.float32,
        )
        self.item_embeddings = tf.Variable(
            self.V_init,
            name='item_embeddings',
            shape=(self.num_items, self.dim),
            dtype=tf.float32,
        )
        super(PointwiseMF, self).build(None)

    def call(self, users, items):
        """ call function that returns rating from user_id amd item_id

        Args:
            users (ndarray): array of user_id
            items (ndarray): array of itemid

        Returns:
            r_hats : a estimated values of rating (or relevance)
        """
        with tf.name_scope('pointwise'):
            self.u_embed = tf.nn.embedding_lookup(self.user_embeddings, users)
            self.i_embed = tf.nn.embedding_lookup(self.item_embeddings, items)
            self.r_hats = tf.reduce_sum(tf.matmul(self.u_embed, tf.transpose(self.i_embed)), 1)
            return self.r_hats

    def get_user_embeddings(self, users):
        """ This is a function that returns user embeddings

        Args:
            users (ndarray): array of user_id

        Returns:
            u_embed (ndarray): numpy array of user embeddings
        """
        u_embed = tf.nn.embedding_lookup(self.user_embeddings, users)
        return u_embed.numpy()

    def get_item_embeddings(self, items):
        """ This is a function that returns item embeddings

        Args:
            items (ndarray): array of item id

        Returns:
            i_embed (ndarray): numpy array of item embeddings
        """
        i_embed = tf.nn.embedding_lookup(self.item_embeddings, items)
        return i_embed.numpy()


class AbstractMF(tf.keras.Model):
    def __init__(self, data, num_users, num_items, dim, **kwargs):
        """ Simplest matrix factorization model. This is a abstract class.
            You can build a MF model by inheritance of this class.

        Args:
            data ([ndarray]): matrix whose columns are [user_id, item_id, rating].
            num_users ([int]): num of users/
            num_items ([int]): num of items.
            dim ([int]): a dim of latent space.
        """
        super(AbstractMF, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.num_dsample = data.shape[0]
        self.dim = dim
        self.users = data[:, 0]
        self.items = data[:, 1]
        self.ratings = data[:, 2]
        self.R = csr_matrix((self.ratings, (self.users, self.items)), shape=(self.num_users, self.num_items))
        self.U_init, self.V_init = self.svd_init(self.R, self.dim)

        self.PointwiseMF = PointwiseMF(num_users, num_items, dim, self.U_init, self.V_init.T)

    def svd_init(self, M, dim):
        """ A function to intialize user and item embeddings by SVD.

        Args:
            M (matrix): matrix to decompose
            dim (int): a dim of latent space.

        Returns:
            A : left matrix.
            B : rithg matrix.
        """
        U, S, V = randomized_svd(M, dim)
        U_padded = np.zeros((U.shape[0], dim))
        U_padded[:, :U.shape[1]] = U
        U = U_padded

        V_padded = np.zeros((dim, V.shape[1]))
        V_padded[:V.shape[0], :] = V
        V = V_padded

        S_padded = np.zeros(dim)
        S_padded[:S.shape[0]] = S
        S = S_padded

        S = np.diag(np.sqrt(S))
        A = np.dot(U, S)
        B = np.dot(S, V)

        return A, B

    def call(self, users, items):
        """ This is a call function that returns rating from user_id amd item_id

        Args:
            users (ndarray): array of user_id
            items (ndarray): array of itemid

        Returns:
            r_hats (ndarray): a estimated values of rating (or relevance)
        """
        r_hats = self.PointwiseMF(users, items)
        return r_hats

    def fit(self):
        pass

    def transform_prob(self, users, items):
        r_hats = self.PointwiseMF(users, items)
        r_hats_prob = tf.math.sigmoid(r_hats)
        return r_hats_prob.numpy()

    def get_user_embeddings(self, users):
        return self.PointwiseMF.get_user_embeddings(users)

    def get_item_embeddings(self, items):
        return self.PointwiseMF.get_item_embeddings(items)

    def get_user_embeddings_all(self):
        users = np.arange(self.num_users)
        return self.PointwiseMF.get_user_embeddings(users)

    def get_item_embeddings_all(self, items):
        items = np.arange(self.num_items)
        return self.PointwiseMF.get_item_embeddigns(items)


class BinaryMF(AbstractMF):
    def __init__(self, data, num_users, num_items, dim, **kwargs):
        """ Simplest binary matrix factorization model.

        Args:
            data ([ndarray]): matrix whose columns are [user_id, item_id, rating].
            num_users ([int]): num of users/
            num_items ([int]): num of items.
            dim ([int]): a dim of latent space.
        """
        super(BinaryMF, self).__init__(data, num_users, num_items, dim, **kwargs)

    def fit(self, max_iter=10, lr=1.0e-4, n_batch=256, verbose=False, verbose_freq=50):
        """ This is a function to run training.

        Args:
            max_iter (int, optional): max iteration. Defaults to 10.
            lr (float, optional): learning rate. Defaults to 1.0e-4.
            n_batch (int, optional): batch size. Defaults to 256.
            verbose (bool, optional): whether displaying loss or not. Defaults to False.
            verbose_freq (int, optional): frequency of displaying loss. Defaults to 50.
        """
        optimizer = tf.optimizers.Adam(lr)
        train_loss = tf.keras.metrics.Mean()
        bce = tf.keras.losses.BinaryCrossentropy()

        @tf.function
        def train_step(users, items, ratings):
            with tf.GradientTape() as tape:
                r_hats = self.PointwiseMF(users, items)
                loss = bce(ratings, tf.math.sigmoid(r_hats))
            grad = tape.gradient(loss, sources=self.PointwiseMF.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.PointwiseMF.trainable_variables))
            train_loss.update_state(loss)
            return None

        for i in range(max_iter):
            indices = np.random.choice(np.arange(self.num_dsample), n_batch)
            users = self.users[indices].astype(int)
            items = self.items[indices].astype(int)
            ratings = self.ratings[indices]
            train_step(users, items, ratings)
            if verbose and (i % verbose_freq == 0):
                print(f"iter {i} loss : {train_loss.result().numpy()}")

        print(f"iter {i} loss : {train_loss.result().numpy()}")
