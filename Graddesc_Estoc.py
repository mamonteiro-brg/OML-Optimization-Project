import numpy  as np

np.random.seed(23)

# implementa regressão linear com gradiente descendente estocástico
class linear_regr(object):

    def __init__(self, learning_rate=0.0001, batch_size=5, training_iters=50):
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size

    def fit(self, X_train, y_train, plot=False):

        # formata os dados
        if len(X_train.values.shape) < 2:
            X = X_train.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        # inicia os parâmetros com pequenos valores aleatórios
        # (nosso chute razoável)
        self.w_hat = np.random.normal(0,5, size = X[0].shape)

        for i in range(self.training_iters):

            # cria os mini-lotes
            offset = (i * self.batch_size) % (y_train.shape[0] - self.batch_size)
            batch_X = X[offset:(offset + self.batch_size), :]
            batch_y = y_train[offset:(offset + self.batch_size)]

            gradient = np.zeros(self.w_hat.shape) # inicia o gradiente

            # atualiza o gradiente com informação dos pontos do lote
            for point, yi in zip(batch_X, batch_y):
                gradient +=  (point * self.w_hat - yi) * point

            gradient *= self.learning_rate
            self.w_hat -= gradient

    def predict(self, X_test):
        # formata os dados
        if len(X_test.values.shape) < 2:
            X = X_test.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        return np.dot(X, self.w_hat)

regr = linear_regr(learning_rate=0.0003, training_iters=40)
regr.fit(dados['x'], dados['y'])
