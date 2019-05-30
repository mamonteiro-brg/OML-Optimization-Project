import pandas as pd
import numpy as np # para álgebra linear
np.random.seed(0) # para consistência nos resultados

dados = pd.DataFrame()
dados['x'] = np.linspace(-10,10,100)
dados['y'] = 5 + 3*dados['x'] + np.random.normal(0,3,100)

# define a função custo
def L(y, y_hat):
    return ((y-y_hat) ** 2).sum()

# implementa regressão linear com gradiente descendente
class linear_regr(object):

    def __init__(self, learning_rate=0.0001, training_iters=50):
        self.learning_rate = learning_rate
        self.training_iters = training_iters

    def fit(self, X_train, y_train):

        # formata os dados
        if len(X_train.values.shape) < 2:
            X = X_train.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        # inicia os parâmetros com pequenos valores aleatórios
        # (nosso chute razoável)
        self.w_hat = np.random.normal(0,5, size = X[0].shape)

        for _ in range(self.training_iters):

            gradient = np.zeros(self.w_hat.shape) # inicia o gradiente

            # computa o gradiente com informação de todos os pontos
            for point, yi in zip(X, y_train):
                gradient +=  (point * self.w_hat - yi) * point

            # multiplica o gradiente pela taxa de aprendizado
            gradient *= self.learning_rate

            # atualiza os parâmetros
            self.w_hat -= gradient

    def predict(self, X_test):
        # formata os dados
        if len(X_test.values.shape) < 2:
            X = X_test.values.reshape(-1,1)
        X = np.insert(X, 0, 1, 1)

        return np.dot(X, self.w_hat)

regr = linear_regr(learning_rate=0.0005, training_iters=30)
regr.fit(dados['x'], dados['y'])
