import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from tensorflow.keras.models import Sequential


class AROOptimizer:
    def __init__(self, model):
        self.model = model

    def loss_function(self, hyperparameters, X, y):
        # Set the hyperparameters
        lstm_units = int(hyperparameters[0])
        dense_units = int(hyperparameters[1])

        # Build the LSTM model with the given hyperparameters
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(lstm_units, return_sequences=True))
        self.model.add(LSTM(lstm_units, return_sequences=False))
        self.model.add(Dense(dense_units))
        self.model.add(Dense(1))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model and return the loss
        self.model.fit(X, y, epochs=10, batch_size=16, verbose=0)
        y_pred = self.model.predict(X)
        loss = mean_squared_error(y, y_pred)
        return loss

    def optimize(self, X, y):
        # Define the parameter bounds for optimization
        parameter_bounds = [(16, 128), (4, 32)]

        # Define the optimization problem
        problem = {
            'name': 'LSTM Hyperparameter Optimization',
            'loss': self.loss_function,
            'dimensions': parameter_bounds,
            'args': (X, y),
        }

        # Run the optimization using the Nelder-Mead method
        result = minimize(self.loss_function, x0=np.array([64, 8]), args=(X, y), method='Nelder-Mead')

        # Print the optimized hyperparameters
        print('Optimized Hyperparameters:')
        print('LSTM Units:', int(result.x[0]))
        print('Dense Units:', int(result.x[1]))
