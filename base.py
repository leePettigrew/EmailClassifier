"""
base.py

Defines the BaseModel abstract class which enforces that every model
implements the train, predict, and print_results methods. This abstraction
ensures that the main controller can interact with any model through a uniform interface.
"""

class BaseModel:
    def train(self, data):
        """
        Train the model using the data provided.
        :param data: Data object containing training features and labels.
        """
        raise NotImplementedError("train() must be implemented by subclasses.")

    def predict(self, data):
        """
        Generate predictions using the trained model.
        :param data: Data object containing testing features.
        :return: Predictions in a consistent format (e.g., numpy array with columns for y2, y3, y4).
        """
        raise NotImplementedError("predict() must be implemented by subclasses.")

    def print_results(self, data):
        """
        Evaluate the model's predictions against true labels and print the results.
        :param data: Data object containing testing features and true labels.
        """
        raise NotImplementedError("print_results() must be implemented by subclasses.")
