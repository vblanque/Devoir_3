import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if(nn.as_scalar(self.run(x))) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        # Initialize the mistake flag
        mistake = True

        # Continue training until no mistakes are made
        while mistake:
            # Reset the mistake flag
            mistake = False

            # Iterate over the dataset
            for x, y in dataset.iterate_once(1):
                # Check if the prediction is incorrect
                if self.get_prediction(x) != nn.as_scalar(y):
                    # Update the weights
                    self.w.update(x, nn.as_scalar(y))

                    # Set the mistake flag to True
                    mistake = True

        
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        self.maximal_loss = 0.01 #0.02 is enough 
        self.batch_size = 10  # Set the batch size for training
        self.learning_rate = 0.05  # Set the learning rate
        self.n = 3  # Set the number of layers
        self.dim = 100  # Set the dimension of each layer
        self.w = []  # Initialize the list to store the weights
        self.b = []  # Initialize the list to store the biases
        for i in range(self.n):
            if(i==0):
                self.w.append(nn.Parameter(1, self.dim))  # weights for the first layer
                self.b.append(nn.Parameter(1, self.dim))  # biases for the first layer
            elif(i==self.n-1):
                self.w.append(nn.Parameter(self.dim, 1))  # weights for the last layer
                self.b.append(nn.Parameter(1, 1))  # biases for the last layer
            else:
                self.w.append(nn.Parameter(self.dim, self.dim))  # weights for the hidden layers
                self.b.append(nn.Parameter(1, self.dim))  # biases for the hidden layers

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Apply ReLU activation function to the input layer
        res = nn.ReLU(nn.AddBias(nn.Linear(x, self.w[0]), self.b[0]))

        # Apply ReLU activation function to the hidden layers
        for i in range(1, self.n-1, 1):
            res = nn.ReLU(nn.AddBias(nn.Linear(res, self.w[i]), self.b[i]))

        # Apply linear transformation to the output layer
        return nn.AddBias(nn.Linear(res, self.w[self.n-1]), self.b[self.n-1])

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        # Calculate the predicted y-values
        predicted_y = self.run(x)

        # Compute the square loss between predicted y-values and true y-values
        loss = nn.SquareLoss(predicted_y, y)

        return loss

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        total_loss = 0  # Initialize the total loss variable
        epoch_size = dataset.x.shape[0]  # Get the size of the dataset
        iteration = 0  # Initialize the iteration counter

        # Iterate over the dataset indefinitely
        for x, y in dataset.iterate_forever(self.batch_size):
            iteration += 1  # Increment the iteration counter

            # Check if we have completed a full epoch
            if iteration > epoch_size:
                iteration = 1  # Reset the iteration counter
                total_loss = 0  # Reset the total loss

            loss = self.get_loss(x, y)  # Calculate the loss
            total_loss += nn.as_scalar(loss)  # Update the total loss

            grad = nn.gradients(loss, self.w + self.b)  # Calculate the gradients

            # Update the weights and biases for each layer
            for i in range(self.n):
                self.w[i].update(grad[i], -self.learning_rate)  # Update the weights
                self.b[i].update(grad[self.n + i], -self.learning_rate)  # Update the biases

            # Check if we have completed a full epoch and the average loss is below the threshold
            if iteration == epoch_size and total_loss / iteration < self.maximal_loss:
                break  # Exit the training loop


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        self.required_precsision = 0.974
        self.input_size = 784  # Set the input size
        self.output_size = 10 # Set the output size
        self.batch_size = 200  # Set the batch size for training
        self.learning_rate = 0.6  # Set the learning rate
        self.n = 4  # Set the number of layers
        self.dim = 300  # Set the dimension of each layer
        self.w = []  # Initialize the list to store the weights
        self.b = []  # Initialize the list to store the biases
        for i in range(self.n):
            if(i==0):
                self.w.append(nn.Parameter(self.input_size, self.dim))  # weights for the first layer
                self.b.append(nn.Parameter(1, self.dim))  # biases for the first layer
            elif(i==self.n-1):
                self.w.append(nn.Parameter(self.dim, self.output_size))  # weights for the last layer
                self.b.append(nn.Parameter(1, self.output_size))  # biases for the last layer
            else:
                self.w.append(nn.Parameter(self.dim, self.dim))  # weights for the hidden layers
                self.b.append(nn.Parameter(1, self.dim))  # biases for the hidden layers

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Apply ReLU activation function to the input layer
        res = nn.ReLU(nn.AddBias(nn.Linear(x, self.w[0]), self.b[0]))

        # Apply ReLU activation function to the hidden layers
        for i in range(1, self.n-1, 1):
            res = nn.ReLU(nn.AddBias(nn.Linear(res, self.w[i]), self.b[i]))

        # Apply linear transformation to the output layer
        return nn.AddBias(nn.Linear(res, self.w[self.n-1]), self.b[self.n-1])

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        # Calculate the predicted y-values
        predicted_y = self.run(x)

        # Compute the square loss between predicted y-values and true y-values
        loss = nn.SoftmaxLoss(predicted_y, y)

        return loss

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        epoch_size = dataset.x.shape[0]/self.batch_size  # Get the size of the epoch (nbr of batch)
        iteration = 0  # Initialize the iteration counter
        print(f"epoch_size = {epoch_size}")

        # Iterate over the dataset indefinitely
        for x, y in dataset.iterate_forever(self.batch_size):
            iteration += 1  # Increment the iteration counter
            print(f"iteration = {iteration}")

            # Check if we have completed a full epoch
            if iteration > epoch_size:
                iteration = 1  # Reset the iteration counter

            loss = self.get_loss(x, y)  # Calculate the loss
            grad = nn.gradients(loss, self.w + self.b)  # Calculate the gradients

            # Update the weights and biases for each layer
            for i in range(self.n):
                self.w[i].update(grad[i], -self.learning_rate)  # Update the weights
                self.b[i].update(grad[self.n + i], -self.learning_rate)  # Update the biases
            
            # Check if we have completed a full epoch and the average loss is below the threshold
            if iteration == epoch_size:
                accuracy = dataset.get_validation_accuracy()
                print(f"validation_accuracy = {accuracy}")
                # Adjust lerning rate with current accuracy
                if accuracy > 0.91:
                    self.learning_rate = 0.4
                if accuracy > 0.96:
                    self.learning_rate = 0.3
                # Stop the training
                if accuracy > self.required_precsision:
                    break  # Exit the training loop
