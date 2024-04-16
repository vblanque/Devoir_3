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
        self.batch_size = 10
        self.learning_rate = 0.0005 * self.batch_size
        self.n = 3
        self.dim = 100
        self.w = []
        self.b = []
        for i in range(self.n):
            if(i==0):
                self.w.append(nn.Parameter(1, self.dim))
                self.b.append(nn.Parameter(1, self.dim))
            elif(i==self.n-1):
                self.w.append(nn.Parameter(self.dim, 1))
                self.b.append(nn.Parameter(1, 1))
            else:
                self.w.append(nn.Parameter(self.dim, self.dim))
                self.b.append(nn.Parameter(1, self.dim))

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        res = nn.ReLU(nn.AddBias(nn.Linear(x, self.w[0]), self.b[0]))
        for i in range(1, self.n-1, 1):
            res = nn.ReLU(nn.AddBias(nn.Linear(res, self.w[i]), self.b[i]))
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
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y , y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        total_loss = 0
        epoch_size = dataset.x.shape[0]
        iteration = 0
        for x, y in dataset.iterate_forever(self.batch_size):
            iteration += 1
            if iteration > epoch_size:
                iteration = 1
                total_loss = 0
            loss = self.get_loss(x, y)
            total_loss += nn.as_scalar(loss)
            grad = nn.gradients(loss, self.w + self.b)
            for i in range(self.n):
                self.w[i].update(grad[i], -self.learning_rate)
                self.b[i].update(grad[self.n + i], -self.learning_rate)

            if iteration == epoch_size and total_loss / iteration < 0.02:
                break


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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

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
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
