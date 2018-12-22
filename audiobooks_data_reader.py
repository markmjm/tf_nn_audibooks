import numpy as np


# Create a class that will do the batching for the algorithm
# This code is extremely reusable. You should just change Audiobooks_data everywhere in the code
class Audiobooks_Data_Reader():
    # Dataset is a mandatory arugment, while the batch_size is optional
    # If you don't input batch_size, it will automatically take the value: None
    def __init__(self, dataset, batch_size=None):

        # The dataset that loads is one of "train", "validation", "test".
        # e.g. if I call this class with x('train',5), it will load 'Audiobooks_data_train.npz' with a batch size of 5.
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))

        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size

    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()

        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1

        # One-hot encode the targets. In this example it's a bit superfluous since we have a 0/1 column
        # as a target already but we're giving you the code regardless, as it will be useful for any
        # classification task with more than one target column
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1

        # The function will return the inputs batch and the one-hot encoded targets
        return inputs_batch, targets_one_hot

    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data:
    # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self