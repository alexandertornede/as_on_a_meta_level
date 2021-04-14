import pickle


# save base learner for later use
def save_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)


# save base learner for later use
def load_pickle(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
