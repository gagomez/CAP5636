# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        maxWeights = None
        maxC = -1
        maxCAccuracy = -1
        for c in Cgrid:

            weights = {}
            for label in self.legalLabels:
                weights[label] = util.Counter()

            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):

                    features = trainingData[i]
                    prediction = self.classsifyWithWeights([features], weights)[0]
                    actual = trainingLabels[i]

                    if prediction != actual:
                        norm = features * features

                        t = (((weights[prediction] - weights[actual]) * features) + 1.0) / (2 * norm)

                        t = min(t, c)

                        for key in features:
                            f = t * features[key]
                            weights[prediction][key] = weights[prediction][key] - f
                            weights[actual][key] = weights[actual][key] + f

            accuracy = 0
            predictions = self.classsifyWithWeights(validationData, weights)
            for i in range(len(validationData)):
                if predictions[i] == validationLabels[i]:
                    accuracy += 1

            print accuracy,'/',len(validationData)

            if accuracy == maxCAccuracy:
                if c < maxC:
                    maxC = c
                    maxWeights = weights
            elif accuracy > maxCAccuracy:
                maxC = c
                maxWeights = weights
                maxCAccuracy = accuracy

        self.weights = maxWeights

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        return self.classsifyWithWeights(data, self.weights)

    def classsifyWithWeights(self, data, weights):

        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses