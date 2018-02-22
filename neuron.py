import random

class Neuron:

    def __init__(self):
        self.weights = []
        self.threshold = 0
        self.learning_rate = 0.2

    def generate_weights(self, num_inputs):
        weight_count = 0
        while weight_count < num_inputs:
            self.weights.append(random.uniform(-.25, .25))
            weight_count += 1
        # append the bias node
        self.weights.append(random.uniform(-.25, .25))

    def predict_output(self, inputs, target):
        input_sum = 0
        for index, item in enumerate(inputs):
            input_sum += inputs[index] * self.weights[index]
        # finally add the value for the bias node
        input_sum += (-1 * self.weights[-1])
        if input_sum > self.threshold:
            predicted_target = 1
        else:
            predicted_target = 0
        # if the output was incorrect then we need to adjust the weights for the neuron
        if predicted_target != target:
            self.adjust_weights(target, inputs, predicted_target)

        return predicted_target

    def adjust_weights(self, target, inputs, predicted_target):

        for index, input in enumerate(inputs):
            self.weights[index] = self.weights[index] - (self.learning_rate*(predicted_target - target) * inputs[index])

        # adjust bias node weight
        self.weights[-1] = self.weights[-1] - (self.learning_rate*(predicted_target - target) * -1)


def main():
    """Uses our neuron class to predict the correct output by training it and adjusting the weights"""
    # inputs represent all combinations of a 3-input or logic gate
    or_inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    # targets are based off of or gate output
    targets = [0, 1, 1, 1, 1, 1, 1, 1]
    neuron = Neuron()
    # create weights for each input we have (in this case 3)
    neuron.generate_weights(num_inputs=3)
    predicted_targets = []
    # have the neuron predict the outputs 100 times, adjusting the weights as needed
    for i in range(100):

        for index, row in enumerate(or_inputs):
            neuron.predict_output(row, targets[index])

    # after training output the final predicted targets
    for index, row in enumerate(or_inputs):
        predicted_targets.append(neuron.predict_output(row, targets[index]))
    # show if the predictions match what was expected
    print(neuron.weights)
    print(predicted_targets)


main()
