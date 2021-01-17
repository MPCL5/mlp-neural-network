import random
import active_functions


class MLP:
    first_weights = []
    secound_weights = []

    def __init__(self, input_count, hidden_count, output_count, alpha):
        # Step 1. random weightd between -1,1
        self.first_weights = [
            [random.random() * 2 - 1 for i in range(input_count + 1)] for j in hidden_count]
        self.secound_weights = [
            [random.random() * 2 - 1 for i in range(hidden_count + 1)] for j in output_count]
        self.output_count = output_count
        self.alpha = alpha

    def __calculate_net_input(self, inputs, weights):
        result = weights[len(inputs)]  # equal to bias
        for i in range(len(inputs)):
            result += weights[i] * inputs[i]

        return result

    def forward(self, inputs):
        # Step 4,5.
        first_net_inputs = []
        for hidden_neuron in self.first_weights:
            output = self.__calculate_net_input(inputs, hidden_neuron)
            first_net_inputs.append(output)

        # Step 6.
        secount_outputs = []
        for visible_neuron in self.secound_weights:
            # becase active function is identity function. I don't use active function here to improve performance
            output = self.__calculate_net_input(first_net_inputs, visible_neuron)
            secount_outputs.append(active_functions.bipolar_sigmoid(output))

        return secount_outputs

    def backward(self, cases):
        # Step 3.
        for case in cases:
            # Step 4,5.
            first_net_inputs = []
            for hidden_neuron in self.first_weights:
                output = self.__calculate_net_input(
                    case['inputs'], hidden_neuron)
                first_net_inputs.append(output)

            # Step 6.
            secound_net_inputs = []
            secount_outputs = []
            for visible_neuron in self.secound_weights:
                # becase active function is identity function. I don't use active function here to improve performance
                output = self.__calculate_net_input(
                    first_net_inputs, visible_neuron)
                secound_net_inputs.append(output)
                secount_outputs.append(
                    active_functions.bipolar_sigmoid(output))

            # Step 7.
            delta_k = []
            for i in range(len(self.secound_weights)):
                little_delta = ((1 if i+1 == case['target'] else -1) - secount_outputs[i]) * \
                    active_functions.derivative_bipolar_sigmoid(
                        secound_net_inputs[i])

                delta_k.append(little_delta)

            # Step 8.
            delta_j = []
            for j in range(len(self.first_weights)):
                # becase derivative of identy fucntion is 1 I don't consider it.
                little_delta = 0
                for k in range(len(self.secound_weights)):
                    little_delta += self.secound_weights[k][j] * delta_k[k]

                delta_j.append(little_delta)

            # Step 9.
            for i in range(len(self.secound_weights)):
                for j in range(len(self.secound_weights[i]) - 1):
                    self.secound_weights[i][j] = self.alpha * \
                        delta_k[i] * secount_outputs[i][j]

                # calculate bias.
                self.secound_weights[i][-1] = self.alpha * delta_k[i]

            for i in range(len(self.first_weights)):
                for j in range(len(self.first_weights[i]) - 1):
                    self.first_weights[i][j] = self.alpha * \
                        delta_j[i] * secount_outputs[i][j]

                # calculate bias.
                self.first_weights[i][-1] = self.alpha * delta_j[i]
