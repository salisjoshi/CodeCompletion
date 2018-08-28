import tflearn
import numpy
import tensorflow as tf

class Code_Completion_Baseline:

    def __init__(self):
        self.max_sequence = 5

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def two_hot(self, prefix, suffix):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[prefix]] = 1
        vector[self.string_to_number[suffix]] = 1
        return vector

    def zero_hot(self):
        return [0] * len(self.string_to_number)

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        print(self.number_to_string)
        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                prefix_sequences = []
                suffix_sequences = []
                if self.max_sequence <= idx < len(token_list) - self.max_sequence:
                    prefix_sequences = token_list[idx - self.max_sequence:idx]
                    suffix_sequences = token_list[idx + 1: idx + self.max_sequence + 1]
                    output_token_string = self.token_to_string(token_list[idx])
                    temp_xs = prefix_sequences + suffix_sequences
                    xs.append([self.string_to_number[self.token_to_string(token)] for token in temp_xs])
                    ys.append(self.one_hot(output_token_string))

        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, self.max_sequence * 2, 1])
        self.net = tflearn.lstm(self.net, 192, return_seq=True)
        self.net = tflearn.lstm(self.net, 192)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        print("Loading saved model")
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
        print("model loaded")

    def train(self, token_lists, model_file):
        print("training model")
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        xs = numpy.reshape(numpy.array(xs), [-1, self.max_sequence * 2, 1])
        self.model.fit(xs, ys, show_metric=True, validation_set=0.1, shuffle=True, batch_size=512, run_id='lstm_all_data')
        self.model.save(model_file)
        print("model saved")

    def query(self, prefix, suffix):
        input_token = []
        prefix_len = len(prefix)
        suffix_len = len(suffix)
        pre_seq_len = post_seq_len = self.max_sequence

        while pre_seq_len > prefix_len:
            input_token.append(-1)
            pre_seq_len -= 1

        if pre_seq_len > 0:
            temp = prefix[prefix_len - pre_seq_len:]
            input_token.extend([self.string_to_number[self.token_to_string(token)] for token in temp])

        temp = []
        while post_seq_len > suffix_len:
            temp.append(-1)
            post_seq_len -= 1

        if post_seq_len > 0:
            temp = [self.string_to_number[self.token_to_string(token)] for token in suffix[:post_seq_len]] + temp

        input_token.extend(temp)

        input_token = numpy.reshape(numpy.array(input_token), [-1, self.max_sequence * 2, 1])

        y = self.model.predict(input_token)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)

        return [best_token]
