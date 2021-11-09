"""
data_utils.py: script for data handling during runtime
"""

import torch


class DataUtils:

    def vocab_word2int(self, path_to_vocab_file):
        """
        Create a word2int dictionary from a vocab file
        e.g. print: {'who': 0}
        :param path_to_vocab_file:
        :return:
        """
        word2int = {}
        indx = 0
        with open(path_to_vocab_file) as f:
            for line in f:
                word2int[line.strip()] = indx
                indx += 1
        return word2int

    def vocab_int2word(self, path_to_vocab_file):
        """
        Transform word2int dictionary into an int2word dictionary
        e.g. print: {'0': word}
        :param path_to_vocab_file:
        :return:
        """
        word2int = self.vocab_word2int(path_to_vocab_file)
        int2word = {v: k for k, v in word2int.items()}
        return int2word

    def int2text(self, indices, int2word):
        """
        Transform a list of indices according to a lookup dictionary
        :param indices: List of indices representing words, according to a vocab file
        :param int2word: An int2word dictionary
        :return: list of words (corresponding to the indices)
        """
        result = []
        for element in indices:
            if element in int2word:
                result.append(int2word[element])
            else:
                result.append(str(element))
        return result

    def text2index(self, text_array, word2int):
        """
        use a word2int representation to turn an array of word sentences into an array of indices
        :param text_array: array of words
        :param word2int: a dictionary word2int
        :return: int representation of a sentence
        """
        text2index = []
        for sentence in text_array:
            indexes = []
            for word in sentence.split(' '):
                if word in word2int:
                    indexes.append(word2int.get(word))
                else:
                    indexes.append("1")  # <unk>
            text2index.append(indexes)
        return text2index

    def get_file_length(self, path_to_vocab_file):
        """
        Get file length of a vocab file
        -- !! Assuming each line contains ONE SINGLE UNIQUE WORD !! --
        :param path_to_vocab_file:
        :return: Amount of single unique words in a file
        """
        count = 0
        with open(path_to_vocab_file, 'r') as f:
            for line in f:
                count += 1
        return count

    def get_kp_text_max_lengths(self, dl_train, dl_val, dl_test):
        """
        Get the length of the longest keypoint file and the length of the longest sentence
        :param dl_train: data loader container train data
        :param dl_val: data loader container val data
        :param dl_test: data loader container test data
        :return: len of longets kp file, len of longest sentence, [max lenghts of datasets]
        """
        kp_max_total = 0
        text_max_total = 0
        kp_text_max_list = []  # save all the values because it takes time to compute and might be useful
        for loader in [dl_train, dl_val, dl_test]:
            max_kp, max_text = self.get_max_loader_length(loader)
            kp_text_max_list.append([max_kp, max_text])

            if max_kp > kp_max_total:
                kp_max_total = max_kp

            if max_text > text_max_total:
                text_max_total = max_text

        return kp_max_total, text_max_total, kp_text_max_list

    def get_max_loader_length(self, dl):
        """
        Get max length of keypoints and text of a folder for the whole dataset (one data_loader)
        :param dl: data laoder
        :return: max length of keypoints and text (dont need to be necessary the same folder)
        """
        kp_max = 0
        text_max = 0
        it = iter(dl)

        while 1:
            try:
                iterator_data = next(it)
            except StopIteration:  # if StopIteration is raised, all data of a loader is used
                break
            source_ten_len = torch.as_tensor(iterator_data[0], dtype=torch.float).view(-1, 1, 274).size()[0]
            target_ten_len = torch.as_tensor(iterator_data[1], dtype=torch.long).view(-1, 1).size()[0]

            if source_ten_len > kp_max:
                kp_max = source_ten_len

            if target_ten_len > text_max:
                text_max = target_ten_len
        return kp_max, text_max
