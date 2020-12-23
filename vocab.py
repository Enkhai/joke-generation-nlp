PAD_token = 0
SOS_token = 1
EOS_token = 2


class Vocabulary:

    def __init__(self):
        self.word2index = {'PAD': PAD_token, 'SOS': SOS_token, 'EOS': EOS_token}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.word2count = {}
        self.num_words = 3
        self.trimmed = False

    def add_sentence(self):
        pass

    def add_word(self):
        pass

    def trim(self):
        pass
