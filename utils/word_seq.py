import numpy as np

class WordSequence(object):
    """
    生成了一个字典，并给出相应的方法
    to_index: word to index
    to_word: index to word
    size: 返回字典大小，并且使用len()方法重载
    fit: 根据语料生成字典
    transform: 将一个句子转化为index形式
    inverse_transform: 将index反向转换为句子形式
    """
    PAD_TAG = '<pad>'
    UNK_TAG = '<unk>'
    START_TAG = '<s>'
    END_TAG = '</s>'
    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }
        self.fited = False
        # print('init have done !')

    def to_index(self, word):
        assert self.fited, "WordSequence 尚未 fit"
        if word in self.dict:
            return self.dict[word]
        return WordSequence.UNK

    def to_word(self, index):
        assert self.fited, "WordSequence 尚未 fit"
        for key, value in self.dict.items():
            if value == index:
                return key
        return WordSequence.UNK_TAG

    def size(self):
        assert self.fited, "WordSequence 尚未 fit"
        return len(self.dict)

    def __len__(self):
        return self.size()

    def fit(self, sentence_set, min_count=5, max_count=None, max_features=None):
        """
        :param sentence_set: 语料集合
        :param min_count: 语料集合当中出现次数小于min_count的词将不会收录在字典当中
        :param max_count: 语料集合当中出现次数大于max_count的词将不会收录在字典当中
        :param max_features: 字典的最大大小，字典将按照出现次数降序排列，取前max_features个key-value对
        """
        assert not self.fited, "WordSequence 只能 fit 一次"

        count = {}
        for sentence in sentence_set:
            arr = list(sentence)
            for word in arr:
                if word not in count:
                    count[word] = 0
                count[word] += 1

        if min_count is not None:
            count = {key: value for key, value in count.items() if value >= min_count}

        if max_count is not None:
            count = {key: value for key, value in count.items() if value <= max_count}

        self.dict = {
            WordSequence.PAD_TAG: WordSequence.PAD,
            WordSequence.UNK_TAG: WordSequence.UNK,
            WordSequence.START_TAG: WordSequence.START,
            WordSequence.END_TAG: WordSequence.END,
        }

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for word, _ in count:
                self.dict[word] = len(self.dict)
        else:
            for word in sorted(count.keys()):
                self.dict[word] = len(self.dict)

        self.fited = True

    def transform(self, sentence, max_len=None):
        """
        :param sentence:
        :param max_len: 设定max_len之后将自动padding到max_len
        :return:
        """
        assert self.fited, "WordSequence 尚未 fit"

        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        for index, word in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(word)

        return np.array(r)

    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):

        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret

def test():
    ws = WordSequence()
    sten_set = [['第', '一', '句', '话'],
                ['第', '2', '句', 'hua'],
                ['第', '三', '个', '词']]

    ws.fit(sentence_set=sten_set, min_count=1)

    indice = ws.transform(['第', '2'], max_len=6)
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)

    print(ws.dict)


if __name__ == '__main__':
    test()