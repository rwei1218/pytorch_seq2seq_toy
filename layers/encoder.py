# @file: Encoder.py
# @time: 2018/8/30
# @author: wr
import torch
import torch.nn as NN
from layers.RNNLayer import RNNLayer
from layers.EmbeddingLayer import EmbeddingLayer
import torch.functional as F
import torch.nn.functional as NNF


class Encoder(NN.Module):
    """
    encoder：
    将输入的原始文本进行编码：
        - layer1: embedding layer:
            - input: [batchSize, squenceLen]
            - output: [batchSize, squenceLen, embDim]
        - layer2: rnn encoder layer:
            - input: [batchSize, squenceLen, embDim]
            - output: 
                - enc_output: [batchSize, squenceLen, hiddenDim * (isBid ? 1:2)]
                - hidden: [layerNum * (isBid ? 1:2), batchSize, hiddenDim]
        : 其中<pad>的时间步位置都是0.0000
        ：encoder的输入数据需要是进行过排序的数据，按照序列长度降序排列
    """
    def __init__(
        self,
        hiddenDim,
        encVocSize,
        encEmbDim,
        isBidirectional = False,
        layerNum = 1,
        dropout = 0.1,
        fineTune = True,
        PADID = 0,
        cellType = 'GRU'):

        super(Encoder, self).__init__()

        self.hiddenDim = hiddenDim
        self.encVocSize = encVocSize
        self.encEmbDim = encEmbDim
        self.layerNum = layerNum
        self.dropout = dropout
        self.PADID = PADID
        self.cellType = cellType
        self.fineTune = fineTune
        self.isBidirectional = isBidirectional

        # layer
        self.embedding = EmbeddingLayer(
            embeddingSize = (self.encVocSize, self.encEmbDim),
            dropout = self.dropout,
            fineTune = self.fineTune,
            paddingIdx = self.PADID
        )

        self.rnn = RNNLayer(
            inputDim = self.encEmbDim,
            hiddenDim = self.hiddenDim,
            layerNum = self.layerNum,
            dropout = self.dropout,
            isBidirectional = self.isBidirectional,
            cellType =  self.cellType
        )

    def forward(self, x, x_length):
        """
        :param: x [batchSize, sequenceLen]
        :param: x_length [batchSize]

        :return
            - enc_output: [batchSize, squenceLen, hiddenDim * (isBid ? 1:2)]
            - hidden: [layerNum * (isBid ? 1:2), batchSize, hiddenDim]
        """
        emb = self.embedding(x)
        x_length = x_length.tolist()
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, x_length, batch_first=True)
        output, hidden = self.rnn(packed)
        enc_output, _ =  torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return enc_output, hidden


# test
def batch_sort(x, xl, y=None, yl=None):
    sorted_xl, id = torch.sort(xl, 0, True)
    sorted_x = x[id]
    if y is None and yl is None:
        return sorted_x, sorted_xl
    else:
        sorted_y = y[id]
        sorted_yl = yl[id]
        return sorted_x, sorted_xl, sorted_y, sorted_yl 

def test():
    ss = [[1, 3, 4, 5, 6, 0, 0, 0, 0, 0],
    [6, 4, 4, 4, 4, 3, 2, 2, 2, 1],
    [5, 4, 2, 1, 1, 2, 3, 1, 0, 0]]

    ss = torch.tensor(ss)
    ss_len = torch.tensor([5, 10, 8])
    # print(ss.size())

    sorted_ss, sorted_ss_len = batch_sort(ss, ss_len)

    enc = Encoder(
        hiddenDim = 5,
        encVocSize = 7,
        encEmbDim = 3,
        isBidirectional = False,
        layerNum=3,
        cellType = 'RNN',
        PADID = 0
    )

    # print(enc.parameters())

    o, h = enc(sorted_ss, sorted_ss_len)
    print('output: ', o)
    print('hidden: ', h)

if __name__ == '__main__':
    test()