# @file: RNNLayer.py
# @time: 2018/8/30
# @author: wr
import torch
import torch.nn as NN


class RNNLayer(NN.Module):
    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            NN.init.xavier_normal(t)
        for t in hh:
            NN.init.xavier_normal(t)
        for t in b:
            NN.init.constant(t, 0)

    def __init__(self, inputDim, hiddenDim, layerNum, dropout, isBidirectional, cellType = "LSTM"):
        '''
        RNN封装
        :param inputDim:         int, 输入的维度，e.g. 接在embedding layer之后则为embedding 的维度
        :param hiddenDim:        int，RNN单元hidden state的维度
        :param layerNum:         int, RNN重复层数
        :param dropout:          float, dropout rate，只有layerNum>1的时候，drop out才有意义， keepProb = 1 - dropout
        :param isBidirectional:  bool 是否双向
        :param cellType:         cell 的type， opt[RNN, LSTM, GRU]
        '''
        super(RNNLayer, self).__init__()
        # copy
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.layerNum = layerNum
        self.dropout = dropout
        self.isBidirectional = isBidirectional
        self.cellType = cellType

        if self.cellType == "LSTM":
            self.rnn = NN.LSTM(input_size=self.inputDim,
                               hidden_size=self.hiddenDim,
                               num_layers=self.layerNum,
                               batch_first=True,
                               dropout=self.dropout,
                               bidirectional=self.isBidirectional)
        elif self.cellType == "GRU":
            self.rnn = NN.GRU(input_size=self.inputDim,
                               hidden_size=self.hiddenDim,
                               num_layers=self.layerNum,
                               batch_first= True,
                               dropout=self.dropout,
                               bidirectional=self.isBidirectional)
        else:
            self.rnn = NN.RNN(input_size=self.inputDim,
                               hidden_size=self.hiddenDim,
                               num_layers=self.layerNum,
                               batch_first=True,
                               dropout=self.dropout,
                               bidirectional=self.isBidirectional)
        self.init_weights()

    def forward(self, x, lastHidden = None, lastC = None):
        '''
        :param x: input data, shape is [batchSize, sequenceLen, inputDim]
        :intro: 包含两种前向迭代方式：
            ：区别在于RNN迭代时候，是否有初始初始状态的输入。一般来说，在encoder端是没有初始输入的，在decoder端是有初始输入的
            
            ：仅输入序列 x, [batchSize, squenceLen, inputDim]
                y, hn = RNN(x)
                输出y, [batchSize, squenceLen, hiddenDim * (isBid ? 1:2)]
                输出hn, [layerNum * (isBid ? 1:2), batchSize, hiddenDim]
            
            :输入序列 x, [batchsize, squenceLen, inputDim], 输入上一个时间步的状态lastHidden, [layerNum * (isBid ? 1:2), 1, hiddenDim]
                y, ht = RNN(x, lasrHidden)
                输出y, [batchSize, squenceLen, hiddenDim * (isBid ? 1:2)]
                输出ht, [layerNum * (isBid ? 1:2), batchSize, hiddenDim]

            :y是RNNLayer的最后一层的隐状态,
            :注意输出的hn和ht的形状

        '''
        if lastHidden is None:
            if self.cellType == "LSTM":
                y, hn, cn = self.rnn(x)
            else:
                y, hn = self.rnn(x)
            return y, hn
        else:
            if self.cellType == 'LSTM':
                y, ht, ct = self.rnn(x, (lastHidden, lastC))
            else:
                y, ht = self.rnn(x, lastHidden)
            return y, ht

def test():
    sen_batch = torch.randn(3, 4, 2) # [batchSize = 3, squenceLen = 4, embDim = 2]
    lastHidden = torch.randn(1, 3, 5) # [layerNum * (isBid ? 1:2) = 1, batchSize =  3, hiddenDim = 5]
    
    
    rnn1 = RNNLayer(
        inputDim = 2,
        hiddenDim = 5,
        layerNum = 1,
        dropout = 0,
        isBidirectional = False,
        cellType = 'RNN'
    )

    rnn2 = RNNLayer(
        inputDim = 2,
        hiddenDim = 5,
        layerNum = 2,
        dropout = 0,
        isBidirectional = False,
        cellType = 'RNN'
    )

    rnn3 = RNNLayer(
        inputDim = 2,
        hiddenDim = 5,
        layerNum = 2,
        dropout = 0,
        isBidirectional = True,
        cellType = 'RNN'
    )

    print('------rnn1------')
    print('sen_batch: ', sen_batch.size())
    print('layerNum: ', 1)
    print('isBid: ', False)
    print('y, h0 = rnn1(sen_batch)')

    y1, h1 = rnn1(sen_batch)
    print('y1: ',y1.size() )
    print('h1: ', h1.size())



    print('------rnn2------')
    print('sen_batch: ', sen_batch.size())
    print('layerNum: ', 2)
    print('isBid: ', False)
    print('y2, h2 = rnn2(sen_batch)')

    y2, h2 = rnn2(sen_batch)
    print('y2: ', y2.size())
    print('h2: ', h2.size())



    print('------rnn3------')
    print('sen_batch: ', sen_batch.size())
    print('layerNum: ', 2)
    print('isBid: ', True)
    print('y3, h3 = rnn3(sen_batch)')

    y3, h3 = rnn3(sen_batch)
    print('y3: ', y3.size())
    print('h3: ', h3.size())


if __name__ == '__main__':
    test()