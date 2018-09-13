# @file: Decoder.py
# @time: 2018/9/12
# @author: wr
import torch
import torch.nn as NN
from layers.RNNLayer import RNNLayer
from layers.EmbeddingLayer import EmbeddingLayer
from layers.AttentionLayer import AttentionLayer
from layers.MultiSizeAttention import MultiSizeAttention
import torch.functional as F
import torch.nn.functional as NNF

class LuongDecoder(NN.Module):
    def __init__(
        self,
        encHiddenDim,
        decHiddenDim,
        decVocSize,
        decEmbDim,
        layerNum = 1,
        dropout = 0.1,
        fineTune = True,
        PADID = 0,
        enc_isBid = False,
        attentionMethod = 'general'):
        """
        :param: attentionMethod: [multi, dot, general]
            - multi: 多对一的attention
            - dot: 点积
            - general: ~~
        """

        super(LuongDecoder, self).__init__()

        self.encHiddenDim = encHiddenDim
        self.decHiddenDim = decHiddenDim
        self.decVocSize = decVocSize
        self.decEmbDim = decEmbDim
        self.layerNum = layerNum
        self.PADID = PADID
        self.attentionMethod = attentionMethod
        self.fineTune = fineTune
        self.enc_isBid = enc_isBid

        # layer
        self.embedding = EmbeddingLayer(embeddingSize=(self.decVocSize, self.decEmbDim),
                                        dropout = dropout,
                                        fineTune=self.fineTune,
                                        paddingIdx=self.PADID)
        self.dropout = NN.Dropout(dropout)

        if self.attentionMethod == 'multi':
            self.attention = MultiSizeAttention(encHiddenDim = self.encHiddenDim, decHiddenDim = self.decHiddenDim)
        else:
            self.attention = AttentionLayer(method = self.attentionMethod, encHiddenDim = self.encHiddenDim, decHiddenDim = self.decHiddenDim)
        
        self.rnn = RNNLayer(inputDim = self.decEmbDim, hiddenDim = self.decHiddenDim, layerNum = self.layerNum,
                            dropout = dropout, isBidirectional = False, cellType="GRU")
        
        if self.attentionMethod == 'multi':
            self.out = NN.Linear(self.decHiddenDim + 2 * 3 *self.encHiddenDim, self.decVocSize) # 这里将output和context连接在一起
        elif self.attentionMethod == 'dot':
            self.out = NN.Linear(self.decHiddenDim + self.encHiddenDim, self.decVocSize)
        elif self.attentionMethod == 'general':
            self.out = NN.Linear(self.decHiddenDim + 2 * self.encHiddenDim, self.decVocSize)
        
        NN.init.xavier_normal(self.out.weight)
        print('__init__ dec')

    def forward(self, y, lastHidden, encoderOutput):
        """
        在这里定义的是单步的decoder
        hidden_t = RNN(y_t-1, hidden_t-1, attention_t(encoderOutput))
        hidden_t --> out_t
        :param y: [batchSize]
        :param lastHidden: [layerNum, batchSize, decHiddenDim] 上一个时间步的隐状态
        :param encoderOutput: [batchSize, sequenceLen, enc_hiddenDim] encoder端各个时间步的最后一层隐状态层
        :Bid  enc_hiddenDim = 2 * encHiddenDim
        :return: 
            - oup [batchSize, decVocSize]
            - hidden [layerNum, batchSize, decHiddenDim]
            - attenWeight：
                - dot & general [batchSize, sequenceLen]
                - multi [batchSize, sequenceLen - att_Window_Size + 1]
        """
        # y加一维, [batchSize] --> [batchSize, 1]
        y = y.unsqueeze(1)
        
        # 得到yEmb, [batchSize, 1, embDim]
        yEmb = self.embedding(y)
        yEmb = self.dropout(yEmb)

        # 一步解码
        # output: [batchSize, 1, dechiddenDim]
        # hidden: [layNum, batchSize, dechiddenDim]
        output, hidden = self.rnn(yEmb, lastHidden)
        
        # 计算attention
        #   - hidden[-1]: [batchSize, hiddenDim]
        #   - encoderOutput: [batchSize, sequenceLen, hiddenDim]
        #   --> attentionWeight: [batchSize, SequenceLen]
        if self.attentionMethod == 'multi':
            attentionWeight, new_encoderOutput = self.attention(hidden[-1], encoderOutput)
            # attentionWeight: [batchSize, sequenceLen] --> [batchSize, 1, sequenceLen]
            attentionWeight = attentionWeight.unsqueeze(1)

            # torch.bmm batched matrix-matrix product
            # batch1 [b , n, m], batch2 [b , m, p]  out [b , n, p]  out = torch.bmm(batch1, batch2)
            context = torch.bmm(attentionWeight, new_encoderOutput)
            # context: [batchSize, 1, sequenceLen] bmm [batchSize, sequenceLen, hiddenDim] --> [batchSize, 1, hiddenDim]
        else:
            attentionWeight = self.attention(hidden[-1], encoderOutput)
            attentionWeight = attentionWeight.unsqueeze(1)
            context = torch.bmm(attentionWeight, encoderOutput)

        output = output.squeeze(1)
        context = context.squeeze(1)

        # inp: [batchSize, output_dim + context_dim]
        inp = torch.cat((output, context),1) 
        oup = self.out(inp)
        oup = NNF.log_softmax(oup)
        return oup, hidden, attentionWeight


# test
def test():

    print('----decoder 1----')
    dec = LuongDecoder(
        encHiddenDim = 5,
        decHiddenDim = 5,
        decVocSize = 7,
        decEmbDim = 2,
        attentionMethod='general'
    )
    y = torch.tensor([4, 4, 4])
    lasthidden = torch.randn(1, 3, 5)
    encoderOutput = torch.randn(3, 4, 10)
    oup, hidden, attentionWeight = dec(y, lasthidden, encoderOutput)
    print('attentionMethod: general')
    print('oup: ', oup)
    print('hidden: ', hidden)
    print('aw: ', attentionWeight)
    print('                     ')

    print('----decoder 2----')
    dec = LuongDecoder(
        encHiddenDim = 5,
        decHiddenDim = 5,
        decVocSize = 7,
        decEmbDim = 2,
        attentionMethod='dot'
    )
    y = torch.tensor([4, 4, 4])
    lasthidden = torch.randn(1, 3, 5)
    encoderOutput = torch.randn(3, 4, 5)
    oup, hidden, attentionWeight = dec(y, lasthidden, encoderOutput)
    print('attentionMethod: dot')
    print('oup: ', oup)
    print('hidden: ', hidden)
    print('aw: ', attentionWeight)
    print('                     ')

    print('----decoder 3----')
    dec = LuongDecoder(
        encHiddenDim = 5,
        decHiddenDim = 5,
        decVocSize = 7,
        decEmbDim = 2,
        attentionMethod='multi'
    )
    y = torch.tensor([4, 4, 4])
    lasthidden = torch.randn(1, 3, 5)
    encoderOutput = torch.randn(3, 4, 10)
    oup, hidden, attentionWeight = dec(y, lasthidden, encoderOutput)
    print('attentionMethod: multi')
    print('oup: ', oup)
    print('hidden: ', hidden)
    print('aw: ', attentionWeight)
    print('                     ')


if __name__ == '__main__':
    test()