# @file: AttentionLayer.py
# @time: 2018/9/12
# @author: wr
import torch
import torch.nn as NN
import torch.nn.functional as F


class AttentionLayer(NN.Module):
    def __init__(self, method, encHiddenDim, decHiddenDim):
        '''
        :param method:  how to calculate score
            :有dot, general, concat三种, 目前实现了dot，general两种
            :默认encHiddenDim == decHiddenDim
            :当enc_Bid == True时使用general方法，当enc_Bid == False时使用dot方法
            :其中general引入了可变参数
        :param encHiddenDim
        :param decHiddenDim
        '''
        super(AttentionLayer, self).__init__()
        self.method = method
        self.encHiddenDim = encHiddenDim
        self.decHiddenDim = decHiddenDim

        # 引入了可变参数
        # 当 enc_Bid == True 的时候，使用 general
        if self.method == 'general':
            # 由于 enc_Bid == True，因此这里是2 * self.decHiddenDim
            self.attention = NN.Linear(2 * self.encHiddenDim, self.decHiddenDim)
            NN.init.xavier_normal(self.attention.weight)

    def forward(self, hidden, encoderOutput):
        '''
        :param hidden: (deecode端的hidden state)  [batchSize, decHiddenDim]
        :param encoderOutput: (encoder output) [batchSize, sequenceLength, encHiddenDim * (isBid ? 1:2)]
        :return: attention, [batchSize, sequenceLen]
        '''
        sequencelength = encoderOutput.size(1)
        batchSize = encoderOutput.size(0)
        atten = torch.zeros(batchSize, sequencelength)

        if isinstance(hidden.data, torch.cuda.FloatTensor):
            atten = atten.cuda()

        # 对应 enc_isBid == False
        if self.method == 'dot':
            for b in range(batchSize):
                # = [seqlen , encHiddenDim] * [decHiddenDim], dot要求 encHiddenDim == decHiddenDim
                atten[b] = torch.mv(encoderOutput[b], hidden[b])
        # 对应 enc_isBid == True
        elif self.method == 'general':
            for b in range(batchSize):
                # energy = [seqlen , 2 * encHiddenDim] * [2 * encHiddenDim, decHiddenDim] = [seqlen , decHiddenDim]
                energy = self.attention(encoderOutput[b])
                atten[b] = torch.mv(energy, hidden[b])
        else:
            print('未指定attention方法')
        
        return F.softmax(atten)


if __name__ == '__main__':
    attention = AttentionLayer('general', encHiddenDim = 5, decHiddenDim = 5)
    h = torch.randn(3, 5)
    enc_out = torch.randn(3, 4, 10)
    y = attention(h, enc_out)
    print(y)