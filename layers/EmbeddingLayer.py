# @file: AttentionLayer.py
# @time: 2018/9/12
# @author: wr
import torch
import torch.nn as NN


class EmbeddingLayer(NN.Module):

    def __init__(self, embeddingSize = None, embeddingMatrix = None,
                 fineTune = True, dropout = 0.5, paddingIdx = 0,
                 maxNorm = None, normType = 2,  scaleGradByFreq = False, sparse = False):
        '''
        size和matrix必须指定一个
        :param embeddingSize:     matrix 的tuple， e.g. 20000 * 3000
        :param embeddingMatrix:   tensor，预先训练的embedding matrix
        :param fineTune:          bool， 是否fine tune
        :param dropout:           float， drop rate
        :param paddingIdx:        int， 用于padding的id, 在我们word_seq模型当中，使用0做padding符号，则在这里面 0 对应的embedding为[0.0, 0.0, 0.0, ...]
        :param maxNorm:           float，若非None，则renormalize embeddings
        :param normType:          float，计算normalize的时候的p阶
        :param scaleGradByFreq:   bool， 若非None， 则用这个词在mini-batch中的frequency scale 梯度
        :param sparse:            bool
        '''
        super(EmbeddingLayer, self).__init__()
        if embeddingMatrix is not  None:
            embeddingSize = embeddingMatrix.size()
        elif embeddingSize is not None:
            embeddingMatrix = torch.Tensor(embeddingSize[0], embeddingSize[1])
            NN.init.xavier_normal(embeddingMatrix)
        assert (embeddingSize is not None)
        assert (embeddingMatrix is not None)
        self.matrix = NN.Embedding(num_embeddings=embeddingSize[0],
                                   embedding_dim=embeddingSize[1],
                                   padding_idx=paddingIdx,
                                   max_norm=maxNorm,
                                   norm_type=normType,
                                   scale_grad_by_freq=scaleGradByFreq,
                                   sparse=sparse)
        # self.matrix.weight.data.copy_(embeddingMatrix)
        # 为什么要给embedding加一个初始化？而且还是自己指定的初始化
        print('Embedding_requires_grad', fineTune)
        self.matrix.weight.requires_grad = fineTune
        self.dropout = NN.Dropout(p=dropout)

    def forward(self, x):
        '''
        Forward this module
        :param x: token senquence, [batchSize, sentenceLength]
        :return: [batchSize, sentenceLength, embeddingDim]
        '''
        return self.matrix(x)

if __name__ == '__main__':
    """
    """

    ebd = EmbeddingLayer(embeddingSize=[10, 5], dropout=0, fineTune=False, paddingIdx = 0)
    x = torch.LongTensor([[9, 3, 2, 1, 0, 0, 0],[4, 2, 5, 0, 0, 0, 0]])
    y = ebd(x)

    y_pack = torch.nn.utils.rnn.pack_padded_sequence(y, [4, 3], batch_first = True)
    
    print(x)
    print(y)
    print(y_pack)