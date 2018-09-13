import numpy as np
import os
import time
from Config import Config

# 预处理
from utils.local_utils import load_data
from utils.word_seq import WordSequence
from utils.data_utils import batch_flow_seq2seq

# torch相关
import torch
import torch.nn as NN
from layers.EmbeddingLayer import EmbeddingLayer
from layers.RNNLayer import RNNLayer
from layers.encoder import Encoder
from layers.decoder import LuongDecoder
from torch import optim
from Loss import pad_cross_entropy
from torch.nn.utils import clip_grad_norm

# config
conf = Config()


def test_preprocess(ws_x = None, ws_y = None):

    # 读取数据
    print('start reading testing data...')
    input_x = conf.srcSeq_test
    input_y = conf.tarSeq_test
    x_test, y_test = load_data(input_x, input_y)

    # 构建 input_x 和 input_y 的 WordSequence 类
    if ws_x is None and ws_y is None:
        np.random.seed(0)
        ws_x = WordSequence()
        ws_y = WordSequence()

        ws_x.fit(
            sentence_set=x_test,
            max_features=conf.srcVocSize,
            min_count=5
        )
        ws_y.fit(
            sentence_set=y_train,
            max_features=conf.tarVocSize,
            min_count=5
        )

    print('input_voc_size: ', len(ws_x))
    print('target_voc_size: ', len(ws_y))
    print('input_date and target_data have been readed.')

    return x_test, y_test, ws_x, ws_y

def batch_sort(x, xl, y=None, yl=None):
    """
    RNNencoder端，需要输入按照长度从大到小排列的batch数据
    因此在每个batch内进行排序
    """
    sorted_xl, id = torch.sort(xl, 0, True)
    sorted_x = x[id]
    if y is None and yl is None:
        return sorted_x, sorted_xl
    else:
        sorted_y = y[id]
        sorted_yl = yl[id]
        return sorted_x, sorted_xl, sorted_y, sorted_yl 

def output():

    conf = Config()

    # checkpoint
    # - 'Encoder': encoder 参数
    # - 'Decoder': decoder 参数
    # - 'Wordseq_src': ws_x 训练语料的 ws_x
    # - 'Wordseq_tar': ws_y 训练预料的 ws_y
    model = torch.load(conf.save_path + conf.save_version + 'checkpoint.tar')
    ws_x = model['Wordseq_src']
    ws_y = model['Wordseq_tar']

    x, y, ws_x, ws_y = test_preprocess(ws_x, ws_y)

    encoder = Encoder(
        hiddenDim = conf.encHiddenDim,
        encVocSize = len(ws_x),
        encEmbDim = conf.srcEmbDim,
        PADID = conf.PadId,
        layerNum = conf.encLayer,
        isBidirectional = conf.enc_isBid,
        dropout = conf.dropout
    )

    decoder = LuongDecoder(
        encHiddenDim = conf.encHiddenDim,
        decHiddenDim = conf.decHiddenDim,
        decVocSize = len(ws_y),
        decEmbDim = conf.srcEmbDim,
        PADID = conf.PadId,
        layerNum = conf.decLayer,
        enc_isBid= conf.enc_isBid,
        # attentionMethod = 'general' if conf.enc_isBid else 'dot',
        attentionMethod = 'multi',
        dropout = conf.dropout
    )

    flow = batch_flow_seq2seq([x, y], [ws_x, ws_y], batch_size=conf.output_batchSize)

    encoder.load_state_dict(model['Encoder'])
    decoder.load_state_dict(model['Decoder'])

    encoder.eval()
    decoder.eval()


    print('model has been loaded.')

    x, xl, y, yl = next(flow)

    x = torch.LongTensor(x)
    xl = torch.LongTensor(xl)
    y = torch.LongTensor(y)
    yl = torch.LongTensor(yl)

    x, xl, y, yl = batch_sort(x, xl, y, yl)

    enc_output, enc_hidden = encoder(x, xl)
    max_tar_len = conf.decode_max_len # 最多解码dec_max_len个字符，遇到结束字符则结束
    dec_lasthidden = torch.zeros(decoder.layerNum, conf.output_batchSize, conf.decHiddenDim)
    all_decoder_outputs = torch.zeros(conf.output_batchSize, max_tar_len)

    dec_y = torch.LongTensor([WordSequence.START] * conf.output_batchSize)
    
    # [max_tar_len, batchSize, sequenceLen - att_Window_Size + 1]
    att_set = []

    for t in range(max_tar_len):
        dec_output, dec_lasthidden, dec_att = decoder(dec_y, dec_lasthidden, enc_output)
        att_set.append(dec_att)
        # 在这里要根据概率分布，确定在当前时间步的输出   抽样、贪心、beam
        prob, idx = torch.max(dec_output, 1)
        dec_y = idx
        all_decoder_outputs[:, t] = idx
    
    # att_set_tenor = torch.cat(att_set)
    
    for sen in range(conf.output_batchSize):
        print('decode_sen ', sen, ': ', ws_y.inverse_transform(all_decoder_outputs[sen]))
        print('origin_sen ', sen, ': ', ws_y.inverse_transform(y[sen]))
        print('                             ')

    # print('att_set_tensor: ', att_set_tenor)
    # print('att_set_tensor_size', att_set_tenor.size())

    print('end.')

if __name__ == '__main__':
    output()