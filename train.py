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


def train_preprocess(ws_x = None, ws_y = None):
    """
    train阶段预处理：
        - 读取数据
        - 创建input seq和target seq的WordSquence类
        - 返回x_train, y_train, ws_x, ws_y
    """
    # 读取数据
    print('start reading training data...')
    input_x = conf.srcSeq_train
    input_y = conf.tarSeq_train
    x_train, y_train = load_data(input_x, input_y)

    # 构建 input_x 和 input_y 的 WordSequence 类
    if ws_x is None and ws_y is None:
        np.random.seed(0)
        ws_x = WordSequence()
        ws_y = WordSequence()

        ws_x.fit(
            sentence_set=x_train,
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

    return x_train, y_train, ws_x, ws_y


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


def main():
    # preprocess
    x, y, ws_x, ws_y = train_preprocess()

    # model
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

    encoder.train()
    decoder.train()

    print('attention method: ', decoder.attentionMethod)

    if conf.USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # optimizer
    enc_optimizer = optim.Adam(encoder.parameters())
    dec_optimizer = optim.Adam(decoder.parameters())

    # data flow
    flow = batch_flow_seq2seq([x, y], [ws_x, ws_y], batch_size=conf.train_batchSize)

    epoch = 1
    print('epoch: ', epoch)
    
    loss_recoder = {}

    for step in range(1, conf.max_steps+1):

        # 梯度置0，此操作很重要，尤其是在复杂的模型当中
        # 复杂有可能不收敛，或者收敛到完全的<unk>
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # 一个step是对一个batch进行训练
        x, xl, y, yl = next(flow)

        x = torch.LongTensor(x)
        xl = torch.LongTensor(xl)
        y = torch.LongTensor(y)
        yl = torch.LongTensor(yl)

        if conf.USE_CUDA:
            x = x.cuda()
            xl = xl.cuda()
            y = y.cuda()
            yl =yl.cuda()


        x, xl, y, yl = batch_sort(x, xl ,y, yl)
        # print(x)
        # print(y)

        # 输入的需要被排序，按照长度从大到小排序

        enc_output, enc_hidden = encoder(x, xl)
        max_tar_len = max(yl.tolist())
        # 第一个时间步的解码的dec_lasthidden应该是相同的, 因为所有句子都是从<s>开始的额
        dec_lasthidden = torch.zeros(decoder.layerNum, conf.train_batchSize, conf.decHiddenDim)
        all_decoder_outputs = torch.zeros(conf.train_batchSize, max_tar_len, decoder.decVocSize)

        dec_y = torch.LongTensor([WordSequence.START] * conf.train_batchSize)

        if conf.USE_CUDA:
            enc_output = enc_output.cuda()
            enc_hidden = enc_hidden.cuda()
            dec_lasthidden = dec_lasthidden.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()
            dec_y = dec_y.cuda()
        
        for t in range(max_tar_len):
            # dec_y, dec_lasthidden是在不断变更的，体现了解码的时序性
            dec_output, dec_lasthidden, dec_att = decoder(dec_y, dec_lasthidden, enc_output)
            dec_y = y[:, t]
            all_decoder_outputs[:, t, :] = dec_output

        loss = pad_cross_entropy(
            all_decoder_outputs,
            y,
            0
        )
        loss.backward()

        if step % conf.display_step == 0 or step == 1:
            print('step: ', step, 'batch_loss: ', loss)
            loss_recoder[step] = loss.item()

        ec = clip_grad_norm(encoder.parameters(), 5.0)
        dc = clip_grad_norm(decoder.parameters(), 5.0)

        enc_optimizer.step()
        dec_optimizer.step()

        # 每一个epoch保存一个checkpoint: save_path + save_version + str(epoch) + 'checkpoint.tar'
        if step % (int(conf.data_size/conf.train_batchSize)) == 0:

            loss_recoder_file = open('./lossrecoder/' + conf.save_version + str(epoch) + '.txt', 'w', encoding = 'utf8')
            for key, value in loss_recoder.items():
                loss_recoder_file.writelines(str(key)+ ' ' + str(value) + '\n')
            loss_recoder_file.close()

            loss_recoder = {}

            torch.save({'Encoder': encoder.state_dict(),
                        'Decoder': decoder.state_dict(),
                        'Wordseq_src': ws_x,
                        'Wordseq_tar': ws_y},
                        conf.save_path + conf.save_version + str(epoch) + 'checkpoint.tar')
            epoch += 1
            if epoch > conf.max_epochs:
                break
            print('epoch: ', epoch)
            
    print('training is ended !')
    
    # 训练结束之后，再次进行一次保存，这次保存和最后一个checkpoint的参数是相同的，仅仅是为了output方便，output函数默认读取最后一次的参数
    # save_path + save_version + 'checkpoint.tar'
    torch.save({'Encoder': encoder.state_dict(),
                'Decoder': decoder.state_dict(),
                'Wordseq_src': ws_x,
                'Wordseq_tar': ws_y},
                conf.save_path + conf.save_version + 'checkpoint.tar')

    print('model has been saved !')
    

if __name__ == '__main__':
    main()