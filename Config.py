# @file: Config.py
# @time: 2018/9/12
# @author: wr

class Config():
    def __init__(self):
        """
        需要经常修改的参数：
        self.srcSeq_train, self.tarSeq_train,
        self.srcSeq_test, self.tarSeq_test

        self.decode_max_len decode端的最大解码次数、请根据数据集实际情况确定
        self.data_size 训练数据集大小，根据实际情况确定
        self.max_epochs 数据最大训练轮数

        """

        self.USE_CUDA = True

        # Data path
        self.srcSeq_train = './data/src_seq.txt'
        self.tarSeq_train = './data/tar_seq.txt'
        self.srcSeq_test = './data/src_seq.txt'
        self.tarSeq_test = './data/tar_seq.txt'

        # Save model
        self.save_path = './checkpoint/'
        self.save_version = '201809131342'

        # Word sequence
        self.srcVocSize = 30000
        self.tarVocSize = 30000
        self.PadId = 0

        # RNN
        self.srcEmbDim = 256
        self.tarEmbDim = 256
        self.encHiddenDim = 256
        self.decHiddenDim = 256
        self.encLayer = 2
        self.decLayer = 2
        self.decode_max_len = 11
        self.enc_isBid = True

        # Training
        self.data_size = 10000
        self.lr = 0.001
        self.max_steps = 10000000
        self.max_epochs = 2
        self.display_step = 10
        self.train_batchSize = 32
        self.fineTuneEmb = True
        self.dropout = 0.5
        self.clip = 5.0

        # Output
        self.output_batchSize = 20
        self.beamSize = 5
        