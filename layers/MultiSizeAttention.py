# @file: MultiSizeAttention.py
# @time: 2018/9/11
# @author: wr
import torch
import torch.nn as NN
import torch.nn.functional as F


class MultiSizeAttention(NN.Module):
    def __init__(self, encHiddenDim, decHiddenDim, atten_Window_Size = 3):

        super(MultiSizeAttention, self).__init__()
        self.encHiddenDim = encHiddenDim
        self.decHiddenDim = decHiddenDim
        self.atten_Window_Size = atten_Window_Size

        # Bidir * atten_Window_Size = 3
        self.attention = NN.Linear(2 * self.atten_Window_Size * self.encHiddenDim, self.decHiddenDim)
        NN.init.xavier_normal(self.attention.weight)

    def forward(self, hidden, encoderOutput):
        """
        :param hidden: (decoder端的hidden state)  [batchSize, decHiddenDim]
        :param encoderOutput: (encoder output) [batchSize, sequenceLength, enc_hiddenDim]
        :Bidir: enc_hiddenDim = 2 * encHiddenDim

        :return: atten [batchSize, sequenceLength - atten_Window_Size + 1]
                 new_encoder_output [batchSize, sequenceLength - atten_Window_Size + 1, hiddenDim * atten_Window_Size]
        """
        sequencelength = encoderOutput.size(1)
        batchSize = encoderOutput.size(0)

        atten = torch.zeros(batchSize, sequencelength - self.atten_Window_Size + 1)
        new_encoder_output = self.multiEncoderOutput(hidden ,encoderOutput)

        if isinstance(hidden.data, torch.cuda.FloatTensor):
            atten = atten.cuda()
        
        for b in range(batchSize):
            energy = self.attention(new_encoder_output[b])
            atten[b] = torch.mv(energy, hidden[b])
        
        return F.softmax(atten), new_encoder_output
            

    def multiEncoderOutput(self, hidden, encoderOutput, atten_Window_Size = 3):
        """
        :atten_window_size: 窗口大小 2,3,4
        :param hidden: (decoder端的hidden state)  [batchSize, decHiddenDim]
        :encoderOutput: 原始的encoderOutput [batchSize, sequenceLength, enc_hiddenDim]
        :Bidir: enc_hiddenDim = 2 * encHiddenDim
        :return: 经过窗口化操作之后的encoderOutput [batchSize, sequenceLength - atten_Window_Size + 1, enc_hiddenDim * atten_Window_Size]
        """
        sequenceLength = encoderOutput.size(1)
        batchSize = encoderOutput.size(0)
        enc_hiddenDim = encoderOutput.size(2)
        
        new_encoder_output = torch.zeros(batchSize, sequenceLength - atten_Window_Size + 1, enc_hiddenDim * atten_Window_Size)
        if isinstance(hidden.data, torch.cuda.FloatTensor):
            new_encoder_output = new_encoder_output.cuda()

        # 将原始的encodrOutput进行重新拼接
        for i in range(sequenceLength - atten_Window_Size + 1):
            new_tensor = torch.cat((encoderOutput[:, i, :], encoderOutput[:, i + 1, :], encoderOutput[:, i + 2, :]), 1)
            new_encoder_output[:, i, :] = new_tensor

        return new_encoder_output


def test():
    attention = MultiSizeAttention(encHiddenDim = 5, decHiddenDim = 6)
    h = torch.randn(3, 6)
    enc_out = torch.randn(3, 10, 10)
    y = attention(h, enc_out)
    print(y)

if __name__ == '__main__':
    test()
 