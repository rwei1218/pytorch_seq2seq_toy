import torch

"""
对padding处并不进行计算
"""
def pad_cross_entropy(logits, target, pad_id):
    '''

    :param logits: batchSize , seqlen , decvocsize
    :param target: batchSize , seqlen
    :param pad_id:
    :return:
    '''
    logitFlat = logits.view(-1, logits.size(-1)) # batchSize * seqlen,  decvocsize
    targetFlat = target.view(-1, 1)              # batchSize * seqlen,  1
    lossFlat = - torch.gather(logitFlat, dim = 1, index=targetFlat)
    loss = lossFlat.view(*target.size())
    mask = (target != pad_id)
    mask = mask.type_as(loss)
    # if isinstance(loss.data, torch.cuda.FloatTensor):
    #     mask = mask.type_as(loss).cuda()
    # print 'loss', isinstance(loss.data, torch.cuda.FloatTensor), loss
    # print 'mask', isinstance(mask.data, torch.cuda.FloatTensor ) or isinstance(mask.data, torch.cuda.LongTensor), mask
    loss = loss * mask
    lost = loss.sum() / (mask.float().sum())
    return lost

if __name__ == '__main__':
    from torch.autograd import  Variable as V
    logits = V(torch.rand(2, 3, 3))
    import  torch.nn.functional as F
    logits = F.softmax(logits)
    print(logits)
    target = V(torch.LongTensor([[0, 1, 2], [1, 2, 0]]))
    print(pad_cross_entropy(logits, target, 3))
