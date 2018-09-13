import random
import numpy as np
from utils.word_seq import WordSequence

def transform_sentence(sentence, ws, max_len=None, add_end=False):
    """转换一个单独句子
    Args:
        sentence: 一句话，例如一个数组['你', '好', '吗']
        ws: 一个WordSequence对象，转换器
        max_len:
            进行padding的长度，也就是如果sentence长度小于max_len
            则padding到max_len这么长
    Ret:
        encoded:
            一个经过ws转换的数组，例如[4, 5, 6, 3]
        encoded_len: 上面的长度
    """
    encoded = ws.transform(
        sentence,
        max_len=max_len if max_len is not None else len(sentence)
    )
    encoded_len = len(sentence)
    if encoded_len > len(encoded):
        encoded_len = len(encoded)
    return encoded, encoded_len

def batch_flow_seq2seq(data, ws, batch_size, raw=False, add_end=True):
    """从数据中随机 batch_size 个的数据，然后 yield 出去
    Args:
        data:
            是一个数组，必须包含一个或者更多个同等的数据队列数组
        ws:
            可以是一个WordSequence对象，也可以是多个组成的数组
            如果是多个，那么数组数量应该与data的数据数量保持一致，即len(data) == len(ws)
        batch_size:
            批量的大小
        raw:
            是否返回原始对象，如果为True，假设结果ret，那么len(ret) == len(data) * 3
            如果为False，那么len(ret) == len(data) * 2
    例如需要输入问题与答案的队列，问题队列Q = (q_1, q_2, q_3 ... q_n)
    答案队列A = (a_1, a_2, a_3 ... a_n)，有len(Q) == len(A)
    ws是一个Q与A共用的WordSequence对象，
    那么可以有： batch_flow([Q, A], ws, batch_size=32)
    这样会返回一个generator，每次next(generator)会返回一个包含4个对象的数组，分别代表：
    next(generator) == q_i_encoded, q_i_len, a_i_encoded, a_i_len
    如果设置raw = True，则：
    next(generator) == q_i_encoded, q_i_len, q_i, a_i_encoded, a_i_len, a_i
    其中 q_i_encoded 相当于 ws.transform(q_i)
    不过经过了batch修正，把一个batch中每个结果的长度，padding到了数组内最大的句子长度
    """

    all_data = list(zip(*data))

    if isinstance(ws, (list, tuple)):
        assert len(ws) == len(data), \
            'len(ws) must equal to len(data) if ws is list or tuple'

    if isinstance(add_end, bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end, (list, tuple))), \
            'add_end 不是 boolean，就应该是一个list(tuple) of boolean'
        assert len(add_end) == len(data), \
            '如果 add_end 是list(tuple)，那么 add_end 的长度应该和输入数据长度一致'

    mul = 2
    if raw:
        mul = 3

    while True:
        data_batch = random.sample(all_data, batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j], '__len__') else 0
                for x in data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws, (list, tuple)):
                    w = ws[j]
                else:
                    w = ws

                # 添加结尾
                line = d[j]
                if add_end[j] and isinstance(line, (tuple, list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x, xl = transform_sentence(line, w, max_lens[j], add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul + 1].append(xl)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for x in batches]

        yield batches

def test_batch_flow():
    x_data = [
        ['i', 'have', 'a', 'dream'],
        ['that', 'my', 'four', 'little', 'children'],
        ['will', 'one', 'day', 'live', 'in', 'a', 'nation'],
        ['where', 'they', 'will', 'not', 'be', 'judged', 'by', 'the', 'color', 'of', 'their', 'skin'],
    ]
    y_data = [
        ['我', '有', '一个', '梦想'],
        ['我的', '四个', '孩子'],
        ['他们', '将', '生活', '在'],
        ['一个', '不', '以', '肤色', '作为', '评价标准', '的', '国家']
    ]
    ws_X = WordSequence()
    ws_Y = WordSequence()
    ws_X.fit(sentence_set=x_data, min_count=1)
    ws_Y.fit(sentence_set=y_data, min_count=1)
    flow = batch_flow_seq2seq([x_data, y_data], [ws_X, ws_Y], 2)
    x, x_l, y, y_l = next(flow)
    print('x: ', x)
    print('x_l: ', x_l)
    print('y: ', y)
    print('y_l: ', y_l)

if __name__ == '__main__':
    test_batch_flow()