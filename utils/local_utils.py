import re

def clean_str_en(string):
    """
    英文文本数据清洗
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(input_data_file, target_data_file):

    x_text = list(open(input_data_file, 'r', encoding='utf-8').readlines())
    x_text = [s.strip() for s in x_text]
    y_text = list(open(target_data_file, 'r', encoding='utf-8').readlines())
    y_text = [s.strip() for s in y_text]

    x_text = [clean_str_en(sent) for sent in x_text]
    x_text = [sent.split(' ') for sent in x_text]
    y_text = [clean_str_en(sent) for sent in y_text]
    y_text = [sent.split(' ') for sent in y_text]

    return x_text, y_text
