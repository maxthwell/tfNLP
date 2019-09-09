import os, sys, random, linecache
import numpy as np
import jieba
from collections import Counter
from tfnlp.data_processor.posseg import pos_list, pos_dict
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__), path))

def get_file_lines(filename):
    count = 0
    with open(filename, 'rb') as fr:
        while True:
            buffer = fr.read(1024 * 8192)
            if not buffer: break
            count += buffer.count('\n'.encode(encoding='utf-8'))
    return count

class NerDataProcessor():
    def __init__(self, annotation_file, annotation_lines=0, num_step=100, dict_path=None):
        self.num_step=num_step
        self.annotation_file = annotation_file
        self.annotation_lines = annotation_lines if annotation_lines>0 else get_file_lines(annotation_file)
        self.dict_path=dict_path if dict_path else _get_module_path('dict/han_dict.txt')
        self.cht_dict = {}
        self.cht_list = []
        self.num_words = 0
        self.load_dict()
        print('        num_step: ',num_step)
        print('        annotation_file: ',annotation_file)
        print('        annotation_lines: ',self.annotation_lines)
        print('        dict_path: ', dict_path)
        print('        num_words', self.num_words)

    def load_dict(self):
        with open(self.dict_path, 'r') as fp:
            for line in fp:
                self.cht_list.append(line[:-1])
        self.cht_dict = {x:idx for idx,x in enumerate(self.cht_list)}
        self.num_words=len(self.cht_list)

    def deal_line(self,line):
        pass
    
    def get_single_sample(self,idx):
        line = linecache.getline(self.annotation_file, idx)
        sentence, _label = self.deal_line(line)
        len_sentence = len(sentence)
        if len_sentence > self.num_step:
            n_diff = len_sentence - self.num_step
            j = random.randint(0,n_diff)
            sentence = sentence[j:j+self.num_step]
            _label = _label[j:j+self.num_step]
            sl = self.num_step
        else:
            sl = len(sentence)
        data = np.zeros([1,self.num_step])
        label = np.zeros([1,self.num_step])
        for i in range(sl):
            w = sentence[i]
            data[0,i] = self.cht_dict[w] if w in self.cht_list else 0
            label[0,i] = _label[i]
        return sentence, sl, data, label

    def batch_sample(self, batch_size=100, **kwargs):
        wt = kwargs['work_type'] if 'work_type' in kwargs else 'train'
        train_line_list=[i for i in range(self.annotation_lines) if i%100>1]
        cv_line_list = [i for i in range(self.annotation_lines) if i%100==1]
        test_line_list = [i for i in range(self.annotation_lines) if i%100==0]
        while True:
            if wt=='train':
                idx_list = random.choices(train_line_list, k=batch_size)
            elif wt=='cv':
                idx_list = random.choices(cv_line_list, k=batch_size)
            elif wt=='test':
                idx_list = random.choices(test_line_list, k=batch_size)
            sentence_list=[]
            sl_list = []
            data_list = []
            label_list = []
            for idx in idx_list:
                sentence, sl, data, label = self.get_single_sample(idx)
                sentence_list.append(sentence)
                sl_list.append(sl)
                data_list.append(data)
                label_list.append(label)
            yield sentence_list, np.array(sl_list), np.concatenate(data_list,axis=0), np.concatenate(label_list,axis=0)

class WikiDataProcessor(NerDataProcessor):
    def deal_line(self,line):
        s_list = []
        label=[]
        for sub_line in line[:-1].split('  '):
            for w in sub_line.split():
                s_list += [s for s in w]
                label.append(1)
                label += [0 for i in range(len(w)-1)]
            s_list.append(' ')
            label.append(1)
        return s_list[:-1], label[:-1]   
                    
class CorpusZhDataProcessor(NerDataProcessor):
    def deal_line(self, line):
        s_list = []
        label_list = []
        for unit in line.split(' '):
            if '/' in unit:
                p=-2
                try:
                    if unit[-2] == '/':
                        p=-2
                    elif unit[-3] == '/':
                        p=-3
                    elif unit[-4] == '/':
                        p=-4
                except:
                    continue
                w = unit[:p]
                pos = unit[p+1:]
                if pos not in pos_dict: continue
                head_flag=1
                for c in w:
                    s_list.append(c)
                    label_list.append(pos_dict[pos]+35*head_flag)
                    head_flag=0
        return s_list[:-1], label_list[:-1]
   
if __name__=='__main__':
    #dp = WikiDataProcessor(num_step=20,annotation_file='/data/wiki_corpus_chs.txt')
    dp = CorpusZhDataProcessor(annotation_file='/data/corpusZh/B.txt')
    for sentence, sl, data, label in dp.batch_sample(batch_size=1000, work_type='cv'): import pdb;pdb.set_trace()
