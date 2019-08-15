import os, sys, random, linecache
import numpy as np
import jieba
from collections import Counter
    
def get_file_lines(filename):
    count = 0
    with open(filename, 'rb') as fr:
        while True:
            buffer = fr.read(1024 * 8192)
            if not buffer: break
            count += buffer.count('\n'.encode(encoding='utf-8'))
    print('------------ file %s lines: '%filename, count)
    return count


class WikiDataProcessor():
    def __init__(self, wiki_chs_file='/data/wiki_chs.txt', wiki_lines=0, num_step=10, dict_path='/root/tfNLP/data_processor/dict/han_dict.txt'):
        self.wiki_chs_file=wiki_chs_file
        self.wiki_lines = wiki_lines if wiki_lines > 0 else get_file_lines(wiki_chs_file)
        self.dict_path=dict_path
        self.num_step=num_step
        self.cht_dict = {}
        self.cht_list = []
        self.num_words = 0
        self.load_dict()

    def load_dict(self):
        with open(self.dict_path, 'r') as fp:
            for line in fp:
                self.cht_list.append(line[:-1])
        self.cht_dict = {x:idx for idx,x in enumerate(self.cht_list)}
        self.num_words=len(self.cht_list)

    def deal_line(self,line):
        s = ''
        label=[]
        for sub_line in line[:-1].split('  '):
            for w in sub_line.split():
                s += w
                label.append(1)
                label += [0 for i in range(len(w)-1)]
            s += ' '
            label.append(1)
        return s[:-1],label[:-1]   
    
    def get_single_sample(self,idx):
        line = linecache.getline(self.wiki_chs_file, idx)
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
        num_cv = 10000
        num_test = 10000
        cv_idx = 0
        test_idx = num_cv
        while True:
            if wt=='train':
                idx_list = np.random.randint(num_cv+num_test, self.wiki_lines-1,[batch_size])
            elif wt=='cv':
                idx_list = [i for i in range(cv_idx,cv_idx+batch_size)]
                cv_idx += batch_size
                if cv_idx>=num_cv: cv_idx==0
            elif wt=='test':
                idx_list = [i for i in range(test_idx,test_idx+batch_size)]
                test_idx+=batch_size
                if test_idx >= num_cv+num_test: test_idx=0
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
                    

class corpusZhDataProcessor():
    def __init__(self, corpusZh_path='/data/corpusZh/B.txt', corpusZh_lines=0):
        self.corpusZh_path = corpusZh_path
        self.corpusZh_lines = corpusZh_lines if corpusZh_lines>0 else get_file_lines(corpusZh_path)

    def deal_line(self, line):
        unit_list=[]
        for unit in line.split(' '):
            if '/' in unit:
                w, pos = unit.split('/')
                unit_list.append((w,pos))
        return unit_list
   
    def get_single_sample(self, idx):
        line = linecache.getline(self.corpusZh_path, idx)
        for w, pos in self.deal_line(line):
            print(w, pos)
         
    def batch_sample(batch_size=100):
        pass

if __name__=='__main__':
    #wdp = WikiDataProcessor(num_step=1000)
    #for sentence, sl, data, label in wdp.batch_sample(batch_size=1000, work_type='cv'): pass
    dp = corpusZhDataProcessor()
    dp.get_single_sample(2)
