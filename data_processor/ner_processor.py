import os, sys, random
import numpy as np
import jieba
from collections import Counter

class NerDataProcessor():
    def __init__(self, wiki_chs_dir='/data/wiki_chs.txt', num_step=10, dict_path=None):
        self.wiki_chs_dir=wiki_chs_dir
        self.dict_path=dict_path
        self.num_step=num_step
        self.s_list=[]
        self.label_list=[]
        self.cht_dict = {}
        self.cht_list = []
        self.num_words = 0
        self.load_wiki()

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
 
    def load_wiki(self):
        c=Counter()
        with open('/data/wiki_chs.txt','r') as fp:
             i = 0
             for line in fp:
                 if i == 100000: break
                 i+=1
                 print('load wiki line: ', i)
                 s, label = self.deal_line(line)
                 if i<50000: c.update(s)
                 print('sentence length is: ',len(s))
                 self.s_list.append(s)
                 self.label_list.append(label)
        cht_list = [x for x,n in c.items()]
        cht_list.sort(key=lambda x : c[x], reverse=True)
        self.cht_list=['UKN']+cht_list[:10000]
        self.cht_dict = {x:idx for idx,x in enumerate(self.cht_list)}
        self.num_words=len(self.cht_list)

    def batch_sample(self, batch_size=100, **kwargs):
        def get_single_sample(idx):
            sentence = self.s_list[idx]
            _label = self.label_list[idx]
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

        wt = kwargs['work_type'] if 'work_type' in kwargs else 'train'
        is_random = (wt != 'test')
        len_s_list = len(self.s_list)
        num_cv = 10000
        cur_idx = len_s_list-num_cv
        while True:
            if is_random:
                idx_list = np.random.randint(0, len_s_list-num_cv,[batch_size])
            else:
                if cur_idx+batch_size > len_s_list: cur_idx = len_s_list-batch_size
                idx_list = [i for i in range(cur_idx,cur_idx+batch_size)]
            sentence_list=[]
            sl_list = []
            data_list = []
            label_list = []
            for idx in idx_list:
                sentence, sl, data, label = get_single_sample(idx)
                sentence_list.append(sentence)
                sl_list.append(sl)
                data_list.append(data)
                label_list.append(label)
            yield sentence_list, np.array(sl_list), np.concatenate(data_list,axis=0), np.concatenate(label_list,axis=0)
                    


if __name__=='__main__':
    ndp = NerDataProcessor()
    for sentence, sl, data, label in ndp.batch_sample(batch_size=10, work_type='cv'):
        import pdb;pdb.set_trace()
