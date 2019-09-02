import os, sys, random

class WordEmbeddingClassiffierDataProcessor():
    def __init__(self, num_label, num_step=1000, dict_path=None):
        self.dict_path = dict_path
        self.num_label = num_label
        self.num_step=num_step
        self.word_dict = self.load_dict()
        self.num_words = len(self.word_dict)

    def load_dict(self):
        def load(fp):
            wd = {}
            idx=1
            for line in fp:
                w=str(line,'utf8').split()[0]
                wd[w]=idx
                idx += 1
            return wd

        if self.dict_path:
            with open(self.dict_path, 'r') as fp:
                return load(fp)
        import jieba
        with jieba.get_dict_file() as fp:
            return load(fp)
      
class LocalFileClassiffierDataProcessor(WordEmbeddingClassiffierDataProcessor):
    def __init__(self, train_data_dir, num_step=1000, dict_path=None):
        #训练集目录
        self.train_data_dir=train_data_dir
        self.dict_path=dict_path
        self.label_list = os.listdir(self.train_data_dir)
        self.label_dict = {label:idx for idx,label in enumerate(self.label_list)}
        num_label = len(self.label_list)
        super(LocalFileClassiffierDataProcessor,self).__init__(num_label,num_step,dict_path=dict_path) 

    def batch_sample(self, batch_size=100, **kwargs):
        wt = kwargs['work_type'] if 'work_type' in kwargs else 'train'
        all_files = []
        for label in self.label_list:
            for fn in os.listdir('%s/%s'%(self.train_data_dir,label)):
                all_files.append((self.train_data_dir,label,fn))
        len_samples = len(all_files)
        test_fid_list = [i for i in range(len_samples) if i%100==0]
        cv_fid_list = [i for i in range(len_samples) if i%100==1]
        train_fid_list = [i for i in range(len_samples) if i%100>1]
        i=0
        while True:
            if wt=='train':
                sample_fid_list = random.choices(train_fid_list, k=batch_size)
            elif wt=='test':
                sample_fid_list = test_fid_list[i:i+batch_size]
                i += batch_size
                if i >= len(test_fid_list): i=0
            elif wt=='cv':
                sample_fid_list = random.choices(cv_fid_list, k=batch_size)
            N=[]
            S=[]
            X=[]
            Y=[]
            for fid in sample_fid_list:
                fs = all_files[fid]
                N.append('%s/%s/%s'%fs)
                s,x = self.process_file(fs)
                S.append(s)
                X.append(x)
                Y.append(self.label_dict[fs[1]])
            yield N,S,X,Y

    def process_file(self, fs):
        import jieba
        with open('%s/%s/%s'%fs, 'r') as fp: s = fp.read()
        #先进行分词  
        words = list(jieba.cut(s, cut_all=True)) 
        #初始化向量序列  
        data = [0 for i in range(self.num_step)] 
        j = 0  
        #按照词序，依次把用词向量填充序列  
        for i in range(len(words)):
            if i == self.num_step:  
                break  
            w = words[i]  
            if w in self.word_dict:  
                data[i] = self.word_dict[w]  
        return i, data 


class EsDataProcessor(WordEmbeddingClassiffierDataProcessor):
    def __init__(self, num_label=10, num_step=1000, es_point='127.0.0.1:9200'):
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch(es_point)
        super(EsClassiffierDataProcessor, self).__init__(num_label,num_step)
    
    def batch_sample(self, batch_size, **kwargs):
        query = {'query':{'term':{'label_type':'manual'}}}
        num_sample = self.es.count(query)['count']
        from_id = 0
        while True:
            query['from']=from_id
            query['size']=batch_size
            from_id += batch_size
            if from_id >= num_sample:
                from_id=0
            sentence_list=[]
            sl_list=[]
            data_list=[]
            label_list=[]
            for items in self.es.search(body=query)['hits']['hits']:
                sentence,sl,data,label = deal_source(item['_source'])
                sentence_list.append(sentence)
                sl_list.append(sl)
                data_list.append(data)
                label_list.append(label)
            yield sentence_list, sl_list, data_list, label_list
        
    def deal_source(self, _source):
        pass

if __name__=='__main__':
    cdp = LocalFileClassiffierDataProcessor(
      train_data_dir='/data/THUCNews',
    )
    for N,S,X,Y in cdp.batch_sample(2):
        print(N)
        print(S)
        print(X)
        print(Y)
        sys.exit(0)
