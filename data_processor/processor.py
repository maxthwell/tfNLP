import os, sys, random
import jieba

class BaseDataProcessor():
    def __init__(self):
        pass
    
    def batch_sample(batch_size=100, **kwargs):
        pass


class ClassiffierDataProcessor(BaseDataProcessor):
    def __init__(self, train_data_dir, test_data_dir, cv_data_dir, num_steps=10, dict_path=None):
        #训练集目录
        self.train_data_dir=train_data_dir
        #测试集目录
        self.test_data_dir=test_data_dir
        #交叉验证集目录
        self.cv_data_dir=cv_data_dir
        self.dict_path=dict_path
        assert(os.listdir(self.train_data_dir), os.listdir(self.test_data_dir))
        assert(os.listdir(self.train_data_dir), os.listdir(self.cv_data_dir))
        self.label_list = os.listdir(self.test_data_dir)
        self.num_labels = len(self.label_list)
        self.num_steps=num_steps
        self.word_dict = self.load_dict()
    
    def load_dict(self):
        def load(fp):
            wd = {}
            idx=1
            for line in fp:
                w=str(line,'utf8').split()[0]
                print(w)
                wd[w]=idx
                idx += 1
            return wd

        if self.dict_path:
            with open(self.dict_path, 'r') as fp:
                return load(fp)
        with jieba.get_dict_file() as fp:
            return load(fp)
      

    def batch_sample(self, batch_size=100, **kwargs):
        wt = kwargs['work_type'] if 'work_type' in kwargs else 'train'
        if wt=='train':
            data_dir=self.train_data_dir
        elif wt=='test':
            data_dir=self.test_data_dir
        elif wt=='cv':
            data_dir=self.cv_data_dir
        is_random = (wt=='train')
        all_files = []
        for label in self.label_list:
            for fn in os.listdir('%s/%s'%(data_dir,label)):
                all_files.append((data_dir,label,fn))
        len_samples = len(all_files)
        i=0
        while True:
            if is_random:
                sample_file_list = random.sample(all_files,batch_size)
            else:
                sample_file_list = all_files[i:i+batch_size]
            
            N=[]
            S=[]
            X=[]
            Y=[]
            for fs in sample_file_list:
                N.append('%s/%s/%s'%fs)
                s,x = self.process_file(fs)
                S.append(s)
                X.append(x)
                Y.append(fs[1])
            yield N,S,X,Y

    def process_file(self, fs):
        with open('%s/%s/%s'%fs, 'r') as fp: s = fp.read()
        #先进行分词  
        words = list(jieba.cut(s, cut_all=True)) 
        #初始化向量序列  
        data = [0 for i in range(self.num_steps)] 
        j = 0  
        #按照词序，依次把用词向量填充序列  
        for i in range(len(words)):
            if i == self.num_steps:  
                break  
            w = words[i]  
            if w in self.word_dict:  
                data[i] = self.word_dict[w]  
        return i, data 

if __name__=='__main__':
    cdp = ClassiffierDataProcessor(
      train_data_dir='/data/THUCNews',
      test_data_dir='/data/THUCNewsTest',
      cv_data_dir='/data/THUCNewsTest',
    )
    for N,S,X,Y in cdp.batch_sample(2):
        print(N)
        print(S)
        print(X)
        print(Y)
        sys.exit(0)
