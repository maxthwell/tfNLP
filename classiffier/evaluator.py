import numpy as np

class ClassiffierModelEvaluator():
    def __init__(self, num_label=2, label_name_list=None):
        self.num_label=num_label
        self.sts = np.zeros([num_label,num_label],dtype=np.int32)
        self.label_name_list = label_name_list if label_name_list else ['cls_%s'%i for i in range(num_label)]
        self.quota=None

    def add(self,r, c):
        self.sts[r,c] += 1

    def batch_add(self, all_label):
        for r,c in all_label:
            self.sts[r,c] += 1

    def update_quota(self):
        if not self.label_name_list:
            self.label_name_list=[range(self.sts.shape[0])]
        sum_r = np.sum(self.sts,axis=1)
        sum_c = np.sum(self.sts,axis=0)
        quota = np.zeros([3, self.num_label],dtype=np.float)
        rows, cols = self.sts.shape
        for c in range(cols):
            quota[0,c] = self.sts[c,c] / sum_r[c]
            quota[1,c] = self.sts[c,c] / sum_c[c]
        quota[2] = quota[0] * quota[1] * 2 / (quota[0] + quota[1])
        macro_avg = np.mean(quota,axis=1)
        macro_avg = np.reshape(macro_avg,[3,1])
        micro_avg = np.ones([3,1]) * np.trace(self.sts) / np.sum(self.sts) #矩阵的迹/矩阵所有元素之和
        quota = np.concatenate([quota, macro_avg,micro_avg], axis=1)
        self.quota = np.around(quota, decimals=4)

    def __str__(self):
        from prettytable import PrettyTable
        table = PrettyTable(['真实标签\预测标签'] + self.label_name_list + ['宏平均','微平均'])
        for r in range(self.num_label):
            table.add_row([self.label_name_list[r]]+list(self.sts[r])+['/', '/'])
        table.add_row(['召回率']+list(self.quota[0]))
        table.add_row(['精确率']+list(self.quota[1]))
        table.add_row(['F1值']+list(self.quota[2]))
        return str(table)

if __name__=='__main__':
    cme = ClassiffierModelEvaluator(num_label=2,label_name_list=['体育','娱乐'])
    cme.add(0,0)
    cme.add(0,1)
    cme.add(0,1)
    cme.add(1,0)
    cme.add(1,0)
    cme.add(1,0)
    cme.add(1,1)
    cme.update_quota()
    print(cme)



