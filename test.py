import torch

label = torch.zeros(3, 6)  # 首先生成一个全零的多维数组
print("label：", label)
a = torch.ones(3, 5)

b = [[0, 1, 2], [0, 1, 3], [1, 2, 3]]
# 这里需要解释的是，b的行数要小于等于label的行数，列数要小于等于a的列数
print(a)
label.scatter_(1, torch.LongTensor(b), a)
# 参数解释：‘1’：需要赋值的维度；‘torch.LongTensor(b)’：需要赋值的索引；‘a’：要赋的值
print("new_label: ", label)
