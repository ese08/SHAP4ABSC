# coding:utf-8

import numpy as np

with open('result.txt', 'r', encoding='utf-8') as f:
	data = f.readlines()
	data = list(map(lambda x: x.strip(), data))
	std = np.std(np.array(data).astype(float), ddof=1)
	std_2 = np.std(np.array(data).astype(float), ddof=0)  # divided by n
	max_mean = np.mean(np.array(data).astype(float))	
	print('Total: {} results, mean: {:.4f}, sample standard deviation: {:.4f}, population standard deviation: {:.4f}'.format(len(data), max_mean, std, std_2))
