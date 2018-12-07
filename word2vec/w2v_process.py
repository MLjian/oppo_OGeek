# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""

"""删除word2vec文件的第一行"""
with open('./sgns.zhihu.bigram-char', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('./sgns.zhihu_pro.bigram-char', 'w', encoding='utf-8') as f:
    lines_pro = list(map(lambda x: x + '\n', lines[1:]))
    f.writelines(lines_pro)

