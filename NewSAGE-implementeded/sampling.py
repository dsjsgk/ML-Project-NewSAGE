import numpy as np
import torch
"""
采样函数的定义
"""

use_new_sampling = False
def sampling(src_nodes, sample_num, neighbor_table,degree,x):
    """根据源节点采样指定数量的邻居节点，注意使用的是【有放回】的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现【重复的节点】
    
    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的节点数
        neighbor_table {dict} -- 节点到其邻居节点的映射表
    
    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []  # 存放采样后的节点集合（不包括自己）

    # 想要做的优化 不再等概率选择邻居，按照度数概率选择 IDEA1———————————————————————————————————————————————————————————————
    # 寄了 正确率下降%1左右
    # sampling 的时候SIZE大的有放回，SIZE小的无放回 IDEA2
    # 有效，收敛速度变快
    
    for sid in src_nodes:
        # 从节点的邻居中进行【有放回】地进行采样，有重复
        degree_list = np.zeros(len(neighbor_table[sid]))
        sum=0
        for i in range(len(neighbor_table[sid])):
            degree_list[i] = x[neighbor_table[sid][i]].dot(x[sid])+1e-6
        A = torch.from_numpy(degree_list)
        A = torch.nn.functional.dropout(A,p=0.2)
        for i in range(len(neighbor_table[sid])):
            A[i]+=1e-5
            sum+=A[i]
        A=A/sum
        # print([0,1].shape)
        if(use_new_sampling):
            if(sample_num<len(neighbor_table[sid])):
                res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=True,p=A)
            else:
                res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=True,p=A)
        else :
            if(sample_num<len(neighbor_table[sid])):
                res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=True)
            else:
                res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=True)
        # 加入集合
        results.append(res)
    return np.asarray(results).flatten()  # 将列表展平


def multihop_sampling(src_nodes, sample_nums, neighbor_table,degree,x):
    """根据源节点进行多阶采样
    
    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id
        sample_nums {list of int} -- 每一阶需要采样的个数
        neighbor_table {dict} -- 节点到其邻居节点的映射
    
    Returns:
        [list of ndarray] -- 每一阶采样的结果
    """
    sampling_result = [src_nodes]  # 结果是ndarray的列表
    for k, hopk_num in enumerate(sample_nums):  # 枚举，k阶邻居
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table,degree,x)
        # 注意，仍然保留了源节点列表，采样后的结果加在后面，这样保证了可以不断的向后采样
        # 最终sampling_result中存放有0，1，2，...，k阶的采样结果（邻居）
        sampling_result.append(hopk_result)
    return sampling_result
