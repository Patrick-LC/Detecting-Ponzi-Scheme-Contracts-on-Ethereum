1.English Version
2.中文版

============================================
EPTransNet	(Ethereum Phising Transaction Network)							
============================================
By InPlusLab, Sun Yat-sen University


================== References ==================

• Website
	http://xblock.pro/ethereum/

• Cite
	@misc{
	   xblockEthereum,
	   author = {Chen, Liang and Peng, Jiaying and Liu, Yang and Li, Jintang and Xie, Fenfang and Zheng, Zibin},
	   title = {{XBLOCK Blockchain Datasets}: {InPlusLab} Ethereum Phishing Detection Datasets},
	   howpublished = {\url{http://xblock.pro/ethereum/}},
	   year = 2019
	}


================== Change Log =================

• Version 1.0, released on 01/11/2019



=================== Source ====================

This dataset is crawled from https://etherscan.io/accounts/label/phish-hack and https://etherscan.io


================ File Information =================


• Description
	Cryptocurrency, as blockchain’s most famous implementation, suffers a huge economic loss due to phishing scams. In our work, accounts and transactions in Ethereum are treated as nodes and edges, thus detection of phishing accounts can be modeled as a node classification problem. To tackle the problem, we propose a detecting method based on Graph Convolutional Network(GCN) and auto encoer to better mine structural information and precisely distinguish phishing accounts.
	In this work, we collected phishing nodes from Ethereum that reported in Etherscan labeled cloud (https://etherscan.io/accounts/label/phish-hack). Starting from phishing nodes we crawl a huge Ethereum transaction network via second-order BFS. Dataset contains 2,973,489 nodes, 13,551,303 edges and 1,165 labeled nodes.


• Contents
	- A multiple direct graph stored in pickle format.

* Notes * 
The file structure of EthereumG2 and EthereumG3 please refer to EthereumG1. 



———————————— Multiple ddirect graph ———————————— 

"The data stored in pickle format with version: 0.7.5 (python 3.7).
The type of graph object：networkx.classes.multidigraph.MultiDiGraph
Numbers of nodes: 2973489
Numbers of edges: 13551303
Average degree:   4.5574
Nodes' features：
    // The label. 1 means fishing mark node, otherwise 0.
    G.nodes[nodeName]['isp']；

Edges' features:
    G[node1][node2][0]['amount']        // The amount mount of the transaction.
    G[node1][node2][0]['timestamp']     // The timestamp of the transaction.				
							
							
		

* Notes * 
# Quick start tutorial.

import pickle 
import networkx as nx

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

G = load_pickle('./MulDiGraph.pkl')
print(nx.info(G))

# Traversal nodes:
for idx, nd in enumerate(nx.nodes(G)):
    print(nd)
    print(G.nodes[nd]['isp'])
    break

# Travelsal edges:
for ind, edge in enumerate(nx.edges(G)):
    (u, v) = edge
    eg = G[u][v][0]
    amo, tim = eg['amount'], eg['timestamp']
    print(amo, tim)
    break


===================Contact====================

Please contact Jiaying Peng (jypengg@gmail.com) for any questions about the dataset.



============================================
EPTransNet：以太坊钓鱼交易网络							
============================================
By InPlusLab, 中山大学


==================== 参考 =====================

• 相关网站
	http://xblock.pro/ethereum/

• 引用
	@misc{
	   xblockEthereum,
	   author = {Chen, Liang and Peng, Jiaying and Liu, Yang and Li, Jintang and Xie, Fenfang and Zheng, Zibin},
	   title = {{XBLOCK Blockchain Datasets}: {InPlusLab} Ethereum Phishing Detection Datasets},
	   howpublished = {\url{http://xblock.pro/ethereum/}},
	   year = 2019
	}


=================== 变更日志 ====================

• 版本 1.0, 发布于 01/11/2019


=================== 数据来源 ====================

从 https://etherscan.io/accounts/label/phish-hack 爬取的公示检举的钓鱼诈骗嫌疑账户
从 https://etherscan.io 利用相关apis从以太坊进行数据爬取


=================== 文件信息 ====================


• 描述
	作为区块链最著名的实现，加密货币由于钓鱼诈骗而遭受了巨大的经济损失。在我们的工作中，Ethereum中的账户和交易被视为节点和边，因此钓鱼账户的检测可以建模为节点分类问题。针对这一问题，我们提出了一种基于GCN和自动编码器的图卷积网络检测方法来更好地挖掘结构信息，准确区分钓鱼账户。
	本工作从Etherscan(https://etherscan.io/accounts/label/phish-hack) 中公开的有钓鱼结点举报的结点开始，通过二阶的广度搜索从以太坊交易记录爬取了一个大型的以太坊交易网络。 数据集中包括2,973,489个结点, 13,551,303条边和1,165个含标记的结点.							

• 内容
	数据以pickle格式存储，版本为0.7.5。python版本为3.7。
	图对象的类别：networkx.classes.multidigraph.MultiDiGraph
	结点数量: 2973489
	连边数量: 13551303
	平均度:   4.5574
	结点特征：
	    G.nodes[nodeName]['isp']；1为钓鱼标记结点，否则为0.

	连边特征：
	    G[node1][node2][0]['amount'] 交易的数额
	    G[node1][node2][0]['timestamp'] 交易的时间戳						
							
												
							
				
* 注释 * 
快速上手：

import pickle 
import networkx as nx

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

G = load_pickle('./MulDiGraph.pkl')
print(nx.info(G))

# 遍历结点：
for idx, nd in enumerate(nx.nodes(G)):
    print(nd)
    print(G.nodes[nd]['isp'])
    break
    
# 遍历连边：
for ind, edge in enumerate(nx.edges(G)):
    (u, v) = edge
    eg = G[u][v][0]
    amo, tim = eg['amount'], eg['timestamp']
    print(amo, tim)
    break


==================== 联系 =====================

若对数据集有任何问题请联系 彭嘉颖 (jypengg@gmail.com)。