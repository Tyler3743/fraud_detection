# **Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection** 

Jovan Blanuša Maximo Cravero Baraja IBM Research Europe Caltech Zurich, Switzerland Pasadena, CA, USA jov@zurich.ibm.com mcravero@caltech.edu 

Andreea Anghel Luc von Niederhäusern IBM Research Europe IBM Research Europe Zurich, Switzerland Zurich, Switzerland aan@zurich.ibm.com lvn@zurich.ibm.com 

Erik Altman Haris Pozidis Kubilay Atasu IBM Watson Research IBM Research Europe TU Delft Yorktown Heights, NY, USA Zurich, Switzerland Delft, Netherlands ealtman@us.ibm.com hap@zurich.ibm.com kubilay.atasu@tudelft.nl 

## **Abstract** 

In this paper, we present _Graph Feature Preprocessor_ , a software library for detecting typical money laundering patterns in financial transaction graphs in real time. These patterns are used to produce a rich set of transaction features for downstream machine learning training and inference tasks such as detection of fraudulent financial transactions. We show that our enriched transaction features dramatically improve the prediction accuracy of gradient-boostingbased machine learning models. Our library exploits multicore parallelism, maintains a dynamic in-memory graph, and efficiently mines subgraph patterns in the incoming transaction stream, which enables it to be operated in a streaming manner. Our solution, which combines our Graph Feature Preprocessor and gradient-boostingbased machine learning models, can detect illicit transactions with higher minority-class F1 scores than standard graph neural networks in anti-money laundering and phishing datasets. In addition, the end-to-end throughput rate of our solution executed on a multicore CPU outperforms the graph neural network baselines executed on a powerful V100 GPU. Overall, the combination of high accuracy, a high throughput rate, and low latency of our solution demonstrates the practical value of our library in real-world applications. To appear as a conference paper at ACM ICAIF’24. 

## **1 Introduction** 

Financial transactions serve as records documenting the movement of financial funds between accounts. Typically, these transactions are captured in a tabular format, where each row represents a distinct financial transaction, and columns represent basic transaction features such as timestamp, source account, target account, amount transferred, currency, and payment type [1]. While this tabular representation offers a structured view of the data, a more insightful approach emerges when financial transactions are represented as graphs by treating transactions as edges and accounts as vertices of a graph, as illustrated in Figure 1. Such a graph representation enables analysts to uncover insights that may not be immediately apparent in tabular formats. As a result, financial transaction graphs facilitate the efficient analysis and interpretation of complex financial data, aiding in the detection of financial crime [21, 54]. 







<!-- Start of picture text -->
JUN 2,  PUMP<br>MAY 3,  $ 2023 DUMP<br>2023 $ $<br>$ $$<br>$ $<br>MAY 26,  $$<br>2023 $ $$<br>$<br>JUN 8, 2023<br>JUN 1, 2023<br>(a) Circular money laundering (b) Pump and dump<br>$ $<br>$ $ $$$$<br>$$$$<br>$ $<br>$ $<br>(c) Smurfing<br><!-- End of picture text -->

**Figure 1: Crime patterns in financial transaction graphs.** 

Subgraph patterns in financial transaction graphs can often serve as indicators of financial crime. A _simple cycle_ [53], depicted in Figure 1a, is one such pattern and represents a sequence of transactions that transfer funds from one bank account back to the same account. Such a cycle can be an indicator of financial crimes such as money laundering, tax avoidance [32, 74], credit card frauds [54, 61], or circular trading used for stock price manipulation [38, 41, 57]. In addition, a _gather-scatter_ pattern, illustrated in Figure 1b, can suggest a _pump and dump_ stock manipulation scheme [54]. In this scheme, the stock price of a company is artificially increased through the use of social media to attract other traders for investment. After the stock price rises sufficiently, malicious traders sell the stocks. Due to the artificially inflated stock price, its value drops, and other traders suffer financial losses. Furthermore, a _scatter-gather_ pattern, depicted in Figure 1c, can represent a money laundering tactic called _smurfing_ [21, 44, 47, 48, 67, 72], in which a malicious actor employs several intermediary accounts (blue nodes in Figure 1c) to integrate small sums of illicit funds into the legal banking system. Similarly, in cryptocurrency transaction networks, criminals use sophisticated mixing and shuffling schemes to obfuscate the trace of their activities [49]. Such schemes can usually be represented by 

This work was performed when Maximo Cravero Baraja and Kubilay Atasu were with IBM Research Europe, Zurich, Switzerland. 

J. Blanuša, M. Cravero Baraja, A. Anghel, L. von Niederhäusern, E. Altman, H. Pozidis, and K. Atasu 



<!-- Start of picture text -->
Batch of financial transactions<br>ID Timestamp Src. account Dest. account Amount Currency Payment type<br>100 7 JUN 23, 12:45 A B 1250 USD Cheque<br>101 7 JUN 23, 17:12 B C 34 EUR Wire<br>102 8 JUN 23,   8:22 D C 648 CHF Credit card<br>Graph Feature Preprocessor<br>Batch of financial transactions with graph-based features<br>ID Timestamp Src. account Dest. account Amount Currency Payment type Graph-based features<br>100 7 JUN 23, 12:45 A B 1250 USD Cheque<br>101 7 JUN 23, 17:12 B C 34 EUR Wire<br>102 8 JUN 23,   8:22 D C 648 CHF Credit card<br>Gradient-boosting-based ML model<br>Suspicious transactions<br><!-- End of picture text -->

**Figure 2: The overview of our graph ML pipeline for the detection of suspicious financial transactions.** 

subgraph structures [17, 69, 78, 81]. The discovery of such suspicious subgraph patterns may enable locating and stopping criminal activities and their perpetrators. 

Rapid detection and processing of suspicious financial transactions are important to avoid financial losses. As financial data is often represented in a tabular format [1], the fastest and most accurate machine learning models [31] for this input format are gradient-boosting-based models [16, 43]. However, these models cannot take into account the underlying graph structure and cannot discover graph patterns that could be associated with financial crime. Furthermore, a limited set of basic features associated with financial transactions (see Figure 2) does not provide sufficient information to gradient-boosting-based models for detecting suspicious transactions with sufficient accuracy. As a result, the detection of suspicious transactions using these methods poses a challenge. 

To overcome the aforementioned limitations, we propose a solution shown in Figure 2. Specifically, we develop the _Graph Feature Preprocessor_ (GFP) library to produce a rich set of graph-based features for financial transactions. Our library searches for typical financial crime patterns, such as money laundering cycles and scatter-gather patterns (see Figure 1), and encodes these graph patterns into additional columns (i.e., features) of the transaction table. The transaction table enriched with the graph-based features is then forwarded to a pre-trained gradient-boosting-based machine learning model that performs the classification of financial transactions and detects suspicious transactions. As a result, the machine learning model is provided with additional transaction features extracted from the financial transaction graph, which facilitates the detection of transactions associated with financial crime. 

Our contributions can be summarised as follows: 

- We present a graph-based feature extraction library called Graph Feature Preprocessor for enriching the feature set of edges in financial transaction graphs by enumerating suspicious subgraph patterns in graphs as well as by computing various statistical properties of graph vertices. We then use this library to develop a graph machine learning (graph ML) pipeline for monitoring financial transaction networks. Section 2 introduces this library. 



<!-- Start of picture text -->
Graph Feature Preprocessor<br>Dynamic graph management<br>fit<br>create new graph<br>partial_fit update graph in-memory graph<br>Graph pattern mining<br>vertex<br>transform fan-in/out scatter-gather cycle statistics<br>basic<br> features<br> features<br>basic  graph-based<br>Transactions with<br>Input transactions  with  and<br><!-- End of picture text -->





**Figure 3: Our Graph Feature Preprocessor is offered as a scikit-learn preprocessor with** **_fit_ and** **_transform_ methods.** 

- We conduct experiments that demonstrate an improvement of up to 36% in the minority-class F1 score compared to graph neural network (GNN) baselines [12, 22, 35] for money laundering detection tasks. In addition, we demonstrate that our graph ML pipeline executed using 32 cores of an Intel Xeon processor achieves higher throughput rates compared to those GNN baselines executed on an NVIDIA Tesla V100 GPU. Our experimental evaluation is presented in Section 4. 

The GFP library is publicly available on PyPI as part of Snap ML [64–66]. In addition, it is offered with IBM<sup>1</sup> mainframe software products _Cloud Pak for Data on Z_ [37] and _AI Toolkit for IBM Z and LinuxONE_ [36]. Furthermore, an _AI on IBM Z Anti-Money Laundering Solution Template_ [68], which demonstrates how to develop and deploy a graph ML pipeline with GFP using an IBM Z environment, is open-sourced and publicly available<sup>2</sup> . 

## **2 Graph Feature Preprocessor** 

An overview of the Graph Feature Preprocessor (GFP) is given in Figure 3. It operates in a streaming fashion, receiving as input a batch of transactions with only basic features, such as in Figure 2, and producing additional graph-based features as output. GFP stores past financial transactions in an in-memory graph, which is dynamically updated as new transactions are received. The graph-based features are computed by enumerating subgraph patterns in the graph and by generating statistical properties of the accounts stored in that graph. GFP can compute the graph-based features across several CPU cores in parallel, which, together with the dynamic graph representation, enables real-time feature extraction. 

We have implemented GFP as a scikit-learn preprocessor with the _fit/transform_ interface [71] and made it publicly available on PyPI as part of the Snap ML package [64–66]. The main functionality of GFP is implemented by the _transform_ function, which is illustrated in Figure 3. This function inserts a batch of input transactions into the in-memory graph and computes graph-based features for these transactions. Creating the initial in-memory graph is performed by providing some past transactions as an input to the _fit_ function. The existing in-memory graph can be updated without computing any graph features by using the _partial_fit_ function. Other standard preprocessor functions supported by GFP are described in the publicly available documentation [65]. In the rest of this section, we describe the dynamic graph management and graph pattern 

> 1IBM, the IBM logo, and IBM Cloud Pak are trademarks or registered trademarks of International Business Machines Corporation, in the United States and/or other countries. 

> 2https://github.com/ambitus/aionz-st-anti-money-laundering 

Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection 



<!-- Start of picture text -->
Batch of 4 transactions<br>Batch of 4 input transactions with graph-based features<br>𝜀! 𝜀!<br>𝜀" Graph Feature  𝜀"<br>𝜀# Preprocessor 𝜀#<br>𝜀$ 𝜀$<br>𝜀! 𝜀" 𝜀# 𝜀$<br>Threads 0 to 10<br>Recursion trees for 4 input transactions<br><!-- End of picture text -->

**Figure 4: Fine-grained parallelism exploited by GFP. The library searches for cycles independently for each input transaction by recursively exploring the transaction graph. The coarse-grained approach would use only four threads, while the fine-grained approach uses eleven threads.** 

mining components of GFP (see Figure 3), and we describe how the graph-based features produced by the library are encoded. 

## **2.1 Dynamic Graph Management** 

The dynamic graph management component in GFP uses an inmemory graph to represent the financial transaction network. In this scenario, each account is treated as a graph vertex, and each transaction represents an edge from its source account to its destination account. As financial transactions typically include a _timestamp_ indicating when a transaction was created (see Figure 2), financial transaction graphs are considered _temporal graphs_ [34]. Furthermore, financial transaction graphs are also _multigraphs_ [3], as there can be several _parallel edges_ , i.e., edges that connect the same pair of source and destination vertices. Hence, our in-memory graph must be capable of representing temporal multigraphs. 

To enable the seamless processing of transactions in a streaming fashion, our in-memory graph must support the insertion of new transactions and the removal of outdated transactions. We define new transactions as those with timestamps greater than the timestamp of any transaction currently in the in-memory graph. Outdated transactions are identified as those with timestamps smaller than a value _𝑡now_ − _𝛿_ , where _𝑡now_ represents the largest timestamp among the transactions in the in-memory graph and _𝛿_ denotes a user-defined time window. Consequently, the in-memory graph retains only transactions that fall within the time window [ _𝑡now_ − _𝛿_ : _𝑡now_ ], effectively constraining its memory usage. 

Our in-memory graph comprises two main data structures: a _transaction log_ and an _index_ . The transaction log, implemented as a double-ended queue, maintains a list of edges sorted in ascending order of their timestamps. This data structure facilitates the detection and removal of outdated edges by supporting an _𝑂_ (1) operation for removing the edge with the smallest timestamp. The index data structure employs an _adjacency list_ representation to enable fast access to the neighbours of a vertex [20]. Implemented as a vector of hash maps [63], each entry in the vector represents a vertex _𝑣_ , and the hash map associated with that vertex _𝑣_ signifies the adjacency list of _𝑣_ . Vertices are internally mapped to integers 

|**Al**|**gorithm 1:**ScatterGatherStream <sup>�</sup>G(V_,_E)_, batch,𝛿𝑝_<br>�|
|---|---|
|**In**|**put:**G- the input graph with verticesV and edgesE<br>_batch_- a batch of edges; _𝛿𝑝_- the time window|
|**1 p**|**arallel foreach**(u→v_,𝑡𝑢𝑣_) _: batch_**do**<br><br><br><br>|
|**2**|TW=<br>�<br>_𝑡𝑢𝑣_−_𝛿𝑝_:_𝑡𝑢𝑣_<br>�<br>;<br>_⊲_Time window of size _𝛿𝑝_<br>// The first phase|
|**3**|_𝑁_<sup>+</sup>_𝑢_= { ∀_𝑥_| (_𝑢_→_𝑥, 𝑡𝑠_) ∈E ∧_𝑡𝑠_∈TW};|
|**4**|_𝑁_<sup>+</sup>_𝑣_= { ∀_𝑥_| (_𝑣_→_𝑥, 𝑡𝑠_) ∈E ∧_𝑡𝑠_∈TW};|
|**5**|**parallel foreach**w :_𝑁_<sup>+</sup>_𝑣_**do**|
|**6**|_𝑁_<sup>−</sup>_𝑤_= { ∀_𝑥_| (_𝑥_→_𝑤, 𝑡𝑠_) ∈E ∧_𝑡𝑠_∈TW};|
|**7**|_𝐼_=_𝑁_<sup>+</sup>_𝑢_∩_𝑁_<sup>−</sup>_𝑤_;|
|**8**|**if** |_𝐼_| ≥2**then**report scatter-gather pattern{_𝑢, 𝐼,𝑤_};<br>// The second phase|
|**9**|_𝑁_<sup>−</sup><br>_𝑢_<sup>= { ∀</sup><sup>_𝑥_| (</sup><sup>_𝑥_→</sup><sup>_𝑢, 𝑡𝑠_) ∈E ∧</sup><sup>_𝑡𝑠_∈TW };</sup>|
|**10**|_𝑁_<sup>−</sup>_𝑣_= { ∀_𝑥_| (_𝑥_→_𝑣, 𝑡𝑠_) ∈E ∧_𝑡𝑠_∈TW};|
|**11**|**parallel foreach**w :_𝑁_<sup>−</sup><br>_𝑢_<sup>**do**</sup>|
|**12**|_𝑁_<sup>+</sup>_𝑤_= { ∀_𝑥_| (_𝑤_→_𝑥, 𝑡𝑠_) ∈E ∧_𝑡𝑠_∈TW};|
|**13**|_𝐼_=_𝑁_<sup>−</sup>_𝑣_∩_𝑁_<sup>+</sup>_𝑤_;|
|**14**|**if** |_𝐼_| ≥2**then**report scatter-gather pattern{_𝑤, 𝐼, 𝑣_};|



in the range of 0 _,_ 1 _, . . . ,𝑛_ − 1, where _𝑛_ is the number of vertices in the graph. These integers are used to access the adjacency list of a vertex _𝑣_ in this vector. Furthermore, each edge can be accessed in _𝑂_ (1) time using the index, facilitating traversal through the graph, as required by the graph pattern mining component. 

To support the maintenance of parallel edges in the index, each entry in an adjacency list of the vertex _𝑣_ , representing a neighbour _𝑢_ of the vertex _𝑣_ , also contains a list of edges connecting _𝑣_ with _𝑢_ , referred to as the _parallel edge list_ . The edges in this list, also implemented as a double-ended queue, are represented with their ID and timestamp, sorted in ascending order of their timestamps. For this reason, the operations of inserting new edges and removing the outdated edges can be performed in _𝑂_ (1) time. 

## **2.2 Graph Pattern Mining** 

The task of the graph pattern mining component is to produce graph-based features for edges forwarded to the library through the _transform_ function. Two types of graph-based features are supported: _i)_ graph-pattern-based features and _ii)_ vertex-statisticsbased features. 

**Graph-pattern-based features** are computed by extracting graph patterns from the in-memory graph that contain one of the forwarded edges. Our library extracts the following graph patterns: fan-in, fan-out, scatter-gather, gather-scatter, simple cycle, and temporal cycle. Fan-in and fan-out patterns refer to patterns defined by a vertex _𝑣_ and all of its incoming and outgoing edges, respectively. A _gather-scatter_ pattern combines a fan-in pattern of the vertex _𝑣_ with a fan-out pattern of the same vertex _𝑣_ , as illustrated in Figure 1b [72]. A fan-out pattern of a vertex _𝑣_ and a fan-in pattern of a vertex _𝑢_ form a _scatter-gather_ pattern, depicted in Figure 1c, if the fan-out and the fan-in patterns connect vertices _𝑣_ and _𝑢_ , respectively, to the same set of intermediate vertices [72] (blue vertices in Figure 1c). A simple cycle is a path from vertex _𝑣_ to the same vertex _𝑣_ without repeated vertices except for the first and 

J. Blanuša, M. Cravero Baraja, A. Anghel, L. von Niederhäusern, E. Altman, H. Pozidis, and K. Atasu 



<!-- Start of picture text -->
𝑁$ %<br>𝑁! " 𝑁# " 𝑤 𝑁! " 𝑁# " 𝑤 𝐼= 𝑁! " ∩ 𝑁$ % 𝑤<br>𝑢 𝑢<br>𝑢<br>𝑣 𝑣<br>𝑣<br>(a) Determine  𝑁! " and  𝑁# " (b) Determine  𝑁$ % (c) The resulting pattern<br><!-- End of picture text -->

**Figure 5: Enumeration of scatter-gather patterns that contain the edge** _𝑢_ → _𝑣_ **with** _𝑣_ **being an intermediate vertex.** 

last vertex. Finally, a temporal cycle is a simple cycle with edges ordered in time. 

To compute graph-pattern-based features in a streaming manner, our library enumerates new patterns that are formed after inserting the input batch of edges into the graph. The fan-in and fan-out pattern features of a vertex _𝑣_ that belongs to the input batch are determined by counting the number of outgoing and incoming vertices of _𝑣_ , respectively. These features can be determined in _𝑂_ (1) time by simply querying the size of the hash maps that are implementing the adjacency lists of the vertex _𝑣_ in our index data structure (see Section 2.1). A gather-scatter pattern is detected implicitly if the fan-in and fan-out of a vertex _𝑣_ are at least two. Due to space constraints, we omit the description of our algorithm for finding scatter-gather patterns in a streaming manner. 

To enumerate simple cycles and temporal cycles in a streaming manner, we use fine-grained parallel algorithms introduced in Blanuša et al. [6, 7]. These algorithms enable the search for cycles that start from a single edge or a small batch of edges in parallel using several threads. The benefit of these algorithms is that they can process transactions in small batches with high throughput. For instance, if the computation of cycles is parallelised by adopting the _coarse-grained_ parallel approach, recursive cycle search for each edge of a batch is performed by a different thread. However, as shown in Blanuša et al [6, 7] using the coarse-grained approach might result in a suboptimal solution due to the potential workload imbalance across threads. In contrast, _fine-grained_ enumeration algorithms are able to execute the recursive cycle search from a single edge using several threads, as illustrated in Figure 4, thereby increasing the parallelism. As a result, even if the input batch contains one transaction, our library would be able to parallelise the search for cycles. 

To compute scatter-gather pattern in a streaming manner, we use our algorithm illustrated in Figure 5 and presented in Algorithm 1. In this algorithm, (u → v _,𝑡𝑢𝑣_ ) denotes a temporal edge with source vertex _𝑢_ , target vertex _𝑣_ and timestamp _𝑡𝑢𝑣_ . This algorithm processes each edge (u → v _,𝑡𝑢𝑣_ ) in the input batch by searching for all scatter-gather patterns that include that edge. The first and second phase of this algorithm search for scatter-gather patterns that contain _𝑣_ and _𝑢_ as an intermediate vertex, respectively. In the first phase, we first determine the outgoing neighbours of _𝑢_ and _𝑣_ , denoted as _𝑁𝑢_<sup>+</sup> and _𝑁𝑣_<sup>+</sup> , respectively, as shown in Figure 5a. Then, for each outgoing neighbour _𝑤_ of _𝑣_ , we search for incoming neighbours _𝑁𝑤_<sup>−</sup> of the vertex _𝑤_ , which are represented as filled circles in Figure 5b. Afterwards, we perform a set intersection between _𝑁𝑢_<sup>+</sup> and _𝑁𝑤_<sup>−</sup> to find the intermediate vertices _𝐼_ of a scatter gather pattern. Finally, 



<!-- Start of picture text -->
Basic  Graph-pattern-<br>transaction  based  Source account Target account<br>features transaction  features features<br>features<br>Scatter-gather Simple cycles  Temporal cycles<br>2 3 … ≥30 2 3 … ≥10 2 3 … ≥30<br>4 2<br>1 8<br>5 2 1<br>2<br>2 7<br>9 1<br>Multi-hop subgraph pattern transaction features<br>Timestamp Amount Timestamp Amount<br>Fan Deg. statistics statistics Fan Deg. statistics statistics<br>Out edges In edges<br>Target account features: fans, degrees and account statistics<br><!-- End of picture text -->

**Figure 6: Feature encoding: scatter-gather patterns are binned according to the number of intermediate vertices they have, and cycles are binned according to their length.** 

the algorithm reports the resulting scatter-gather pattern defined with vertices _𝑢_ , _𝑤_ , and _𝐼_ , as shown in Figure 5c. The second phase of this algorithm, presented in lines 9–14 of Algorithm 1, is analogous to the first phase, and we omit its description for brevity. Note that this algorithm can be parallelised in a fine-grained manner by parallelising its loops, as shown in Algorithm 1. 

Apart from parallelisation, another method to reduce the time required to find graph patterns is to impose time-window constraints. In this case, a time window parameter _𝛿𝑝_ can be specified for each graph pattern, in which case the library searches only for patterns whose edges have timestamps greater than or equal to _𝑡now_ − _𝛿𝑝_ , where _𝑡now_ represents the largest timestamp among the edges in the in-memory graph. Additionally, the search for simple cycles can be constrained by limiting their maximal length. 

**Vertex-statistics-based features** are computed for the vertices that appear in the input batch of edges. For each such vertex _𝑣_ , some predefined statistical property can be computed using a selected basic feature associated with the outgoing edges of _𝑣_ and its incoming edges. The statistical properties currently supported by our library are: sum, mean, minimum, maximum, median, variance, skew, and kurtosis [46]. For instance, if "Amount" is the selected basic feature used for the calculation of statistical properties, the statistical features include the average and total amount of money an account received or sent. Combining different statistical feature types with different user-specified basic features in this way extends the feature space significantly. 

Vertex-statistics-based features can be determined in a streaming manner through incremental computation. For this purpose, our library maintains second, third, and fourth central moments for each vertex of the graph and for each basic feature used for calculating account statistics (e.g., "Amount"). After inserting or removing an edge _𝑢_ → _𝑣_ , all central moments for _𝑢_ and _𝑣_ are updated incrementally [28, 75]. These central moments are then 

Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection 

**Table 1: Datasets used in the experiments.** 

|Dataset|# nodes|# edges|illicit rate|time span|
|---|---|---|---|---|
|AML HI Small|0.5 M|5 M|0.102%|10 days|
|AML HI Medium|2.1 M|32 M|0.110%|16 days|
|AML HI Large|2.1 M|180 M|0.124%|97 days|
|AML LI Small|0.7 M|7 M|0.051%|10 days|
|AML LI Medium|2.1 M|32 M|0.051%|16 days|
|AML LI Large|2.1 M|180 M|0.057%|97 days|
|ETH Phishing|2.9 M|13 M|0.278%|1261 days|



used to compute the following statistical features: sum, mean, variance, skew, and kurtosis [46]. Note that the computation of each aforementioned statistical feature can be performed in _𝑂_ (1) time. Other statistical features, i.e., minimum, maximum, and median, are simply computed by iterating through the incident edges of a vertex, which is executed in _𝑂_ (Δ) time per statistical feature, where Δ is the maximum degree of a vertex in the graph. 

**Table 2: Successive halving configurations used for hyperparameter tuning of both LightGBM and XGBoost models.** 

|Datasets|AML Small|AML Medium|AML Large|ETH|
|---|---|---|---|---|
|_𝑥_0|1000|100|16|100|
|_𝜂_|2|2|2|2|
|_𝑟_0|0_._1|0_._2|0_._2|0_._1|



**Table 3: Model parameter ranges used at tuning time.** 

|**LightGB**<br>|**M**<br>|**XGBoo**<br>|**st**<br>|
|---|---|---|---|
|**Parameter**|**Range**|**Parameter**|**Range**|
|num_round|(10_,_1000)|num_round|(10_,_1000)|
|num_leaves|(1_,_16384)|max_depth|(1_,_15)|
|learning_rate|10<sup>(−2</sup><sup>_._5</sup><sup>_,_−1)</sup>|learning_rate|10<sup>(−2</sup><sup>_._5</sup><sup>_,_−1)</sup>|
|lambda_l2|10<sup>(−2</sup><sup>_,_2)</sup>|lambda|10<sup>(−2</sup><sup>_,_2)</sup>|
|scale_pos_weight|(1_,_10)|scale_pos_weight|(1_,_10)|
|lambda_l1|10<sup>(0</sup><sup>_._01</sup><sup>_,_0</sup><sup>_._5)</sup>|colsample_bytree|(0_._5_,_1_._0)|
|||subsample|(0_._5_,_1_._0)|



early_stopping_rounds = 20 

## **2.3 Feature Encoding** 

The encoding of the features produced by the _transform_ function of GFP is shown in Figure 6. Each row of the output feature table stores the feature vector of a single transaction. Across different columns of a feature vector, there are basic transaction features, graph-pattern-based transaction features, and the account features of the source and the destination account of the transaction. The account features consist of vertex-statistics-based features and features based on fan-in and fan-out patterns, both of which are singlehop patterns. Features based on fan-in and fan-out patterns are computed for each account _𝑣_ and represent the number of accounts connected to _𝑣_ in those patterns. Graph-pattern-based transaction features are computed using multi-hop subgraph patterns: scattergather, hop-constrained simple cycles, and temporal cycles. For each transaction, our library reports the number of multi-hop subgraph patterns of different sizes that this transaction is part of. Example features based on multi-hop subgraph patterns are given in Figure 6, where the first transaction participates in 4 scatter-gather patterns with 3 intermediate vertices and in 2 temporal cycles with 30 or more edges. Even though these multi-hop subgraph patterns can also be used to compute account features, computing them as transaction features provides more compact feature vectors. 

## **3 Experimental setup** 

**Datasets.** Table 1 presents the datasets used in the evaluation. The AML datasets are publicly available synthetic AML datasets produced by the _AMLworld_ generator[1]. These datasets contain transactions labelled as licit or illicit, and, thus, they can be directly used with our graph ML pipeline that performs transaction classification. The datasets are available in two variants: one with a higher illicit rate (AML HI) and one with a lower illicit rate (AML LI). In addition, we use the ETH Phishing dataset, which is a real-world Ethereum dataset [15, 82] with 1 _,_ 165 accounts labelled as phishing. To enable transaction classification using the ETH Phishing dataset, we label a transaction of this dataset as phishing if its destination account is labelled as phishing. As a result, 0 _._ 278% of Ethereum transactions are labelled as phishing. 

**Baselines.** We use LightGBM (version 3.1.1) [43] and XGBoost (version 1.7.5) [16] boosting machines, which are widely-used ML models for tabular data, as machine learning models for our graph ML pipeline. We compare our graph ML pipeline with LightGBM and XGBoost models trained exclusively using basic features, without incorporating features generated by our Graph Feature Preprocessor. To perform hyper-parameter tuning of these models, we employ a successive halving model tuning approach [40]. As additional baselines, we use the following graph neural networks (GNNs): Graph Isomorphism Network (GIN) [35, 83], GIN with edge updates (GIN+EU) [4, 12], and Principal Neighbourhood Aggregation (PNA) [22, 77]. GIN+EU baseline is similar to LaundroGraph [12], which is a GNN specifically designed for anti-money laundering. The accuracy results for these GNNs on the AML datasets are obtained from Altman et al. [1]. Furthermore, all of the baselines, as well as our graph ML pipeline, are trained without the source and destination account IDs of the transactions. This prevents the models from identifying money laundering transactions based on the memorisation of account IDs. 

**Graph Feature Preprocessor setup.** We configure GFP to extract the graph-based features in the following way. The features are extracted from the AML datasets using a time window of six hours for scatter-gather patterns and a time window of one day for the rest of the graph-based features. We specify a cycle-length constraint of 10 for simple cycle enumeration. We use the "Amount" and "Timestamp" fields of the basic transaction features to generate the vertex-statistics-based features. Feature extraction from the ETH Phishing dataset is performed using a 20-day time window for all graph-based features. In addition, we disable the generation of temporal cycles and specify a hop constraint of 5 for simple cycle enumeration. We use the "Amount", "Timestamp", and "Block Nr." fields of the basic transaction features to generate the account statistics. We selected these parameters after some careful exploration aimed at finding the best trade-offs between the throughput of GFP and the accuracy of the ML models used for scoring. 

**Graph ML pipeline training.** The training step of our graph ML pipeline is illustrated in Figure 7a. First, the transactions available 

J. Blanuša, M. Cravero Baraja, A. Anghel, L. von Niederhäusern, E. Altman, H. Pozidis, and K. Atasu 



<!-- Start of picture text -->
Data splitting Graph Feature Preprocessor<br>Dynamic graph<br>transform Update  Create graph  Training<br>graph features ML model<br>Enriched train/val set<br>Train/val set<br>(a) Train pipeline<br>All transactions<br>Trained ML model<br>Graph Feature Preprocessor<br>fit Create<br>graph<br>Past transactions<br>Dynamic graph Load<br>transform Update  Create graph  ML model Suspicious<br>graph features transactions<br>Run inference<br>Test set<br>Bat ches of  test  Batches of enriched test<br>transactions (b) Inference pipeline transactions<br>Time<br><!-- End of picture text -->

**Figure 7: Train and inference components of our graph ML pipeline for the detection of suspicious transactions.** 

for training are ordered in ascending order of their timestamps and are split into train, validation, and test sets. This split is performed in such a way that the transactions from the train set have the lowest timestamps and the transactions from the test set have the highest. Then, the transactions from the train and validation sets are forwarded to GFP to generate the enriched graph-based features for the transactions from these two sets. To prevent any form of information leakage at training time, the training set is processed before the validation set. In that case, graph-based features for the transactions of the train set are computed on the graph created using only those transactions, and thus no information from the validation set is used. Finally, the train and validation sets with enriched features are then used to train the gradient boosting models [16, 43]. 

**Boosting machine parameter tuning.** As part of training the gradient-boosting-based models, we perform hyper-parameter tuning using the successive halving approach [40]. This approach starts by randomly sampling _𝑥_ 0 model parameter combinations using a fraction _𝑟_ 0 ≤ 1 of the train set. Then, for a given _𝜂_ > 1 parameter, the algorithm finds the best _𝑥_ 0/ _𝜂_ configurations, which are used in the next round of successive halving that uses _𝜂_ × _𝑟_ 0 of the train set. This process continues until the fraction of the training set used for evaluation reaches 1. The successive halving parameters used in our experiments are given in Table 2 and the parameter ranges of LightGBM and XGBoost models used for hyperparameter tuning are given in Table 3. 

**Graph ML inference.** The inference step of our graph ML pipeline is shown in Figure 7b. First, we load the model trained using the setup shown in Figure 7a. Then, we initialise GFP by loading past financial transactions using the _fit_ function. These past financial transactions are used to create the initial in-memory graph. Next, the transactions from the test set are grouped into batches and forwarded to GFP using the _transform_ function. This function updates the existing dynamic graph using the forwarded 

transactions and enriches those transactions with graph-based features of the same type as those generated in the train setup (see Figure 7a). Finally, the enriched test transactions are sent to the pre-trained machine learning model for detection of transactions associated with financial crime. 

**Data split.** To tune the parameters of the models and to test the model generalisation performance, we split the input data into train, validation, and test sets. The train and validation sets are used by the successive halving scheme to tune the model, while the test set is used for the final evaluation of the model. For AML datasets, the splitting is performed such that 60% of transactions with the smallest timestamps is selected as a training set, the next 20% transactions with the smallest timestamps excluding the ones from the training set are selected as a validation set, and the rest are selected as the test set. For the ETH dataset, we define the timestamp of an account as the minimum timestamp among the transactions that involve this account and split the accounts of the dataset such that 65% of the accounts with the smallest timestamp exist only in the training set, the next 15% of the accounts exist only in the validation dataset, and the rest are in the test set. Splitting the datasets in the aforementioned way prevents data leakage in our experiments. 

## **4 Results** 

In this section, we evaluate the accuracy of our graph ML pipeline and other baselines trained on the datasets from Table 1. We refer to our graph ML pipeline that uses LightGBM and XGBoost as GFP+LightGBM and GFP+XGBoost, respectively. As a measure of accuracy, we use the minority-class F1 score. The F1 scores reported are averaged across five different runs. The standard deviation of the F1 score is also reported for each experiment. 

Our graph ML pipeline requires transactions to arrive in batches. For the AML datasets, the graph ML pipeline uses batch sizes of 128 and 2048. In addition, for the ETH Phishing dataset, graph 

Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection 

**Table 4: Minority class F1 scores (%) of the money laundering detection task using the AML datasets and the phishing detection task using the ETH Phishing dataset. NA stands for not available.** 

|Model|size<br>batch|Small|AML HI<br>Medium|Large|Small|AML LI<br>Medium|size<br>batch<br>Large|ETH Phishing|
|---|---|---|---|---|---|---|---|---|
|GIN [35]|∞|28.70 ± 1.13|42.30 ± 0.44|NA|7.90 ± 2.78|3.86 ± 3.62|NA<br>∞|26.92 ± 7.52|
|GIN+EU [4, 12]|∞|47.73 ± 7.86|49.26 ± 4.02|NA|20.62 ± 2.41|6.19 ± 8.32|NA<br>∞|33.92 ± 7.34|
|PNA[22]|∞|56.77 ± 2.41|59.71 ± 1.91|NA|16.45 ± 1.46|27.73 ± 1.65|NA<br>∞|51.49 ± 4.29|
|LightGBM [43]|—|21.30 ± 0.30|18.60 ± 0.10|24.50 ± 0.20|2.05 ± 0.81|3.3 ± 0.48|4.04 ± 0.16<br>—|13.74 ± 0.54|
|**GFP**+LightGBM|128|62.86 ± 0.25|59.48 ± 0.15|58.03 ± 0.19|20.83 ± 1.50|24.74 ± 0.46|23.67 ± 0.11<br>128|40.17 ± 0.22|
|**GFP**+LightGBM|2048|60.52 ± 0.59|56.12 ± 0.37|54.76 ± 0.08|17.99 ± 0.60|21.06 ± 0.08|22.65 ± 0.59<br>∞|51.00 ± 1.01|
|XGBoost [16]|—|19.75 ± 0.89|20.10 ± 0.22|10.61 ± 6.73|0.21 ± 0.22|0.40 ± 0.14|0.00 ± 0.00<br>—|15.52 ± 0.15|
|**GFP**+XGBoost|128|63.23 ± 0.17|65.69 ± 0.26|42.68 ± 12.93|27.28 ± 0.69|31.03 ± 0.22|24.23 ± 0.12<br>128|37.01 ± 2.45|
|**GFP**+XGBoost|2048|64.77 ± 0.47|59.19 ± 0.29|56.88 ± 0.21|28.25 ± 0.80|21.36 ± 0.90|22.64 ± 0.15<br>∞|49.40 ± 0.54|
|GNNs:<br>GIN+EU<br>GIN<br>k<br>30 k<br>s]|PNA<br> <br>|GFP + LightGBM, b<br>GFP + XGBoost, ba|atch size:<br>128<br>tch size:<br>128|2048<br>2048|Scat<br>Simp|ter-Gather enum<br>le cycle enum.|.<br>Tempo<br>End-to|ral cycle enum.<br>-end|
|17 k<br>k<br>16 k<br>21 k<br>23<br>20 k<br>ans./||6 k<br> <br>5 k<br>k<br>k|19 k<br>18 k|k<br>k|40<br><br>AML|HI Small|40<br>AML HI Sm|all|
|<br>12<br> <br>8 k<br>8 k<br>0<br>10 k<br>ughput [tr|7 k<br>5 k<br>7 k<br>4 k<br>10 k<br>4 k<br>10 k|1<br>11 k<br>1<br>5 k<br> <br>12<br>5 k<br>14|7 k<br>5 k<br>7 k<br>6 k<br>4 k<br>10 k<br>6 k<br>4 k<br>10 k|5 k<br>12<br>4 k<br>13|10<br>20<br>30<br>Speedup<br>batc|h size = 2048|10<br>20<br>30<br>batch size|= ∞|
|AML HI<br>hro|AML HI|AML HI<br>A|ML LI<br>AML LI|AML LI|0||0||
|Small<br>T|Medium|Large<br>S|mall<br>Medium|Large|0|16<br>32<br>48|64<br><br>0<br>16|32<br>48<br>64|





<!-- Start of picture text -->
28.25 ± 0.80 21.36 ± 0.90 22.64 ± 0.15 ∞ 49.40 ± 0.54<br>Scatter-Gather enum. Temporal cycle enum.<br>Simple cycle enum. End-to-end<br>40 40<br>AML HI Small AML HI Small<br>30 batch size = 2048 30 batch size = ∞<br>20 20<br>10 10<br>0 0<br>0 16 32 48 64 0 16 32 48 64<br>40 40<br>AML HI Medium AML HI Medium<br>30 batch size = 2048 30 batch size = ∞<br>20 20<br>10 10<br>0 0<br>0 16 32 48 64 0 16 32 48 64<br>Number of threads Number of threads<br>Speedup<br>Speedup<br><!-- End of picture text -->

**Figure 8: Our graph ML pipeline has higher throughput compared to GNN baselines executed on a V100 GPU.** 

feature extraction is performed using batch sizes of 128 and ∞. When using a batch size of ∞, all the transactions of the test set are made available to GFP in a single batch. Using a batch size of ∞ essentially corresponds to an offline solution and, in principle, can lead to better accuracy because, in this case, the future transactions are also visible during feature extraction. However, if real-time processing capability is required by an application, the batch size will have to be constrained. Note that GNN baselines require the entire dataset to be available in memory, making it effectively an offline solution with batch size ∞. 

**Figure 9: Scalability of executing different parts of our GFP library, as well as its end-to-end execution. The speedup is relative to the single-threaded execution.** 

shown in Table 5. We observe that including graph features based on fan-in and fan-out patterns already improves the minority class F1 score by more than 30% compared to the case that uses only basic transaction features. Including multi-hop graph pattern features, i.e., features based on cycles and scatter-gather patterns, further improves the F1 score by up to 4%. Finally, by incorporating vertexstatistics-based features produced by GFP, our graph ML pipeline is able to achieve higher accuracy compared to the PNA baseline (see Table 4). Thus, each type of graph-based feature contributes to the overall accuracy of our graph ML pipeline. 

**AML results.** The minority class F1 scores of the ML models that perform laundering detection using AML datasets are shown in Table 4. Clearly, our graph-based features lead to significant improvements in the F1 scores achieved by gradient-boosting models. Without our graph-based features, the maximum F1 score that LightGBM and XGBoost achieve is 24 _._ 5% for the AML HI datasets and 4 _._ 04% for the AML LI datasets. The reason for this low accuracy is that the labels in AML datasets are highly imbalanced, and the number of illicit transactions in these datasets is at most 0 _._ 13% of the total number of transactions (see Table 1). Our graph ML pipeline, in which LightGBM and XGBoost models use our graph-based features in addition to basic features, achieves up to a 46% higher F1 scores than the models that use only basic features. Furthermore, our graph ML pipeline that uses XGBoost models consistently achieves higher F1 scores than GNN baselines. Compared to PNA, the GNN baseline with the highest accuracy, our graph ML pipeline with XGBoost achieves up to an 8% higher F1 score for AML HI datasets and up to an 11 _._ 8% higher F1 score for LI datasets. 

Figure 8 shows the throughput of our graph ML pipeline and GNN baselines. The performance of our graph ML pipeline is evaluated using 64 software threads of the Cascade Lake Intel Xeon Processor available from IBM Cloud [19], and the performance of GNN baselines is evaluated on an NVIDIA Tesla V100 GPU. We observe that our graph ML pipeline is able to achieve higher throughput than GNN baselines when it receives transactions in batches of 2048. This throughput is the result of the scalable parallel graph pattern mining algorithms that GFP uses, as shown in Figure 9. This figure also shows that our streaming scatter-gather algorithm, introduced in Section 2.2, scales almost linearly with the number of software threads when batch size is infinity. As a result 

The effect of different types of graph-based features produced by GFP on the accuracy of our graph ML pipeline for the AML task is 



<!-- Start of picture text -->
Target Sum ammountRecUSD Out _— +0.17<br>Temporal Cycle length 2 ns 0.14<br>Source Sum ammountRecUSDPayment FormatOut eda 0.06 +0.09<br>Source Var timestamp In es +0.05<br>Amount Received [USD] a +0.04<br>Source Fan Out—0.04 Si<br>Source Ratio Out a +0.04<br>Payment Currency Mm + 0.04<br>Target Fan Out Mig + 0.03<br>Target Ratio In Mmmm + 0.03<br>Target Ratio Out Mim +0.02<br>Simple Cycle length 2 Gum + 0.02<br>Target Sum timestamp Out —0.02 |<br>Source Sum ammountRecUSD In = +0.02<br>Amount Received mmm +0.02<br>Hour mmm +0.01<br>Source Ratio In -0.01 =<br>Target Var timestamp In —0:01 a<br>Sum of 113 other features —0.02 Emm<br>Se<br>—0.05 0.00 0.05 0.10 0.15<br>SHAP value<br><!-- End of picture text -->

Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection 

our graph ML pipeline to operate in a streaming manner with low per-batch latency and higher throughput compared to the GNN baselines presented in the experiments. This capability makes GFP suitable for scenarios that require real-time processing. 

We have also shown that the graph-based features generated by GFP can significantly improve the accuracy of gradient-boostingbased machine learning models. The graph-based features improve the minority class F1 score of gradient-boosting-based machine learning models by up to 46% for the synthetic AML datasets and by up to 35% for a real-world phishing detection dataset extracted from Ethereum. Furthermore, we show that our solution achieves up to a 36% higher F1 score than GNN baselines for the AML task. In particular, our graph ML pipeline achieves up to a 24% higher minority-class F1 score compared to the GIN+EU baseline with the similar architecture to LaundroGraph [12], which is a GNN designed specifically for anti-money laundering. 

The application scope of our GFP library is not limited to money laundering detection. Given that a cycle in a graph can be an indicator of tax avoidance [32], circular trading [38, 41, 57], and credit card frauds [54, 61], a GFP could also help to detect these types of frauds. However, the reliance on pre-defined subgraph patterns, such as cycles, is one drawback of this library, which we plan to address as part of the future work by adding the support for subgraph matching using user-defined subgraph patterns in GFP [73]. Furthermore, we plan to add support for feature extraction based on additional subgraph patterns, such as cliques [9] and bicliques [59]. Being able to enumerate these patterns could enable the detection of close-knit communities [51] as well as stacked money laundering patterns [1] encountered in various different financial crime scenarios. 

## **Acknowledgments** 

The support of Swiss National Science Foundation (project number 172610) for this work is gratefully acknowledged. The authors would like to thank Donna Eng Dillenberger, Thomas Parnell, Martin Petermann, Evan Rivera, and Elpida Tzortzatos from IBM for their support, feedback, and suggestions during the course of this work. 

## **References** 

- [1] Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. 2023. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. In _NeurIPS’23, Datasets and Benchmarks Track_ . 

- [2] Amazon. 2023. Amazon Fraud Detector. https://aws.amazon.com/fraud-detector/ Accessed: 2023-01-10. 

- [3] V K Balakrishnan. 1997. _Graph Theory_ . McGraw-Hill Professional, New York, NY. 

- [4] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. 2018. Relational inductive biases, deep learning, and graph networks. _arXiv preprint arXiv:1806.01261_ (2018). 

- [5] Austin R. Benson, David F. Gleich, and Jure Leskovec. 2016. Higher-order organization of complex networks. _Science_ 353, 6295 (2016), 163–166. https: //doi.org/10.1126/science.aad9029 

- [6] Jovan Blanuša, Paolo Ienne, and Kubilay Atasu. 2022. Scalable Fine-Grained Parallel Cycle Enumeration Algorithms. In _Proceedings of the 34th ACM Symposium on Parallelism in Algorithms and Architectures_ . ACM, Philadelphia PA USA, 247–258. https://doi.org/10.1145/3490148.3538585 

- [7] Jovan Blanuša, Kubilay Atasu, and Paolo Ienne. 2023. Fast Parallel Algorithms for Enumeration of Simple, Temporal, and Hop-constrained Cycles. _ACM Trans. Parallel Comput._ 10, 3 (Sept. 2023), 1–35. https://doi.org/10.1145/3611642 

- [8] Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, and Michael M. Bronstein. 2023. Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting. _IEEE Trans. Pattern Anal. Mach. Intell._ 45, 1 (Jan. 2023), 657–668. https://doi.org/10.1109/TPAMI.2022.3154319 

- [9] Coen Bron and Joep Kerbosch. 1973. Algorithm 457: finding all cliques of an undirected graph. _Commun. ACM_ 16, 9 (Sept. 1973), 575–577. https://doi.org/10. 1145/362342.362367 

- [10] Chiranjeeb Buragohain, Knut Magne Risvik, Paul Brett, Miguel Castro, Wonhee Cho, Joshua Cowhig, Nikolas Gloy, Karthik Kalyanaraman, Richendra Khanna, John Pao, Matthew Renzelmann, Alex Shamis, Timothy Tan, and Shuheng Zheng. 2020. A1: A Distributed In-Memory Graph Database. In _Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data_ . ACM, Portland OR USA, 329–344. https://doi.org/10.1145/3318464.3386135 

- [11] Shaosheng Cao, XinXing Yang, Cen Chen, Jun Zhou, Xiaolong Li, and Yuan Qi. 2019. TitAnt: online real-time transaction fraud detection in Ant Financial. _PVLDB_ 12, 12 (Aug. 2019), 2082–2093. https://doi.org/10.14778/3352063.3352126 

- [12] Mário Cardoso, Pedro Saleiro, and Pedro Bizarro. 2022. LaundroGraph: SelfSupervised Graph Representation Learning for Anti-Money Laundering. In _Proceedings of the Third ACM International Conference on AI in Finance_ . 130–138. 

- [13] Andrew Carter, Andrew Rodriguez, Yiming Yang, and Scott Meyer. 2019. Nanosecond Indexing of Graph Data With Hash Maps and VLists. In _Proceedings of the 2019 International Conference on Management of Data_ . ACM, Amsterdam Netherlands, 623–635. https://doi.org/10.1145/3299869.3314044 

- [14] Tao-Hung Chang and Davor Svetinovic. 2020. Improving Bitcoin Ownership Identification Using Transaction Patterns Analysis. _IEEE Transactions on Systems, Man, and Cybernetics: Systems_ 50, 1 (2020), 9–20. https://doi.org/10.1109/TSMC. 2018.2867497 

- [15] Liang Chen, Jiaying Peng, Yang Liu, Jintang Li, Fenfang Xie, and Zibin Zheng. 2019. XBLOCK Blockchain Datasets: InPlusLab Ethereum Phishing Detection Datasets. http://xblock.pro/ethereum/. 

- [16] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ (San Francisco, California, USA) _(KDD ’16)_ . ACM, New York, NY, USA, 785–794. https://doi.org/10.1145/2939672.2939785 

- [17] Xucan Chen, Mohammad Al Hasan, Xintao Wu, Pavel Skums, Mohammad Javad Feizollahi, Marie Ouellet, Eric L. Sevigny, David Maimon, and Yubao Wu. 2019. Characteristics of Bitcoin Transactions on Cryptomarkets. In _Security, Privacy, and Anonymity in Computation, Communication, and Storage_ , Guojun Wang, Jun Feng, Md Zakirul Alam Bhuiyan, and Rongxing Lu (Eds.). Vol. 11611. Springer International Publishing, Cham, 261–276. https://doi.org/10.1007/978-3-03024907-6_20 Series Title: Lecture Notes in Computer Science. 

- [18] Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. 2020. Can Graph Neural Networks Count Substructures?. In _NeurIPS 2020, December 6-12, 2020, virtual_ , Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). 

- [19] IBM Cloud. 2024. IBM Cloud Docs - Virtual Private Cloud (VPC). https: //cloud.ibm.com/docs/vpc Accessed: 2024-02-08. 

- [20] Thomas H. Cormen (Ed.). 2009. _Introduction to algorithms_ (3rd ed ed.). MIT Press, Cambridge, Mass. OCLC: ocn311310321. 

- [21] Livio Corselli. 2023. Italy: money transfer, money laundering and intermediary liability. _JFC_ 30, 2 (Feb. 2023), 377–388. https://doi.org/10.1108/JFC-10-2019-0137 

- [22] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. 2020. Principal Neighbourhood Aggregation for Graph Nets. In _Advances in Neural Information Processing Systems_ , H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 13260– 13271. 

- [23] Andras Cser, Merritt Maxix, Caroline Provost, and Peggy Dostie. 2022. _The Forrester Wave™: Anti-Money-Laundering Solutions, Q3 2022_ . Technical Report. Forrester. 1–10 pages. https://www.forrester.com/report/the-forrester-wavetm-anti-money-laundering-solutions-q3-2022/RES176346 Accessed: 2023-01-10. 

- [24] Ahmad Naser Eddin, Jacopo Bono, David Aparício, David Polido, João Tiago Ascensão, Pedro Bizarro, and Pedro Ribeiro. 2022. Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs. arXiv:2112.07508 [cs]. 

- [25] David Ediger, Rob McColl, Jason Riedy, and David A. Bader. 2012. STINGER: High performance data structure for streaming graphs. In _2012 IEEE Conference on High Performance Extreme Computing_ . IEEE, Waltham, MA, USA, 1–5. https: //doi.org/10.1109/HPEC.2012.6408680 

- [26] Chantat Eksombatchai, Pranav Jindal, Jerry Zitao Liu, Yuchen Liu, Rahul Sharma, Charles Sugnet, Mark Ulrich, and Jure Leskovec. 2018. Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time. In _Proceedings of the 2018 World Wide Web Conference_ (Lyon, France) _(WWW ’18)_ . 1775–1784. https://doi.org/10.1145/3178876.3186183 

- [27] Wenqi Fan, Yao Ma, Qing Li, Yuan He, Yihong Eric Zhao, Jiliang Tang, and Dawei Yin. 2019. Graph Neural Networks for Social Recommendation. In _The World Wide Web Conference, WWW 2019, San Francisco, CA, USA, May 13-17, 2019_ . ACM, 417–426. https://doi.org/10.1145/3308558.3313488 

- [28] Tony Finch. 2009. Incremental calculation of weighted mean and variance. (01 2009), 1–8. 

- [29] Per Fuchs, Domagoj Margan, and Jana Giceva. 2022. Sortledton: a universal, transactional graph data structure. _Proc. VLDB Endow._ 15, 6 (Feb. 2022), 1173–1186. https://doi.org/10.14778/3514061.3514065 

J. Blanuša, M. Cravero Baraja, A. Anghel, L. von Niederhäusern, E. Altman, H. Pozidis, and K. Atasu 

- [30] Thomas Gaudelet, Ben Day, Arian R Jamasb, Jyothish Soman, Cristian Regep, Gertrude Liu, Jeremy B R Hayter, Richard Vickers, Charles Roberts, Jian Tang, David Roblin, Tom L Blundell, Michael M Bronstein, and Jake P Taylor-King. 2021. Utilizing graph machine learning within drug discovery and development. _Briefings in Bioinformatics_ 22, 6 (05 2021). https://doi.org/10.1093/bib/bbab159 

- [31] Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. 2022. Why do tree-based models still outperform deep learning on typical tabular data?. In _36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks._ , S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 507–520. 

- [32] László Hajdu and Miklós Krész. 2020. Temporal Network Analytics for Fraud Detection in the Banking Sector. In _ADBIS, TPDL and EDA 2020 Common Workshops and Doctoral Consortium_ . Vol. 1260. Springer International Publishing, Cham, 145–157. https://doi.org/10.1007/978-3-030-55814-7_12 Series Title: Communications in Computer and Information Science. 

- [33] William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In _NIPS_ . 

- [34] Petter Holme and Jari Saramäki. 2012. Temporal networks. _Physics Reports_ 519, 3 (Oct. 2012), 97–125. https://doi.org/10.1016/j.physrep.2012.03.001 

- [35] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. 2019. Strategies for pre-training graph neural networks. _arXiv preprint arXiv:1905.12265_ (2019). 

- [36] IBM. 2023. AI Toolkit for IBM Z and LinuxONE. https://www.ibm.com/products/ ai-toolkit-for-z-and-linuxone Accessed: 2024-01-25. 

- [37] IBM. 2023. Cloud Pak for Data. https://www.ibm.com/products/cloud-pak-fordata Accessed: 2023-02-21. 

- [38] Md. Nazrul Islam, S. M. Rafizul Haque, Kaji Masudul Alam, and Md. Tarikuzzaman. 2009. An approach to improve collusion set detection using MCL algorithm. In _2009 12th International Conference on Computers and Information Technology_ . IEEE, Dhaka, Bangladesh, 237–242. https://doi.org/10.1109/ICCIT.2009.5407133 

- [39] Wole Jaiyeoba and Kevin Skadron. 2019. GraphTinker: A High Performance Data Structure for Dynamic Graph Processing. In _2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS)_ . IEEE, Rio de Janeiro, Brazil, 1030–1041. https://doi.org/10.1109/IPDPS.2019.00110 

- [40] Kevin Jamieson and Robert Nowak. 2014. Best-arm identification algorithms for multi-armed bandits in the fixed confidence setting. In _2014 48th Annual Conference on Information Sciences and Systems (CISS)_ . IEEE, Princeton, NJ, USA, 1–6. https://doi.org/10.1109/CISS.2014.6814096 

- [41] Zhi-Qiang Jiang, Wen-Jie Xie, Xiong Xiong, Wei Zhang, Yong-Jie Zhang, and Wei-Xing Zhou. 2013. Trading networks, abnormal motifs and stock manipulation. _Quantitative Finance Letters_ 1, 1 (Dec. 2013), 1–8. doi: 10.1080/21649502.2013.802877. 

- [42] Hiroki Kanezashi, Toyotaro Suzumura, Xin Liu, and Takahiro Hirofuchi. 2022. Ethereum Fraud Detection with Heterogeneous Graph Neural Networks. arXiv:2203.12363 [cs]. 

- [43] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In _Advances in Neural Information Processing Systems_ , Vol. 30. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2017/ file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf 

- [44] Nancy Kinnison and John Madinger (Eds.). 2011. _Money Laundering: A Guide for Criminal Investigators, Third Edition_ . Routledge, Boston, MA. 

- [45] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In _International Conference on Learning Representations_ . 

- [46] Stephen Kokoska and Daniel Zwillinger. 2000. _CRC Standard Probability and Statistics Tables and Formulae, Student Edition_ (0 ed.). CRC Press. https://doi. org/10.1201/b16923 

- [47] Meng-Chieh Lee, Yue Zhao, Aluna Wang, Pierre Jinghong Liang, Leman Akoglu, Vincent S. Tseng, and Christos Faloutsos. 2020. AutoAudit: Mining Accounting and Time-Evolving Graphs. In _2020 IEEE International Conference on Big Data (Big Data)_ . IEEE, Atlanta, GA, USA, 950–956. https://doi.org/10.1109/BigData50022. 2020.9378346 

- [48] Xiangfeng Li, Shenghua Liu, Zifeng Li, Xiaotian Han, Chuan Shi, Bryan Hooi, He Huang, and Xueqi Cheng. 2020. FlowScope: Spotting Money Laundering Based on Graphs. _AAAI_ 34, 04 (April 2020), 4731–4738. https://doi.org/10.1609/ aaai.v34i04.5906 

- [49] Xiao Fan Liu, Xin-Jian Jiang, Si-Hao Liu, and Chi Kong Tse. 2021. Knowledge Discovery in Cryptocurrency Transactions: A Survey. _IEEE Access_ 9 (2021), 37229–37254. https://doi.org/10.1109/ACCESS.2021.3062652 

- [50] Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, and Qing He. 2021. Pick and Choose: A GNN-Based Imbalanced Learning Approach for Fraud Detection. In _Proceedings of the Web Conference 2021_ (Ljubljana, Slovenia) _(WWW_ 

   - _’21)_ . Association for Computing Machinery, New York, NY, USA, 3168–3177. https://doi.org/10.1145/3442381.3449989 

- [51] Zhenqi Lu, Johan Wahlström, and Arye Nehorai. 2018. Community Detection in Complex Networks via Clique Conductance. _Sci Rep_ 8, 1 (Dec. 2018), 5982. 

https://doi.org/10.1038/s41598-018-23932-z 

- [52] Scott M Lundberg and Su-In Lee. 2017. A Unified Approach to Interpreting Model Predictions. In _Advances in Neural Information Processing Systems 30_ . Curran Associates, Inc., 4765–4774. 

- [53] Prabhaker Mateti and Narsingh Deo. 1976. On Algorithms for Enumerating All Circuits of a Graph. _SIAM J. Comput._ 5, 1 (March 1976), 90–99. https: //doi.org/10.1137/0205007 

- [54] Jack Nicholls, Aditya Kuppa, and Nhien-An Le-Khac. 2021. Financial Cybercrime: A Comprehensive Survey of Deep Learning Approaches to Tackle the Evolving Financial Crime Landscape. _IEEE Access_ 9 (2021), 163965–163986. https://doi. org/10.1109/ACCESS.2021.3134076 

- [55] Jack Nicholls, Aditya Kuppa, and Nhien-An Le-Khac. 2021. Financial Cybercrime: A Comprehensive Survey of Deep Learning Approaches to Tackle the Evolving Financial Crime Landscape. _IEEE Access_ 9 (2021), 163965–163986. https://doi. org/10.1109/ACCESS.2021.3134076 

- [56] Catarina Oliveira, João Torres, Maria Inês Silva, David Aparício, João Tiago Ascensão, and Pedro Bizarro. 2021. GuiltyWalker: Distance to illicit nodes in the Bitcoin network. arXiv:2102.05373 [cs]. 

- [57] Girish Keshav Palshikar and Manoj M. Apte. 2008. Collusion set detection using graph clustering. _Data Min Knowl Disc_ 16, 2 (April 2008), 135–164. https: //doi.org/10.1007/s10618-007-0076-8 

- [58] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: online learning of social representations. In _Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining_ . ACM, New York New York USA, 701–710. https://doi.org/10.1145/2623330.2623732 

- [59] Erich Prisner. 2000. Bicliques in Graphs I: Bounds on Their Number. _Combinatorica_ 20, 1 (Jan. 2000), 109–117. https://doi.org/10.1007/s004930070035 

- [60] Xiao Qin, Nasrullah Sheikh, Berthold Reinwald, and Lingfei Wu. 2021. Relationaware Graph Attention Model with Adaptive Self-adversarial Training. In _AAAI’21_ . AAAI Press, 9368–9376. 

- [61] Xiafei Qiu, Wubin Cen, Zhengping Qian, You Peng, Ying Zhang, Xuemin Lin, and Jingren Zhou. 2018. Real-time constrained cycle detection in large dynamic graphs. _PVLDB_ 11, 12 (Aug. 2018), 1876–1888. doi: 10.14778/3229863.3229874. 

- [62] Susie Xi Rao, Shuai Zhang, Zhichao Han, Zitao Zhang, Wei Min, Zhiyao Chen, Yinan Shan, Yang Zhao, and Ce Zhang. 2021. xFraud: explainable fraud transaction detection. _PVLDB_ 15, 3 (Nov. 2021), 427–436. https://doi.org/10.14778/ 3494124.3494128 

- [63] C++ reference. 2023. std::unordered_map. https://en.cppreference.com/w/cpp/ container/unordered_map Accessed: 2023-02-21. 

- [64] IBM Research. 2022. Graph Feature Preprocessor Public Examples. https://github.com/IBM/snapml-examples/blob/main/examples/graph_ feature_preprocessor/graph_feature_preprocessor.ipynb Accessed: 2023-03-3. 

- [65] IBM Research. 2022. Graph Feature PreprocessorDocumentation. https://snapml. readthedocs.io/en/latest/graph_preprocessor.html Accessed: 2023-01-10. 

- [66] IBM Research. 2022. Snap ML PyPI package. https://pypi.org/project/snapml/ Accessed: 2023-01-10. 

- [67] Peter Reuter and Edwin M. Truman. 2004. _Chasing Dirty Money: The Fight Against Money Laundering_ . Institute for International Economics, Washington, DC, Chapter Money Laundering: Methods and Markets. 

- [68] Evan Rivera, Jovan Blanuša, Jawaharlal Rajan, Alexis Landis, and Haris Pozidis. 2024. AI on IBM Z Anti-Money Laundering Solution Template. https://github. com/ambitus/aionz-st-anti-money-laundering Accessed: 2024-10-02. 

- [69] Viktoria Ronge, Christoph Egger, Russell W. F. Lai, Dominique Schröder, and Hoover H. F. Yin. 2021. Foundations of Ring Sampling. _Proceedings on Privacy Enhancing Technologies_ 2021, 3 (July 2021), 265–288. https://doi.org/10.2478/ popets-2021-0047 

- [70] Roman Schulte-Sasse, Stefan Budach, Denes Hnisz, and Annalisa Marsico. 2021. Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms. _Nature Machine Intelligence_ 3, 6 (2021), 513–526. https://doi.org/10.1038/s42256-021-00325-y 

- [71] scikit-learn developers. 2022. Scikit-learn: Preprocessing Data. https://scikitlearn.org/stable/modules/preprocessing.html Accessed: 2023-01-16. 

- [72] Michele Starnini and Charalampos E. Tsourakakis et al. 2021. Smurf-Based Antimoney Laundering in Time-Evolving Transaction Networks. In _Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track_ . Vol. 12978. Springer International Publishing, Cham, 171–186. https://doi.org/10.1007/9783-030-86514-6_11 

- [73] Shixuan Sun and Qiong Luo. 2020. In-Memory Subgraph Matching: An Indepth Study. In _Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data_ . ACM, Portland OR USA, 1083–1098. https://doi.org/10. 1145/3318464.3380581 

- [74] Toyotaro Suzumura and Hiroki Kanezashi. 2021. Anti-Money Laundering Datasets: InPlusLab Anti-Money Laundering DataDatasets. http://github.com/ IBM/AMLSim/. 

- [75] Katharina Tschumitschew and Frank Klawonn. 2012. Incremental Statistical Measures. In _Learning in Non-Stationary Environments_ , Moamar SayedMouchaweh and Edwin Lughofer (Eds.). Springer New York, New York, NY, 21–55. https://doi.org/10.1007/978-1-4419-8020-5_2 

Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection 

- [76] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. _International Conference on Learning Representations_ (2018). 

- [77] Petar Velickovic, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. 2019. Deep Graph Infomax. _ICLR (Poster)_ 2, 3 (2019), 4. 

- [78] Samourai Wallet. 2021. Whirlpool Coinjoin. https://samouraiwallet.com/ whirlpool 

- [79] Jianian Wang, Sheng Zhang, Yanghua Xiao, and Rui Song. 2021. A Review on Graph Neural Network Methods in Financial Applications. _CoRR_ abs/2111.15367 (2021). arXiv:2111.15367 

- [80] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom Robinson, and Charles E Leiserson. 2019. Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. _arXiv preprint arXiv:1908.02591_ (2019). 

- [81] Jiajing Wu, Jieli Liu, Weili Chen, Huawei Huang, Zibin Zheng, and Yan Zhang. 2021. Detecting Mixing Services via Mining Bitcoin Transaction Network With Hybrid Motifs. _IEEE Trans. Syst. Man Cybern, Syst._ (2021), 1–13. https://doi.org/ 10.1109/TSMC.2021.3049278 

- [82] Xblock. 2024. Ethereum Phishing Transaction Network. https://www.kaggle. com/datasets/xblock/ethereum-phishing-transaction-network Accessed: 202301-27. 

- [83] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How powerful are graph neural networks? _arXiv preprint arXiv:1810.00826_ (2018). 

- [84] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Cheekong Lee. 2021. Motifbased Graph Self-Supervised Learning for Molecular Property Prediction. _CoRR_ abs/2110.00987 (2021). arXiv:2110.00987 

- [85] Xiaowei Zhu, Guanyu Feng, Marco Serafini, Xiaosong Ma, Jiping Yu, Lei Xie, Ashraf Aboulnaga, and Wenguang Chen. 2020. LiveGraph: a transactional graph storage system with purely sequential adjacency list scans. _Proc. VLDB Endow._ 13, 7 (March 2020), 1020–1034. https://doi.org/10.14778/3384345.3384351 

- [86] Yongchun Zhu, Dongbo Xi, Bowen Song, Fuzhen Zhuang, Shuai Chen, Xi Gu, and Qing He. 2020. Modeling Users’ Behavior Sequences with Hierarchical Explainable Network for Cross-domain Fraud Detection. In _Proceedings of The Web Conference 2020_ . ACM, Taipei Taiwan, 928–938. https://doi.org/10.1145/ 3366423.3380172 

