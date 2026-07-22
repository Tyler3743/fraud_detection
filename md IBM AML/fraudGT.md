# **FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection** 

Junhong Lin Massachusetts Institute of Technology US junhong@mit.edu 

## Yada Zhu 

Xiaojie Guo 

IBM T.J. Watson Research Center US xiaojie.guo@ibm.com 

IBM T.J. Watson Research Center US yzhu@us.ibm.com 

Erik Altman 

Erik Altman Julian Shun IBM T.J. Watson Research Center Massachusetts Institute of Technology US US ealtman@us.ibm.com jshun@mit.edu 

Samuel Mitchell 

Massachusetts Institute of Technology US sammit@mit.edu 

### **Abstract** 

### **Keywords** 

Fraud detection plays a crucial role in the financial industry, preventing significant financial losses. Traditional rule-based systems and manual audits often struggle with the evolving nature of fraud schemes and the vast volume of transactions. Recent advances in machine learning, particularly graph neural networks (GNNs), have shown promise in addressing these challenges. However, GNNs still face limitations in learning intricate patterns, effectively utilizing edge attributes, and maintaining efficiency on large financial graphs. To address these limitations, we introduce FraudGT, a simple, effective, and efficient graph transformer (GT) model specifically designed for fraud detection in financial transaction graphs. FraudGT leverages edge-based message passing gates and an edge attribute-based attention bias to enhance its ability to discern important transactional features and differentiate between normal and fraudulent transactions. Our model achieves state-of-the-art performance in detecting fraudulent activities while demonstrating high throughput and significantly lower latency compared to existing methods. We validate the effectiveness of FraudGT through extensive experiments on multiple large-scale synthetic financial datasets. FraudGT consistently outperforms other models, achieving 7.8–17.8% higher F1 scores, while delivering an average of 2 _._ 4× greater throughput and reduced latency. Our code and datasets are available at https://github.com/junhongmit/FraudGT. 

Financial transaction networks, fraud detection, graph transformers, graph neural networks, graph learning 

#### **ACM Reference Format:** 

Junhong Lin, Xiaojie Guo, Yada Zhu, Samuel Mitchell, Erik Altman, and Julian Shun. 2024. FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection. In _5th ACM International Conference on AI in Finance (ICAIF ’24), November 14–17, 2024, Brooklyn, NY, USA._ ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3677052.3698648 

### **1 Introduction** 

Fraud detection is a critical task in the financial industry, encompassing various applications such as anti-money laundering [9, 32, 58], malicious account/commodity detection (e.g., in online payment systems [10, 68] and e-commerce systems [7, 8, 42]), and spam detection [50, 57]. Financial fraud can lead to significant financial losses, reputation damage, and regulatory penalties for financial institutions. Traditional fraud detection methods, which rely heavily on rule-based systems and manual audits [28], are often inadequate due to the evolving nature of fraud schemes and the sheer volume of transactions that need to be monitored. Rule-based systems also suffer from low accuracy. 

Recent advances in machine learning, particularly in graph neural networks (GNNs), have shown promise in enabling effective and efficient fraud detection. GNNs are well-suited for learning from graph-structured data and have been successfully applied in various important domains such as biology [31], chemistry [67], and recommendation systems [59]. Their ability to capture complex structures in data makes them an ideal choice for financial fraud detection, where transactional relationships, as shown in Figure 1(a), can be represented as graphs, as illustrated in Figure 1(b). Here each node represents a financial account, and each edge represents a financial transaction between two accounts. Financial transaction graphs are often directed multigraphs, where edges (or transactions) have a direction, and there can be multiple edges between two nodes (or accounts). By using message passing to propagate information between connected nodes [20], GNNs can learn representations that capture the graph structure and compute the fraud likelihood for each transaction or account based on these learned representations. 

### **CCS Concepts** 

• **Mathematics of computing** → **Graph algorithms** ; • **Computing methodologies** → **Neural networks** ; • **Information systems** → **Data mining** ; • **Security and privacy** → **Intrusion/anomaly detection and malware mitigation** ; • **Applied computing** → **Business process monitoring** . 



This work is licensed under a Creative Commons Attribution International 4.0 License. 

_ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA_ © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1081-0/24/11 https://doi.org/10.1145/3677052.3698648 

However, traditional GNNs face the following challenges in effectively handling the unique characteristics of financial graphs. 

292 



<!-- Start of picture text -->
Trans Timestamp Src Src Dst Dst Amount | Currency Payment<br>ID. Bank | Account | Bank | Account Type<br>(a) Transactions in Tabular Format<br>- 8 QT<br>:afta tite<br>’v7 $ $ 019 N N c¢ - V N<br>I\ ——< ByuaJ § \ (1)Fan-Out (2) Fan-In<br>XY a 7 2 ’ £9 i, ‘<br>SS = $ [e ea ‘ = \<br>ys $ S- ! $1<br>Pra ~sN y 1<br>1 ‘ o N , A\ / niII (3)Gather-Scatter (4)Gather Scatter-<br>I\ z_ yrsot! 1d T ee vA /<br>\ N say ’ a y<br>NS aa z . ¢<br>ooo (5) Bipartite (6) Stack<br>(b) Transactions in Graph Format (c) Additional Fraud Patterns<br><!-- End of picture text -->

FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection 

ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA 

include event-based models that analyze individual transactions for anomalies [12, 34] and sequence-based models which detect fraud by identifying suspicious patterns over time [29, 41]. The second line of work utilizes GNNs to capture complex relationships in financial transaction graphs. GNN models have shown to be effective in many fraudulent scenarios, including anti-money laundering [9, 32, 58], credit card fraud [18, 60, 62], malicious account detection [17, 42], and fake review manipulation [57]. 

_Graph Neural Networks (GNNs)._ GNNs have become a cornerstone in graph-based learning tasks due to their ability to model complex relationships within graph-structured data. Traditional GNNs rely on a message-passing mechanism [20], which collects information from neighbors and combines it with the node’s own features to update the representation of a node . GNNs are divided into two categories: spectral GNNs and spatial GNNs. 

Spectral GNNs apply graph convolution operations in the spectral domain. ChebNet [16] approximates graph convolution using polynomial expansion. GCN [38] performs spectral convolutions on graphs to capture structure and feature information. 

Spatial GNNs apply the convolution operation on the graph structure by leveraging the information of neighborhood nodes. GraphSAGE [24] proposes a general inductive framework that can efficiently update the representation of sampled nodes. GAT [56] leverages a self-attention mechanism to enable distinct treatment of various neighbors during the embedding update of a node. 

Despite their success, common GNNs exhibit several limitations. GNNs with limited expressivity [1, 40, 51] may struggle to learn intricate patterns within a graph, which are essential for tasks like financial fraud detection. Common GNNs also inadequately utilize or neglect rich information contained in edge attributes [21], primarily focusing on node features. This limitation hinders their effectiveness on tasks where edge information is crucial, such as financial transaction graphs. 

_Graph Transformers (GTs)._ GTs [39, 45, 46, 46, 63, 63, 66] extend the transformative capabilities of conventional transformer architectures, which have made significant strides in both natural language processing [35, 55, 66] and computer vision [25, 47, 65]. By using powerful attention mechanisms, transformers overcome limitations in traditional message-passing GNNs [24, 38], such as over-smoothing and over-squashing [51]. However, many of these models overlook the critical role of edge features in financial transaction graphs. Our work aims to bridge this gap by introducing edge-related components to enhance a GT’s ability to discern important transactional features indicative of fraudulent activities, addressing the shortcomings of existing GTs in the context of financial fraud detection. 

### **3 Proposed Method** 

In this section, we first present preliminaries on graph representation of financial transaction networks and graph transformers. Then, we introduce the architecture and methodology of FraudGT. 

a set of _𝑛_ nodes representing accounts and E ⊆V ×V corresponds to a set of edges representing transactions between these accounts and there can be multiple edges between two nodes. If the graph is node-attributed or edge-attributed, the node attribute matrix _𝑿_ ∈ R<sup>_𝑛_×</sup><sup>_𝑑𝑛_</sup> assigns attributes to each node, and the edge attribute tensor **E** ∈ R<sup>_𝑛_×</sup><sup>_𝑛_×</sup><sup>_𝑑𝑒_</sup> assigns attributes to each edge. _𝑑𝑛_ and _𝑑𝑒_ are the dimensions of node and edge attributes, respectively. We use **E** _𝑖𝑗_ ∈ R<sup>_𝑑𝑒_</sup> to denote the attribute of the edge that connects node _𝑣𝑖_ and _𝑣 𝑗_ , and **E** _𝑖𝑗_ = 0 means there is no edge between _𝑣𝑖_ and _𝑣 𝑗_ . 

_Graph Transformers._ The goal of graph transformers is to learn a node representation that captures graph structure, based on featurebased proximities between different positions in the input node feature matrix. The learned representation is then used in downstream tasks, such as computing the fraud likelihood of an account or transaction. A graph transformer is a stack of _𝐿_ layers with blocks of multi-head attention (MHA) modules and fully connected feedforward networks (FFN) in each layer. Let G be a graph with node feature matrix _𝑿_ = [ _𝒙_ 1 _, 𝒙_ 2 _, . . . , 𝒙𝑛_ ]<sup>_𝑇_</sup> ∈ R<sup>_𝑛_×</sup><sup>_𝑑𝑛_</sup> , where _𝒙𝑖_ ∈ R<sup>_𝑑𝑛_</sup> is the node feature of node _𝑣𝑖_ . In each layer _𝑙_ ( _𝑙 >_ 0), given the hidden feature matrix _𝑯_<sup>(</sup><sup>_𝑙_−1)</sup> ∈ R<sup>_𝑛_×</sup><sup>_𝑑𝑛_</sup> , where _𝑯_<sup>(0)</sup> = _𝑿_ , the MHA module first linearly projects the input _𝑯_<sup>(</sup><sup>_𝑙_−1)</sup> to the query, key, and value spaces. This is computed using the following equations, where the projection using weight matrices _𝑾𝑄_<sup>(</sup><sup>_ℎ,𝑙_)</sup> _, 𝑾𝐾_<sup>(</sup><sup>_ℎ,𝑙_)</sup> , and _𝑾𝑉_<sup>(</sup><sup>_ℎ,𝑙_)</sup> ∈ R<sup>_𝑑𝑛_×</sup><sup>_𝑑ℎ_</sup> results in the matrices _𝑸_<sup>(</sup><sup>_ℎ,𝑙_)</sup> , _𝑲_<sup>(</sup><sup>_ℎ,𝑙_)</sup> , and _𝑽_<sup>(</sup><sup>_ℎ,𝑙_)</sup> , representing the query, key, and value spaces, respectively: 



Then, multiple attention heads are used to compute the scaled dotproduct, as shown in Equation (2), where the softmax function is applied row-wise, _𝑾𝑂_<sup>(</sup><sup>_𝑙_</sup> _𝑛_<sup>)∈R</sup><sup>_𝑑𝑛_×</sup><sup>_𝑑𝑛_is a learnable weight matrix,</sup><sup>_𝑑ℎ_</sup> denotes the feature dimension of the matrices _𝑸_<sup>_ℎ,𝑙_</sup> and _𝑲_<sup>_ℎ,𝑙_</sup> , _ℎ_ = 1 to _𝐻_ denotes the index of different attention heads, and ∥ denotes the concatenation operator. 



The multi-head attention module MHA( _𝑯_<sup>(</sup><sup>_𝑙_−1)</sup> ) concatenates several attention heads together. By combining the result with additional residual connections and normalization, the transformer layer updates features _𝑯_<sup>(</sup><sup>_𝑙_−1)</sup> as follows: 



where _𝜎_ refers to the activation function, and _𝑾_ 1<sup>(</sup><sup>_𝑙_)</sup> ∈ R<sup>_𝑑𝑛_×</sup><sup>_𝑑𝑓_</sup> and _𝑾_ 2<sup>(</sup><sup>_𝑙_)</sup> ∈ R<sup>_𝑑𝑓_×</sup><sup>_𝑑𝑛_</sup> are trainable parameters in the feedforward network (FFN) layer. The final output _𝑯_<sup>(</sup><sup>_𝐿_)</sup> ∈ R<sup>_𝑛_×</sup><sup>_𝑑𝑛_</sup> can be used as the updated node representation for downstream tasks. 

### **3.2 FraudGT** 

### **3.1 Preliminaries** 

_Graph Representation._ A financial transaction network can be represented as a directed multigraph G = (V _,_ E _, 𝑿,_ **E** ), where V is 

To handle financial transaction graphs, we introduce a novel graph transformer model, FraudGT, which consists of new components compared to existing graph transformers, including an edge-based 

294 



<!-- Start of picture text -->
i : ReversepReveneMessgeH iM | ils7Bi \o ° =)5 | FraudGT Encoder Block x £ layers PadWE _ £(2)(1) Edge-based Graph Attention Messagei Passing Gate<br>i‘, on aad $ ()aN we. Residual Connections Residual Connections :  £(3) Edge-based Attention Bias<br>H Gi) Hi 2 iS Gi) QZ; F |" | X H heads: : (4) Directed Multigraph Enhancements<br>NG : ¢ Port Numbering NOL 2 : 3 Port<br>——@: ii secetccerneeeeeeeyO°’Ego IDs oO 2@||| a, | :eeeH Pepera : oma|| E a ee 8Z &Be3<br>H q | Feature G Sismoid, |: : {o. | O7>| FEN [>© [a<br>i;!H NeighborhoodSampled ;;: Va  &4 |antH NeighborhoodA Gees4 : rr PETES vnaaasSPrereerrerrererer ieee:: Residual Connections<br><!-- End of picture text -->

FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection 

ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA 

this issue by incorporating an edge-based attention bias. This bias helps the model pay more attention to suspicious transactions by adjusting the attention scores based on edge attributes. The at- _𝑇_ tention bias term _𝒃_<sup>(</sup><sup>_ℎ,𝑙_)</sup> = **E**<sup>′(</sup><sup>_𝑙_)</sup> _𝑾_<sup>(</sup><sup>_ℎ,𝑙_)</sup> ∈ R<sup>_𝑑ℎ_</sup> is element-wise _𝑖𝑗_ � _𝑖𝑗_ � _𝐸_ added to the original attention score as shown in Equation (8), with _𝑾𝐸_<sup>(</sup><sup>_ℎ,𝑙_)</sup> ∈ R<sup>_𝑑𝑒_×</sup><sup>_𝑑ℎ_</sup> being a learnable weight vector and **E** _𝑖𝑗_<sup>′(</sup><sup>_𝑙_)</sup> being the feature of edge between _𝑣𝑖_ and _𝑣 𝑗_ . This adjustment ensures that transactions with attributes that are more indicative of fraud receive higher attention scores. 

_3.2.4 Directed Multigraph Enhancements._ Financial transaction graphs are often modeled as directed multigraphs since multiple transactions can exist between accounts. Therefore, components that are beneficial for directed multigraph learning can potentially improve fraud detection performance. Egressy et al. [19] presents a set of enhancements—reverse message passing, port numbering, and ego ID—that perform feature augmentations to improve the expressivity of GNNs in directed multigraphs. We incorporate combinations of these enhancements into FraudGT to obtain different variants. Based on our comprehensive experiments, we observe that some of the variants can improve fraud detection accuracy while maintaining efficiency. The enhancements are described as follows: 

- **Reverse Message Passing (RMP).** Standard GNNs only pass messages in the direction of edges in directed graphs. Nodes without incoming edges receive no messages from their neighbors and, therefore, cannot leverage the graph structure to refine their node features. To overcome this issue, bidirectional message passing is provided to allow communication in both directions. The message direction is indicated in the edge feature to let the model distinguish between incoming and outgoing edges, as illustrated in Figure 2. 

- **Port Numbering.** Multiple transactions can exist between two accounts. Distinguishing edges from the same neighbor and edges from different neighbors can help the model detect more complicated fraudulent patterns. Port numbering [49] serves this purpose by assigning local IDs to each neighbor at a node (Figure 2). We assign each directed edge an incoming and outgoing port number, and edges coming from (or going to) the same node, receive the same incoming (or outgoing) port number. 

- **Ego IDs.** Although RMP and port numbering help with detecting more suspicious patterns, they are not sufficient for detecting directed cycles. Ego IDs [64] were introduced to help detect cycles in graphs by marking a center node with a distinct (binary) feature so that it can be recognized when a sequence of messages cycles back around to it. We adopt this idea in our framework. 

_3.2.5 Training and Prediction._ In this paper, we aim to predict the anomaly score of each edge (transaction). The final classifier is comprised of a simple feed-forward network MLP(·) and a sigmoid function _𝜎_ . Let ∥ denote vector concatenation. The prediction _𝑦_ ˆ _𝑖𝑗_ for an edge between nodes _𝑣𝑖_ and _𝑣 𝑗_ is 



FraudGT is trained using a supervised learning approach. The objective is to minimize the binary cross-entropy loss between the predicted probabilities and the true labels of the transactions. Let 

**Table 1: Statistics of datasets used in the experiments.** 

|Data|set|# Nodes|# Edges|Illicit Rate|Time Span|Split [%]|
|---|---|---|---|---|---|---|
|AML|Small-HI|515,088|5,078,345|0.102%|10 days|64/19/17|
|AML|Small-LI|705,907|6,924,049|0.051%|10 days|64/19/17|
|AML|Medium-HI|2,077,023|31,898,238|0.110%|16 days|61/17/22|
|AML|Medium-LI|2,032,095|31,251,483|0.051%|16 days|61/17/22|
|AML|Large-HI|2,116,168|179,702,229|0.124%|97 days|60/20/20|
|AML|Large-LI|2,070,980|176,066,557|0.057%|97 days|60/20/20|



_𝑦𝑖𝑗_ be the true label of the transaction between nodes _𝑣𝑖_ and _𝑣 𝑗_ , and _𝑦_ ˆ _𝑖𝑗_ be the predicted anomaly score of the transaction. The loss function L is defined as 



_3.2.6 Inference Computational Complexity._ For a given batch of transactions, FraudGT extracts a subgraph from the sampled neighborhood and performs _𝐿_ layers attention calculations across all edges. The edge-based message passing gate and attention bias are computed for every edge. Therefore, the complexity of attention computation is linear to the number of edges in the batch, which we denote as |E _batch_ |. Each attention calculation is linear in _𝑑𝑛_ and _𝑑𝑒_ , the number of hidden dimensions in the node embeddings and edge embeddings, respectively. 

RMP and port numbering are feature augmentations that are only performed once during preprocessing and have linear time complexity in the total number of edges. Ego ID adds a binary feature to each node per sampled subgraph and has time complexity linear in the total number of nodes |V _batch_ | in the sampled subgraph. Therefore, the overall computational complexity of FraudGT on a batch of transactions is _𝑂_ (|E _batch_ | _𝐿𝑑_ + |V _batch_ |), where _𝑑_ = max( _𝑑𝑛,𝑑𝑒_ ). 

### **4 Experiments** 

In this section, we present comprehensive experiments evaluating FraudGT. We show that FraudGT matches or outperforms all other methods in fraud detection accuracy, while also being faster. 

### **4.1 Experimental Setup** 

_4.1.1 Datasets._ Given the strict privacy regulations around financial data, real-world datasets are not readily available. While some commercial datasets exist, they are not publicly available [26, 48, 52]. Individual banks and institutions often only have access to their own transaction data, missing the broader context of customer behavior across multiple institutions. Furthermore, these datasets typically suffer from poor labeling, as many money laundering activities go undetected [53, 54], especially when they involve transactions across different banks. As a result, creating ground truth labels is particularly challenging in this domain. Instead, we use large-scale simulated money laundering data [3]. The simulator behind these datasets generates realistic financial transaction graphs by modeling agents (banks, companies, and individuals) in a virtual world. The generator uses well-established laundering patterns to add realistic money laundering (illicit) transactions. We use all three sizes of the synthetic datasets that are publicly available on Kaggle 

296 

ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA 

Lin et al. 

**Table 2: Benchmark results of F1 scores (%) of various GNN methods. Standard deviations are calculated over 5 runs with different random seeds. We highlight the first and second best results.** 

|||Average Rank|AML Small-HI|AML Small-LI|AML Medium-HI|AML Medium-LI|AML Large-HI|AML Large-LI|
|---|---|---|---|---|---|---|---|---|
||MLP|21.2|0.42 ±0.04|0.13 ±0.05|0.06 ±0.12|0.15 ±0.02|0.66 ±0.45|0.36 ±0.03|
||LightGBM+GFs [5]|11.3|62.86 ±0.25|20.83 ±1.50|59.48 ±0.15|20.85 ±0.38|58.03 ±0.19|23.67 ±0.11|
||XGBoost+GFs[5]|8.8|63.23 ±0.17|27.30 ±0.33|65.70 ±0.26|28.16 ±0.14|42.68 ±12.93|24.23 ±0.12|
||GatedGCN [6]|18.0|38.54 ±2.25|17.18 ±4.02|41.61 ±4.57|11.90 ±3.98|36.96 ±1.88|8.97 ±7.61|
|s|GAT [56]|21.3|0.28 ±0.15|0.13 ±0.01|0.36 ±0.07|0.12 ±0.01|0.94 ±0.15|0.35 ±0.09|
|NN|GIN [30, 61]|16.5|40.04 ±5.40|23.26 ±3.56|45.40 ±6.15|12.19 ±3.01|36.32 ±2.55|6.06 ±4.73|
|i-G|GIN+RMP [33]|14.2|45.03 ±7.02|18.80 ±2.55|53.26 ±4.82|11.74 ±2.00|59.29 ±3.22|10.88 ±4.95|
|ult|GIN+Ports [49]|18.2|54.83 ±2.08|18.70 ±1.08|41.96 ±1.77|11.39 ±5.11|40.15 ±1.38|0.20 ±0.28|
|-M|GIN+Ego ID [64]|15.2|46.03 ±4.38|18.21 ±3.67|52.84 ±5.94|21.82 ±2.13|53.31 ±4.12|5.42 ±5.58|
|non|GIN+EU [4]|15.7|44.97 ±5.41|23.14 ±9.90|53.13 ±7.89|17.96 ±2.84|41.99 ±2.24|4.88 ±3.88|
||PNA [15]|12.7|61.06 ±5.65|17.98 ±3.69|60.17 ±2.39|31.66 ±0.79|54.69 ±6.66|4.10 ±5.18|
||PNA+EU|10.8|57.23 ±5.61|26.43 ±1.83|61.02 ±2.02|26.51 ±3.31|61.98 ±5.71|3.96 ±6.68|
|Ns|Multi-GIN [19]|12.5|47.42 ±2.93|22.31 ±5.79|54.59 ±2.25|18.72 ±4.65|58.43 ±5.09|17.53 ±7.44|
|GN|Multi-GIN+EU [19]|10.8|57.12 ±2.86|16.23 ±3.23|62.25 ±2.05|22.58 ±2.40|61.50 ±2.23|25.35 ±1.43|
|lti-|Multi-PNA [19]|9.2|68.19 ±2.03|31.33 ±2.58|67.22 ±2.65|26.33 ±2.90|51.85 ±11.16|6.59 ±8.60|
|Mu|Multi-PNA+EU[19]|10.7|68.60 ±3.36|27.79 ±3.63|63.60 ±1.58|17.95 ±5.83|59.02 ±4.63|4.65 ±5.80|
||FraudGT|9.0|69.68 ±1.58|28.69 ±2.05|62.38 ±0.87|24.02 ±0.52|54.35 ±1.65|11.02 ±2.65|
|nts|+RMP|6.3|64.84 ±2.00|33.02 ±3.17|66.37 ±0.47|27.01 ±2.61|65.05 ±1.19|19.17 ±3.22|
|aria|+Ports|3.7|74.90 ±0.55|44.17 ±1.87|72.12 ±1.18|38.62 ±2.85|60.89 ±2.50|30.40 ±5.68|
|d V|+Ego ID|3.8|70.01 ±3.47|34.22 ±1.10|71.72 ±1.29|32.59 ±1.79|65.48 ±0.91|27.94 ±4.77|
|pose|+Ports+Ego ID<br>(PE-FraudGT)|1.8|**76.41** ±**1.45**|45.81 ±1.14|74.22 ±1.74|43.53 ±1.76|68.64 ±2.31|30.44 ±2.76|
|Pro|+RMP+Ports+Ego ID<br>(Multi-FraudGT)|1.2|76.13 ± 0.95|**47.01** ±**2.22**|**75.93** ±**1.92**|**44.06** ±**5.27**|**73.34** ±**1.64**|**37.43** ±**4.94**|



[2], and for each size, we use one dataset with a higher illicit ratio (HI) and one with a lower illicit ratio (LI). The dataset sizes, illicit ratios, and split ratios among the training, validation, and test sets are provided in Table 1. The datasets are split temporally, i.e., we split the transactions after ordering them by their timestamps. 

_4.1.2 Baselines._ We compare FraudGT with three categories of baselines representing the state-of-the-art (SOTA) work in financial fraud detection. The first category consists of computationally efficient baselines, including MLP [22], which performs classification directly on node and edge features, and LightGBM+GFs and XGBoost+GFs [5], which are gradient-boosting methods using precalculated graph-based features (GFs) and tree-based classifiers LightGBM [36] and XGBoost [11] to classify nodes or edges individually. This approach has produced SOTA results in financial applications [43, 58]. 

The second category consists of GNN models with edge features but without directed multigraphs enhancements, which we term non-Multi-GNNs. They include GatedGCN [6], GAT [56], GIN [30], GIN with reverse message passing (+RMP) [33], GIN with ports numbering (+Ports) [49], GIN with ego ID (+Ego ID) [64], GIN with edge updates (+EU) [4], PNA [15], and PNA with edge updates (+EU). 

The third class of baselines is the SOTA GNN models tailored for directed multigraphs [19]: Multi-GINE, Multi-GINE with edge updates (+EU), Multi-PNA, and Multi-PNA with edge updates (+EU). 

Lastly, in addition to FraudGT alone, we incorporate a combination of the directed multigraph enhancements described in 

Section 3.2.4—RMP, port numbering, and ego IDs. We obtain the following variants: FraudGT with RMP (+RMP), FraudGT with port numbering (+Ports), FraudGT with ego ID (+Ego ID), FraudGT with port numbering and ego ID (+Ports+Ego ID or PE-FraudGT), and FraudGT with all three enhancements (+RMP+Ports+Ego ID or Multi-FraudGT). We use neighborhood sampling [24] for all GNN-based models. 

_4.1.3 Evaluation._ Since our datasets are very imbalanced, popular metrics for measuring accuracy are not suitable. Instead, we use the F1 score, consistent with previous works [3, 19] and aligns well with what banks and regulators use in real-world scenarios. Test performance is reported for the learned parameters of the highest validation performance. We used a POWER9 processor on the IBM Power System AC922 (8335-GTG) running at 2.3–3.8GHz frequency with a 10MB L3 cache size to perform graph sampling. We used an Nvidia V100 GPU with 32GB of memory to perform training and inference. 

### **4.2 Experimental Results** 

_4.2.1 Classification Results._ Table 2 lists the results of each method across the datasets, with standard deviations calculated over 5 runs with different random seeds. We make the following observations. First, the proposed FraudGT and its variants demonstrate leading performance on all AML datasets, as indicated by their average rankings. Notably, the vanilla FraudGT is competitive with or outperforms non-Multi-GNN baselines, such as GatedGCN and PNA, on AML Small-HI, AML Small-LI, AML Medium-HI, and AML Large-LI. For example, on the AML Small-HI dataset, FraudGT 

297 



<!-- Start of picture text -->
=m Multi-GIN Multi-PNA+EU mmm Multi-PNA FraudGT+Ports<br>LightGBM+GFs mmm XGBoost+GFs jm FraudGT+RMP ™@mm_ FraudGT+Ports+Ego ID (PE-FraudGT)<br>mm Multi-GINE+EU FraudGT ™@mm FraudGT+Ego ID jm FraudGT+RMP+Ports+Ego ID (Multi-FraudGT)<br>F1 [%] Throughput [trans/s] Per-batch latency [ms]<br>N Lol e N w + u a Lol N Wsu<br>lo} o oso ao aoo i=}o lo}Kr fo}KF OoKR Sogr oOKF oOKR lo}KR uwoO uo uuuOo 000<br>z arte) 96 Toca Seo8<br>ans 57.1268.60 ESz 120.28126.87 zs 761101<br>3 mm09,0868.19 un3 19.1023.54 418.3 un3 = 87107<br>2= 464.8470.01 | 2= 13.87 48.29 a= lm 42 148<br>x= 74.5076.41) | <== 47.69]47.44, == |e 4343<br>76.13 13.78 149<br>z mbes oo Snog<br>= 16.23 > iH 26.28 > 78<br>o 427.79 KS H19.01 = H108<br>3 3831.33 09 ow3 zie!20.01 147.17. w3 14 102,naa<br>2 33.02 2 13.11 > 156<br>=c 34.2244.1745.81 =c H44.4747.3448.00)| =© jog 464343<br>47,01 13.44 152<br>>= 54.59159.4862.25 >= 110.4514.0815.52 Ba3 134(1471196<br>o= H62.38Hee 50 o= mae: 34.48 =o 159 seal 97<br>a 67.22 2 14.70 2 139<br>§=7 aii-72H71.72.1274.22) §=<7 = 32.2633.7334.67. §=7 #635961 es<br>75.93 9.09 225<br>>= 20.8518.7222.58 >= 110.5015.9917.11 3b= 1211129i195<br>®=z |e 417.95 24.0228.16 =z®= 10.19113.15 H33.90 -z2 #60 1156201<br>2© |mammH127.0126.33 a< 19.2313.56 2< 151222<br>3i £438.6232.59 37 132.7534.86 37 Ho359<br>c 43.53 c 34.28 c 60<br>44.06 9.29 221<br>z see? ma 19 (168 iti<br>cr 8500 z 16.68 2 #298 HBS<br>5 H5442 . 3568 cr5 13.7522.81 cr5 H90 49<br>°== 160.8965.48H68.64: ®x= : H23.8322.5722.87 =x°= H8691.90<br>73.34 6:20 B30<br>> ee ete 19 1168 p53<br>s 25.35 > jmays.93 z B46<br>. 4.65 = 6.21 << 31<br>ia® Hilp38 ie1727.94 a®- coe. baie 23.4825 12 arc® eet87 ioeB<br>c 430.40 c 23.87 c 86<br>30.44 24.76 83<br>37.43 6:39 B21<br><!-- End of picture text -->

ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA 

Lin et al. 

attention bias (FraudGT w/o Edge-based Attention Bias) performs better than the model without the edge-based message passing gate, but still falls short of FraudGT with both components, indicating that the edge-based attention also contributes significantly to the model’s performance. These findings demonstrate the importance of both the edge-based message passing gate and edge-based attention in achieving state-of-the-art performance in financial fraud detection. 

### **5 Conclusion** 

In this paper, we introduced FraudGT, a simple, effective, and efficient graph transformer model designed for financial fraud detection in transaction graphs. FraudGT addresses several key challenges inherent in financial transaction graphs, including learning complex patterns, effective use of edge information, and computational efficiency. Leveraging the strengths of GTs to capture complex patterns and relationships within financial transaction data, FraudGT incorporates an edge-based message passing gate and an edge-based attention bias, allowing the model to focus on critical transactional features indicative of fraudulent activities. Through extensive evaluation on various publicly-available large-scale synthetic datasets, we show that FraudGT significantly outperforms existing baselines and achieves state-of-the-art performance. While synthetic datasets provide an essential testing ground due to privacy concerns in real-world financial data, for future work, it would be valuable to evaluate FraudGT on real-world financial fraud datasets. This would provide further validation of FraudGT’s applicability in real-world settings and enhance its potential for deployment in practical financial fraud detection systems. 

### **Acknowledgments** 

This work is funded by the MIT-IBM AI Watson Lab. In addition, Julian Shun is supported by NSF awards #CCF-1845763, #CCF2316235, and #CCF-2403237. 

### **References** 

- [1] Uri Alon and Eran Yahav. 2020. On the Bottleneck of Graph Neural Networks and Its Practical Implications. In _International Conference on Learning Representations (ICLR)_ . 

- [2] Erik Altman. 2019. IBM Transactions for Anti-Money Laundering (AML). https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-foranti-money-laundering-aml. 

- [3] Erik Altman, Jovan Blanuša, Luc Von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. 2024. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. _Advances in Neural Information Processing Systems (NeurIPS)_ 36 (2024). 

- [4] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. 2018. Relational Inductive Biases, Deep Learning, and Graph Networks. _arXiv preprint arXiv:1806.01261_ (2018). 

- [5] Jovan Blanuša, Maximo Cravero Baraja, Andreea Anghel, Luc von Niederhäusern, Erik Altman, Haris Pozidis, and Kubilay Atasu. 2024. Graph Feature Preprocessor: Real-Time Extraction of Subgraph-Based Features from Transaction Graphs. _arXiv preprint arXiv:2402.08593_ (2024). 

- [6] Xavier Bresson and Thomas Laurent. 2017. Residual Gated Graph ConvNets. _arXiv preprint arXiv:1711.07553_ (2017). 

- [7] Bokai Cao, Mia Mao, Siim Viidu, and S Yu Philip. 2017. HitFraud: A Broad Learning Approach for Collective Fraud Detection in Heterogeneous Information Networks. In _IEEE International Conference on Data Mining (ICDM)_ . 769–774. 

- [8] Shaosheng Cao, XinXing Yang, Cen Chen, Jun Zhou, Xiaolong Li, and Yuan Qi. 2019. TitAnt: Online Real-time Transaction Fraud Detection in Ant Financial. _Proceedings of the VLDB Endowment (PVLDB)_ 12, 12 (2019). 

- [9] Mário Cardoso, Pedro Saleiro, and Pedro Bizarro. 2022. LaundroGraph: SelfSupervised Graph Representation Learning for Anti-Money Laundering. In _Proceedings of the ACM International Conference on AI in Finance_ . 130–138. 

- [10] Liang Chen, Jiaying Peng, Yang Liu, Jintang Li, Fenfang Xie, and Zibin Zheng. 2020. Phishing Scams Detection in Ethereum Transaction Network. _ACM Transactions on Internet Technology (TOIT)_ 21, 1 (2020), 1–16. 

- [11] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In _Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)_ . 785–794. 

- [12] Tianyi Chen and Charalampos Tsourakakis. 2022. AntiBenford Subgraphs:Unsupervised Anomaly Detection in Financial Networks. In _Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)_ . 2762–2770. 

- [13] Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. 2020. Can Graph Neural Networks Count Substructures? _Advances in Neural Information Processing Systems (NeurIPS)_ 33 (2020), 10383–10395. 

- [14] Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. 2019. On the Equivalence Between Graph Isomorphism Testing and Function Approximation with GNNs. _Advances in Neural Information Processing Systems (NeurIPS)_ 32 (2019). 

- [15] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veličković. 2020. Principal Neighbourhood Aggregation for Graph Nets. _Advances in Neural Information Processing Systems (NeurIPS)_ 33 (2020), 13260–13271. 

- [16] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. 2016. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. _Advances in Neural Information Processing Systems (NeurIPS)_ 29 (2016). 

- [17] Zhihao Ding, Jieming Shi, Qing Li, and Jiannong Cao. 2023. Effective Multi-Graph Neural Networks for Illicit Account Detection on Cryptocurrency Transaction Networks. _arXiv preprint arXiv:2309.02460_ (2023). 

- [18] Yifan Duan, Guibin Zhang, Shilong Wang, Xiaojiang Peng, Ziqi Wang, Junyuan Mao, Hao Wu, Xinke Jiang, and Kun Wang. 2024. CaT-GNN: Enhancing Credit Card Fraud Detection via Causal Temporal Graph Neural Networks. _arXiv preprint arXiv:2402.14708_ (2024). 

- [19] Béni Egressy, Luc Von Niederhäusern, Jovan Blanuša, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. 2024. Provably Powerful Graph Neural Networks for Directed Multigraphs. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , Vol. 38. 11838–11846. 

- [20] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017. Neural Message Passing for Quantum Chemistry. In _International Conference on Machine Learning (ICML)_ . 1263–1272. 

- [21] Liyu Gong and Qiang Cheng. 2019. Exploiting Edge Features for Graph Neural Networks. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ . 9211–9219. 

- [22] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. _Deep Learning_ . MIT Press. 

- [23] Oscar M Granados and Andrés Vargas. 2022. The Geometry of Suspicious Money Laundering Activities in Financial Networks. _EPJ Data Science_ 11, 1 (2022), 6. 

- [24] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. _Advances in Neural Information Processing Systems (NeurIPS)_ 30 (2017). 

- [25] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. 2022. A Survey on Vision Transformer. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 45, 1 (2022), 87–110. 

- [26] Daniel A Harris, Kyla L Pyndiura, Shelby L Sturrock, and Rebecca AG Christensen. 2022. Using real-world transaction data to identify money laundering: Leveraging traditional regression and machine learning techniques. _STEM Fellowship Journal_ 7, 1 (2022), 21–32. 

- [27] Jing He, Jiao Tian, Yuanyuan Wu, Xinyi Cia, Kai Zhang, Mengjiao Guo, Hui Zheng, Junfeng Wu, and Yimu Ji. 2021. An Efficient Solution to Detect Common Topologies in Money Launderings Based on Coupling and Connection. _IEEE Intelligent Systems_ 36, 1 (2021), 64–74. 

- [28] Waleed Hilal, S Andrew Gadsden, and John Yawney. 2022. Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances. _Expert Systems with Applications (ESWA)_ 193 (2022), 116429. 

- [29] Bryan Hooi, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. 2016. FRAUDAR: Bounding Graph Fraud in the Face of Camouflage. In _Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)_ . 895–904. 

- [30] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. 2020. Strategies for Pre-training Graph Neural Networks. In _International Conference on Learning Representations (ICLR)_ . 

- [31] Kexin Huang, Cao Xiao, Lucas M Glass, Marinka Zitnik, and Jimeng Sun. 2020. SkipGNN: Predicting Molecular Interactions with Skip-Graph Networks. _Scientific Reports_ 10, 1 (2020), 1–16. 

- [32] Woochang Hyun, Jaehong Lee, and Bongwon Suh. 2023. Anti-Money Laundering in Cryptocurrency via Multi-Relational Graph Neural Network. In _Advances in Knowledge Discovery and Data Mining: Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)_ . 118–130. 

299 

FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection 

ICAIF ’24, November 14–17, 2024, Brooklyn, NY, USA 

- [33] Guillaume Jaume, An-phi Nguyen, María Rodríguez Martínez, Jean-Philippe Thiran, and Maria Gabrani. 2019. EDGNN: A Simple and Powerful GNN for Directed Labeled Graphs. _arXiv preprint arXiv:1904.08745_ (2019). 

- [34] Jiaxin Jiang, Yuan Li, Bryan Hooi, Bingsheng He, Jia Chen, and Johan Kok Zhi Kang. 2024. Spade: A Real-Time Fraud Detection Framework on Evolving Graphs. _Proceedings of the VLDB Endowment (PVLDB)_ 16 (2024), 461–474. 

- [35] Katikapalli Subramanyam Kalyan, Ajit Rajasekharan, and Sivanesan Sangeetha. 2021. AMMUS: A Survey of Transformer-Based Pretrained Models in Natural Language Processing. _arXiv preprint arXiv:2108.05542_ abs/2108.05542 (2021). 

- [36] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. _Advances in Neural Information Processing Systems (NeurIPS)_ 30 (2017). 

- [37] Jinwoo Kim, Saeyoon Oh, and Seunghoon Hong. 2021. Transformers Generalize DeepSets and Can Be Extended to Graphs & Hypergraphs. _Advances in Neural Information Processing Systems (NeurIPS)_ 34 (2021), 28016–28028. 

- [38] Thomas N Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In _International Conference on Learning Representations (ICLR)_ . 

- [39] Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C Bayan Bruss, and Tom Goldstein. 2023. GOAT: A Global Transformer on Large-Scale Graphs. In _Proceedings of the International Conference on Machine Learning (ICML)_ . 17375– 17390. 

- [40] Qimai Li, Zhichao Han, and Xiao-Ming Wu. 2018. Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ . 

- [41] Shenghua Liu, Bryan Hooi, and Christos Faloutsos. 2017. HoloScope: Topologyand-Spike Aware Fraud Detection. In _Proceedings of the ACM on Conference on Information and Knowledge Management (CIKM)_ . 1539–1548. 

- [42] Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, and Le Song. 2018. Heterogeneous Graph Neural Networks for Malicious Account Detection. In _Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM)_ . 2077–2085. 

- [43] Wai Weng Lo, Gayan K Kulatilleke, Mohanad Sarhan, Siamak Layeghy, and Marius Portmann. 2023. Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering Detection in Bitcoin. _Applied Intelligence_ 53, 16 (2023), 19406–19417. 

- [44] Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. 2019. Invariant and Equivariant Graph Networks. In _International Conference on Learning Representations (ICLR)_ . 

- [45] Erxue Min, Runfa Chen, Yatao Bian, Tingyang Xu, Kangfei Zhao, Wenbing Huang, Peilin Zhao, Junzhou Huang, Sophia Ananiadou, and Yu Rong. 2022. Transformer for Graphs: An Overview from Architecture Perspective. _arXiv preprint arXiv:2202.08455_ (2022). 

- [46] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. 2022. Recipe for a General, Powerful, Scalable Graph Transformer. _Advances in Neural Information Processing Systems (NeurIPS)_ 35 (2022), 14501–14515. 

- [47] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. 2021. Vision Transformers for Dense Prediction. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ . 12179–12188. 

      - You Need. _Advances in Neural Information Processing Systems (NeurIPS)_ 30 (2017). 

   - [56] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In _International Conference on Learning Representations (ICLR)_ . 

   - [57] Jianyu Wang, Rui Wen, Chunming Wu, Yu Huang, and Jian Xiong. 2019. FDGARS: Fraudster Detection via Graph Convolutional Networks in Online App Review System. In _Proceedings of the International Conference on World Wide Web (WWW)_ . 310–316. 

   - [58] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom Robinson, and Charles E Leiserson. 2019. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. _arXiv preprint arXiv:1908.02591_ (2019). 

   - [59] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph Neural Networks in Recommender Systems: A Survey. _ACM Computing Surveys (CSUR)_ 55, 5 (2022), 1–37. 

   - [60] Sheng Xiang, Mingzhi Zhu, Dawei Cheng, Enxia Li, Ruihui Zhao, Yi Ouyang, Ling Chen, and Yefeng Zheng. 2023. Semi-Supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ . 14557–14565. 

   - [61] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019. How Powerful are Graph Neural Networks?. In _International Conference on Learning Representations (ICLR)_ . 

   - [62] Kuan Yan, Junbin Gao, and Dmytro Matsypura. 2023. FIW-GNN: A Heterogeneous Graph-Based Learning Model for Credit Card Fraud Detection. In _IEEE International Conference on Data Science and Advanced Analytics (DSAA)_ . 1–10. 

   - [63] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. 2021. Do Transformers Really Perform Bad for Graph Representation?. In _Advances in Neural Information Processing Systems (NeurIPS)_ . 28877–28888. 

   - [64] Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec. 2021. Identity-Aware Graph Neural Networks. In _Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)_ , Vol. 35. 10737–10745. 

   - [65] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. 2022. Scaling Vision Transformers. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ . 12104–12113. 

   - [66] Shuaicheng Zhang, Qiang Ning, and Lifu Huang. 2022. Extracting Temporal Event Relation with Syntax-Guided Graph Transformer. In _North American Chapter of the Association for Computational Linguistics (NAACL)_ . 379–390. 

   - [67] Tianyi Zhao, Yang Hu, Linda R Valsdottir, Tianyi Zang, and Jiajie Peng. 2021. Identifying Drug–Target Interactions Based on Graph Convolutional Network and Deep Neural Network. _Briefings in Bioinformatics_ 22, 2 (2021), 2141–2150. 

   - [68] Qiwei Zhong, Yang Liu, Xiang Ao, Binbin Hu, Jinghua Feng, Jiayu Tang, and Qing He. 2020. Financial Defaulter Detection on Online Credit Payment via Multi-View Attributed Heterogeneous Information Network. In _Proceedings of the International Conference on World Wide Web (WWW)_ . 785–795. 

- [48] Susie Xi Rao, Shuai Zhang, Zhichao Han, Zitao Zhang, Wei Min, Zhiyao Chen, Yinan Shan, Yang Zhao, and Ce Zhang. 2021. xFraud: Explainable Fraud Transaction Detection. _Proceedings of the VLDB Endowment (PVLDB)_ 15, 3 (2021), 427–436. 

- [49] Ryoma Sato, Makoto Yamada, and Hisashi Kashima. 2019. Approximation Ratios of Graph Neural Networks for Combinatorial Problems. _Advances in Neural Information Processing Systems (NeurIPS)_ 32 (2019). 

- [50] Kai Shu, Deepak Mahudeswaran, Suhang Wang, and Huan Liu. 2020. Hierarchical Propagation Networks for Fake News Detection: Investigation and Exploitation. In _Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)_ , Vol. 14. 626–637. 

- [51] Yunchong Song, Chenghu Zhou, Xinbing Wang, and Zhouhan Lin. 2023. Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-Smoothing. In _International Conference on Learning Representations (ICLR)_ . 

- [52] Michele Starnini, Charalampos E Tsourakakis, Maryam Zamanipour, André Panisson, Walter Allasia, Marco Fornasiero, Laura Li Puma, Valeria Ricci, Silvia Ronchiadin, Angela Ugrinoska, et al. 2021. Smurf-Based Anti-Money Laundering in Time-Evolving Transaction Networks. In _European Conference on Machine Learning and Data Mining (ECML PKDD)_ . 171–186. 

- [53] United Nations Office on Drugs and Crime. 2022. Money Laundering. https: //www.unodc.org/unodc/en/money-laundering/overview.html Accessed: 202410-05. 

- [54] U.S. Department of the Treasury. 2022. National Money Laundering Risk Assessment. https://home.treasury.gov/system/files/136/2022-National-MoneyLaundering-Risk-Assessment.pdf 21 pages. 

- [55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All 

300 

