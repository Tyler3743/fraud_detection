# **LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection** 

Stan Verlaan Utrecht University Utrecht, Netherlands stan.verlaan@proton.me 

Ioana Hulpus, Erik Jan van Leeuwen Utrecht University Utrecht University Utrecht, Netherlands Utrecht, Netherlands i.r.karnstedt-hulpus@uu.nl e.j.vanleeuwen@uu.nl 



## **Abstract** 

We enhance Graph Neural Networks (GNNs) for identifying suspicious accounts involved in money laundering patterns. Extending the work of Egressy et al. (AAAI 2024), we propose a novel GNN architecture to detect suspicious subgraph motifs in the weighted temporal networks underlying financial data. Our architecture allows for the indication of edge directionality within a single Aggregator function, element-wise edge weight multiplication, and an LSTM aggregator that can learn from the sequential order of edges imposed by timestamps. The resulting model, LAS-GNN, is based on an inductive learning framework and can generalize across different networks. Experimental results on synthetic networks show that LAS-GNN is robust and can identify basic money laundering motifs to near perfection, outperforming a graph isomorphism network benchmark with edge features. 

**Figure 1: An example of a financial network [1], where the edges include a date and amount of money. Suspicious transactions that together form a cycle are highlighted in red.** 



## **CCS Concepts** 

• **Computing methodologies** → **Supervised learning** ; **Neural networks** . 

## **Keywords** 

anti-money laundering, financial networks, temporal motif detection, graph neural networks 

### **ACM Reference Format:** 

Stan Verlaan, Ioana Hulpus, , and Erik Jan van Leeuwen. 2025. LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection. In _6th ACM International Conference on AI in Finance (ICAIF ’25), November 15–18, 2025, Singapore, Singapore._ ACM, New York, NY, USA, 9 pages. https: //doi.org/10.1145/3768292.3770410 

## **1 Introduction** 

Money laundering (ML) is the process of concealing the origin of illegitimately obtained money. Given its association with criminal activities, the detection of ML processes is of paramount importance. Significant progress on machine learning methods for detecting ML activity has been made in recent years (see e.g. the surveys [9, 27, 28]), particularly utilizing the underlying network structure of transaction data [26]. A financial network can be seen as a directed graph, where edges represent transactions between accounts (vertices) and edges can have attributes such as time and value of the transaction [6, 17]. The presence of certain _motifs_ or 

This work is licensed under a Creative Commons Attribution 4.0 International License. _ICAIF ’25, Singapore, Singapore_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2220-2/25/11 https://doi.org/10.1145/3768292.3770410 

**Figure 2: Six (out of eight) money laundering motifs of [43].** **_Top row:_ fan-out, fan-in, gather-scatter.** **_Bottow row:_ scattergather, simple cycle, and bipartite clique. Nodes marked in red can be seen as key accounts and are to be detected.** 

patterns, for example a directed cycle (see Figure 1), can potentially be an indicator of money laundering [16, 19, 42, 46]. Suzumura and Kanezashi [43] introduced a set of common money laundering motifs in financial networks (see Figure 2). For example, the scattergather motif is associated with smurfing [42], where a criminal seeks to hide the illicit source of money by splitting it among many accounts, then gathering it before reintroducing it in the economy. Arguably, in ML activities, the chronology of the involved transactions also bears some relevance. Thus, understanding ML activities allows the problem of detecting money laundering processes to be modeled as finding motifs in directed temporal graphs. 

Many approaches (machine learning-based and otherwise) exist for finding motifs in directed temporal graphs (see e.g. [4, 5, 17, 24, 30, 33, 39, 49, 54]). We focus on the potential capabilities of Graph Neural Networks (GNNs) [50] for this problem. GNNs learn node representations by taking both the graph structure and the node and edge features into account. Intuitively, they propagate 

256 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

Stan Verlaan, Ioana Hulpus, , and Erik Jan van Leeuwen 

node information along the edges of the graph to update node representations. Combined with multiple layers and non-linear activation functions, this can capture complex relations and patterns in the data and makes GNNs well suited for finding motifs [8, 11, 52]. However, as far as we are aware, no previous GNN-based approach for money laundering motif detection was designed to account for the temporal data in financial transactions. 

In recent years, many GNNs that take temporal data into account have been proposed (see the overviews [14, 34, 41, 51]). In contrast to the typical requirements of temporal GNN models, however, the problem of temporal motif detection does not need to compute (and recover) for each node a representation at every time step (or time window). Rather, we aim to represent the nodes based on the chronology of the transactions. The hypothesis is that such chronology-aware node representations are able to identify _keynodes_ that participate in temporal motifs. 

We also remark that existing temporal GNN models do not account holistically for edge directions. For example, TGN [37], a GNN framework that generalizes many existing temporal GNNs, learns two different functions for aggregating messages corresponding to incoming and outgoing edges respectively. We show in this paper that such approaches to edge directionality fail to capture the interplay between edge chronology and edge directions. 

Therefore, we propose a novel GNN framework for finding temporal and weighted motifs in directed graphs. We build on the approach of Egressy et al. [11], notably their use of ego IDs [53]. To train our model to detect temporal motifs, we employ a long short-term-memory (LSTM) [20] aggregator that learns from the sequential order of edges imposed by timestamps. Moreover, we propose a novel message passing scheme for directed graphs, termed _signed message passing_ , that allows for the aggregation of messages from incoming and outgoing neighbours simultaneously. 

We evaluate LAS-GNN on synthetically generated networks. Our experiments isolate the effect of each model component: directed message passing, edge weight representation, and the LSTM aggregator. The results show that the components work synergistically, achieving almost perfect detection of the targeted temporal motifs. We also compare its performance at detecting temporal motifs to the state-of-the-art GNN architecture of Egressy et al. [11]. 

## **2 Related Work** 

One can identify two main and complementary directions for detecting money laundering activity in financial networks: (i) data-driven approaches where the model is trained directly on real financial data [7, 10, 31], and (ii) expert-driven pattern matching approaches where the models are specialized for detecting domain expert defined patterns [12, 19, 32, 43]. Our work falls in the latter category. 

We cast money laundering detection as a motif detection problem. In this direction, methods such as Flowscope [32] are designed to target one motif. GNN models, however, can be trained to detect arbitrary graph motifs, albeit not without challenges. 

First, off-the-shelf GNNs are aces at learning node representations but fail at detecting subgraph motifs. Recently, You et al. [53] proposed and Egressy et al. [11] perfected an elegant approach for detecting motifs that are centered on a node: they set the so-called _ego ID_ feature of a node, “the ego”, to 1, and to 0 for all nodes in its 

neighbourhood. When ego IDs are paired with a message passing model that allows messages to travel the edges in both directions, intuitively the diffusion of the “ego’s” representation through “ego’s” neighbourhood captures the graph structure around the “ego”. 

Second, the typical GNN approach to edge directionality is what we call _directional message passing_ , where messages strictly follow the direction of the edges. However, in financial networks, outgoing transactions are just as relevant to an account’s information as incoming transactions. Simply taking the underlying undirected graph ( _bidirectional message passing_ [23]) does not differentiate between the sender and the receiver. The most common alternative is to define separate Aggregator functions for the incoming and outgoing edges, respectively [11, 37, 38]. To emphasize that this method implicitly defines two types of relations in the graph, we name it _heterogeneous message passing_ . Egressy et al. [11] show that ego IDs and heterogeneous message passing taken together transform any MPNN into a _universal_ GNN, that can distinguish any two non-isomorphic (sub-)graphs, while not mistakenly distinguishing any two isomorphic (sub-)graphs. Nevertheless, as we show in this paper, heterogeneous message passing cannot capture the relation between edge directions and edge timestamps. 

Lastly, ML activities oftentimes involve multiple accounts that synchronize their actions in time, but this chronological aspect has not been addressed so far in the GNN motif detection literature. 

As our work proposes a GNN architecture that we train to detect temporal motifs, this work is also related to the domain of temporal GNNs. The problem of temporal graph representation learning takes as input an initial graph and a series of temporally labeled events such as node interactions, node birth, etc. A core aim of these models is to provide for a node _𝑣_ , its representation corresponding to a certain time _𝑡_ [48, 55]. As such, this problem is significantly different from the problem we address: temporal motif detection. We do not analyze the network as it evolves over time; we process a static version of the network with timestamped edges. Also, we are not interested in maintaining a memory of nodes’ states over time; rather, we care to obtain precisely one representation for a node, based on the chronology of the transactions in its subgraph. 

In our solution to temporal motif detection, we use a long shortterm memory (LSTM) cell for aggregating messages from neighbouring nodes. The idea of using an LSTM aggregator in GNNs is not new. Hamilton et al. [18] were the first to propose it. However, a critical concern in GNN research is to enforce permutation invariance: a node’s embeddings should be independent from its neighbourhood ordering. This requirement mostly constrains the choice of Aggregator function. Hamilton et al. exploited the expressivity of an LSTM aggregator, but ensured permutation invariance by repeated neighbour sampling and order randomization. 

In contrast, in our work we actually exploit the permutation sensitivity of an LSTM aggregator, because we need it to learn from the temporal sequence of transactions. From this perspective, our approach is among the very few [22] permutation sensitive GNN models in literature. Lastly, we note that most temporal GNNs such as TGN [37] (which generalizes many others), use some form of recurrent neural network (usually RNN, GRU, or LSTM) in their architecture. However, this typically occurs in the Update function to smooth the transition between the hidden layers representations of nodes, or in the Memory function to model the sequence of a 

257 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection 





<!-- Start of picture text -->
(a) Temporal motif occurrences.<br>(b) Non-temporal motif occurrences.<br><!-- End of picture text -->

**Figure 3: Examples of temporal and non-temporal motif occurrences in temporal networks. From right to left: gatherscatter, scatter-gather, directed cycle. The numbers on the edges correspond to timestamps.** 

node’s states over time. Rossi et al. [37] mention the possibility of an RNN aggregator, without further elaboration. 

## **3 Method: LAS-GNN** 

## **3.1 Weighted and Temporal Motifs** 

Given are a (directed) graph _𝐺_ = ( _𝑉, 𝐸_ ) and, for each edge ( _𝑢, 𝑣_ ) ∈ _𝐸_ , a weight _𝑤_ ( _𝑢, 𝑣_ ) ∈[0 _,_ 1] and an integer timestamp time( _𝑢, 𝑣_ ). 

_Definition 3.1._ A (directed) graph _𝐻_ occurs as a _motif_ in _𝐺_ if a subgraph _𝐻_<sup>′</sup> of _𝐺_ is isomorphic to _𝐻_ ; that is, there is a bijection _𝜙_ : _𝑉_ ( _𝐻_<sup>′</sup> ) → _𝑉_ ( _𝐻_ ) such that ( _𝑢, 𝑣_ ) ∈ _𝐸_ ( _𝐻_<sup>′</sup> ) if and only if ( _𝜙_ ( _𝑢_ ) _,𝜙_ ( _𝑣_ )) ∈ _𝐸_ ( _𝐻_ ). We call _𝜙_ the _isomorphism function_ . 

We now extend Definition 3.1 to the weighted setting. Edge weights can be interpreted in two ways. First, we can view them as monetary values. Second, we can view them as a money-laundering risk indicator per transaction, set by an external algorithm, e.g. using measures of node relatedness that indicate the strength of the connection. For both cases, having a consistent lower bound on the weight of all edges in the transaction is well motivated; in particular, for the first case, many anti-money laundering directives contain reporting thresholds (see e.g. [13]). Hence, we define: 

_Definition 3.2._ Given _𝑟_ ∈[0 _,_ 1], a (directed) graph _𝐻_ occurs as a _weighted motif_ in _𝐺_ if, for a subgraph _𝐻_<sup>′</sup> of _𝐺_ isomorphic to _𝐻_ , it holds that _𝑤_ ( _𝑢, 𝑣_ ) ≥ _𝑟_ for all ( _𝑢, 𝑣_ ) ∈ _𝐻_<sup>′</sup> . 

We now extend Definition 3.1 to the temporal setting. Recall the following well-known graph-theoretic notion. A _feedback node set_ (or feedback vertex set) of a (directed) graph _𝐻_ is a set _𝐹_ ⊆ _𝑉_ ( _𝐻_ ) such that _𝐻_ − _𝐹_ has no (directed) cycle. It is _minimal_ if no proper subset of _𝐹_ is also a feedback node set. 

_Definition 3.3._ A (directed) graph _𝐻_ with minimal feedback node set _𝐹_ occurs as a _temporal motif_ in _𝐺_ if there is a subgraph _𝐻_<sup>′</sup> of _𝐺_ isomorphic to _𝐻_ with isomorphism function _𝜙_ such that for each _𝑣_ ∈ { _𝑥_ ∈ _𝑉_ ( _𝐻_<sup>′</sup> ) | _𝜙_ ( _𝑥_ ) ∉ _𝐹_ }, the inequality max _𝑢_ ∈ _𝑁_ in _𝐻_<sup>′(</sup><sup>_𝑣_){time(</sup><sup>_𝑢, 𝑣_)}</sup><sup>_<_</sup> min _𝑤_ ∈ _𝑁_ out _𝐻_<sup>′(</sup><sup>_𝑣_){time(</sup><sup>_𝑣,𝑤_)} holds.</sup> 

Informally, the definition states that for every node in the motif, except those in _𝐹_ , the time of all incoming edges in the motif is lower than the time of any outgoing edge in the motif. This generalizes existing definitions of temporal motifs (see e.g. [17, 36, 39]), and allows to consider arbitrary motif graphs. We remark that we do not enforce a time window constraint [36]; rather, we emphasize the chronological aspect of the motif. 

Figure 3 illustrates our temporal motif definition. The first two motifs are acyclic, and thus we can use an empty feedback node set; hence, all nodes must respect the time constraint in an occurrence. In the third motif, a cycle, any node can constitute the feedback node set; thus, any one node “is allowed” to violate the constraint. In our example, this is the red node. The red nodes in Figure 3a are the key-nodes we train our model to detect. Our goal is to find temporal motif occurrences such as those illustrated in Figure 3 in a given graph _𝐺_ that occur simultaneously in the weighted (Definition 3.2) and temporal (Definition 3.3) sense. We are approaching the problem as a supervised classification problem, such that the model is trained on a graph that contains these motifs and motif key-nodes are labeled as the positive class, while all other nodes are labeled as the zero class. We are addressing the inductive setting, in which the model must generalize and be able to classify nodes belonging to the targeted motifs in new graphs. 

## **3.2 Model architecture** 

We propose a novel GNN architecture. Like Egressy et al. [11], our model has ego IDs [53] at its core, but it deviates substantially in its handling of edge directions and time. We describe our key ideas. 

First, we propose a single Aggregator function to handle edge directions. We do this by indicating edge direction by weighting the embeddings over the two directions differently within the Aggregator function. As far as we are aware, this approach has not yet appeared in the literature. We call this _signed message passing_ : 





Aggregate<sup>(</sup><sup>_𝑘_)</sup> is any Aggregator function for layer _𝑘_ . _Linear_ is a linear layer that transforms an input scalar into a vector the size of the node embeddings, and ⊙ represents element-wise multiplication. Instead of multiplying every component of _𝒉𝑢_<sup>(</sup><sup>_𝑘_−1)</sup> with the scalar 1 or −1, the model can learn a separate weight factor for each component. Thus, it can determine how and which parts of the embedding to differentiate for the two directions. 

Next, in the weighted version of LAS-GNN, we incorporate weights in this approach by replacing _Linear_ (1) in Equation 2 by the weight _Linear_ ( _𝑤_ ( _𝑢, 𝑣_ )) for _𝑣_ ’s incoming neighbours _𝑢_ , and _Linear_ (−1) by _Linear_ (− _𝑤_ ( _𝑣,𝑢_ )) for _𝑣_ ’s outgoing neighbours _𝑢_ . As a result, the message transferred over that edge is multiplied by the linear representation of the corresponding weight. This way, the weight determines the importance of the message to its recipient. 

We argue that the integration of edge weights and signed message passing is particularly intuitive in financial networks, where negative weights correspond to negative transactions in the opposite direction. This edge weight multiplication is an alternative to 

258 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

Stan Verlaan, Ioana Hulpus, , and Erik Jan van Leeuwen 





<!-- Start of picture text -->
GNN layer k<br>GNN  GNN  GNN<br>layer 1 layer ... layer K<br>Batch<br>ReLU Mean<br>Norm<br>N<br>N<br>Labels<br>N<br>N<br>Linear<br>N<br>64<br>64<br>1 64<br>output<br>Softmax<br>Classification<br><!-- End of picture text -->

**Figure 4: The architecture of LAS-GNN. Left: The global architecture. Right: A single GNN layer, shown with LSTM aggregation.** 

more standard approaches that treat the weight as an edge feature. It can also serve as a domain expert-defined attention coefficient [44]. 

Finally, we use _Long Short-Term Memory_ (LSTM) [20] to capture temporal information. LSTM was proposed by Hamilton et al. [18] as a possible Aggregate function, but applied to a random permutation of the node’s neighbours. Instead, we aim to capture the network dynamics and aggregate messages from neighbours in increasing order of timestamp corresponding to the edge. More formally, we define the ordered sequence of neighbours of node _𝑣_ as _𝑁_<sup>_𝑇_</sup> ( _𝑣_ ) = ⟨ _𝑢_ 1 _, . . . ,𝑢𝑛_ ⟩, where _𝑢𝑖_ ∈ _𝑁_ ( _𝑣_ ) is the _𝑖_ th neighbour of _𝑣_ in the sequence, such that time( _𝑢𝑖, 𝑣_ ) _<_ time( _𝑢 𝑗 , 𝑣_ ) for all _𝑖 < 𝑗_ . Then, _𝒂𝑣_<sup>(</sup><sup>_𝑘_)</sup> = LSTM<sup>(</sup><sup>_𝑘_) ��</sup> _𝒉𝑢_<sup>(</sup><sup>_𝑘_−1)</sup> | _𝑢_ ∈ _𝑁_<sup>_𝑇_</sup> ( _𝑣_ )�<sup>�</sup> . Note that this ordering is agnostic to edge direction. 

We employ one LSTM for each layer _𝑘_ . This allows the model to learn from different sequential orderings at each depth. The LSTMs we employ consist of a single layer of the same size as the GNN layers. The weights of the LSTMs are updated simultaneously with the other weights of the GNN during backpropagation. 

We refer to our ultimate GNN model, which includes ego IDs, signed message passing, edge weight multiplication, and LSTM aggregation, as **L** STM-based **A** ggregation and **S** igned message passing GNN (LAS-GNN) (see Figure 4). Formally, 



We mention some implementation details. Note that ego IDs fail when node embeddings are calculated for all nodes simultaneously, since the GNN cannot differentiate between nodes that all have ego ID 1. By conducting _mini-batch training_ , we only consider the embedding generation for a small group of seed nodes (i.e. a batch) at a time. For each batch, a small number of nodes are randomly sampled, their neighbourhoods are extracted in a breadth-first manner, and message passing is conducted in the corresponding subgraphs. If the sampled neighbourhoods of these seed nodes do not overlap, this issue with ego IDs does not arise. Thus, we ensure disjoint sampling of neighbourhoods, leading to disjoint subgraphs. 

After _𝐾_ rounds of message passing, the final node embeddings are transformed by a linear layer, and then normalized by a softmax activation function to obtain the classification probabilities. 

Regarding training, we apply the Adam optimizer [25] along with a standard binary cross-entropy loss function. To address data 

imbalance, we used the ratio of positive class labels out of the total number of samples as a weight in the loss calculation. 

## **4 Experimental Setup** 

In our experiments, we focus on measuring the effectivity of the three new ingredients to our GNN framework: signed message passing, weighted message passing, and LSTM aggregation. We apply our method on simple directed graphs (we do allow bidirected edges) in temporal and weighted settings. In our networks (described below), we aim to find several of the motifs of Suzumura and Kanezashi [43]. We focus on the gather-scatter (GS), scattergather (SG), and directed cycle motifs in the temporal setting; their key-nodes are marked in red in Figure 3. We denote by C _ℓ_ the directed cycle with _ℓ_ vertices. Initially, we only consider C3. In the cycle, we mark an arbitrary node of the cycle as key-node (this is our minimal feedback node set). In our preliminary experiments in Section 5, we also consider fan-in (FI), fan-out (FO), and bipartite clique (BC). We call all these motifs our _target motifs_ . 

## **4.1 Datasets** 

For the benefit of a controlled experiment, we work with artificial, generated network. Specifically, we create two datasets as follows. 

_WSM Dataset._ We generate artificial networks using the _WattsStrogatz model_ (WSM) [45]. This yields graphs with the target motifs occurring naturally, rather than being injected into the network post hoc, which risks skewing the random distribution and creating bias that affects detection algorithms. Since WSM is undirected by default, we enhance it to create directed graphs by directing edges randomly. Our WSM then has four parameters: total number of nodes _𝑛_ , average degree _𝑘_ , probability of rewiring an edge _𝑝_ rewire, and probability of edge reciprocation _𝑝_ recip (thus (1 − _𝑝_ recip)/2 are the probabilities of each direction separately). After preliminary experiments to check motif occurrence, we set _𝑛_ = 10 000, _𝑘_ = 6, _𝑝_ rewire = 0 _._ 4, _𝑝_ recip = 0 _._ 1. We generate separate WSM networks for training, validation, and testing (all with the same parameter values). This ensures there is no information leakage while testing our inductive approach. 

_LFR Dataset._ We generate artificial networks using the _LFR model_ [29]. This, in contrast to WSM graphs [3], yields graphs with a powerlaw degree distribution and communities [29]. It has the following 

259 

LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

parameters: total number of nodes _𝑛_ , power-law exponent of the degree distribution _𝛾_ , power-law exponent of the community size distribution _𝛽_ , mixing parameter _𝜇_ (the fraction of edges a node shares with nodes outside its community), average degree _𝑘_ , minimum degree _𝑘_ min and maximum degree _𝑘_ max, and minimum community size _𝑠_ min and maximum community size _𝑠_ max. We set _𝑛_ = 100 000, _𝛾_ = 3, _𝛽_ = 1 _._ 5, _𝜇_ = 0 _._ 1, _𝑘_ = 6 (same as for WSM), _𝑘_ min = 0, _𝑘_ max = 20 (this yielded networks with degree 30), _𝑠_ min = 5, _𝑠_ max = 500. We generate separate LFR networks for training, validation, and testing (all with the same parameter values). 

For both WSM and LFR data, we generate times by assigning each edge a unique timestamp using a random permutation of {1 _, . . . ,𝑚_ }, where _𝑚_ is the number of edges in the graph. For the WSM data, we also generate the weight of each edge uniformly at random in [0 _,_ 1]. To ensure that sufficiently many target motifs occur as a weighted motif, we sample half of the naturally occurring motifs and assign weights uniformly at random in [ _𝑟,_ 1], where _𝑟_ is the threshold of weighted motif occurrence. Note that further occurrences may appear by chance. 

For the ground truth, we employ an exhaustive search and classify the key-node of a (weighted and/or temporal) target motif as suspicious (1), while all other nodes are marked non-suspicious (0). 

## **4.2 Compared Models** 

In order to understand the impact of each component that makes up LAS-GNN, we implement and test several versions of the framework. We compare them with a similarly diverse set of models based on a GIN architecture as proposed by Egressy et. al [12]. 

_Baseline._ For the unweighted, non-temporal setting, we implement a GNN model with ADD aggregator and vary the directed message passing mechanism and turn on/off the ego ID. These configurations are used in our preliminary experiments (Section 5). 

_LAS-GNN._ For the LAS-GNN algorithm, after a grid search on validation data, we settled on a learning rate of 0.001, an embedding size of 64, and a batch size of 32. 

In preliminary experiments, we found that four GNN layers suffice to detect all target motifs. This also appears intuitively to be the minimum number of layers required. For instance, in a C4 motif, the ego ID must traverse four edges to return to its original seed node. We also found that using a default element-wise sum aggregator can be advantageous in the first GNN layer. Hence, in our experiments, we use this default aggregator as the first layer, followed by three LSTM layers. 

We consider three variants of LAS-GNN. All use LSTM aggregation: _LAS-GNN_ is our ultimate model, bringing together signed message passing and temporally ordered LSTM aggregation. _LASGNN-Random_ uses a LSTM aggregator but orders the neighbours randomly instead of temporally, and uses signed message passing _LAS-GNN-HMP_ applies heterogeneous message passing (instead of signed message passing) and applies temporally sorted LSTM aggregation to incoming and outgoing neighbours separately. All LAS-GNN models are used with the same set of hyperparameters as specified above. 

_GIN+Ego+HMP._ As state-of-the-art, we compare our approach to the model of Egressy et al. [11], stripped of the port-numbering 

|**Model**|**ego ID**|**MP**|**Weight**|**Time**|**Agg**|
|---|---|---|---|---|---|
|GNN|✓|_any_|✗|✗|ADD|
|GIN+Ego+HMP|✓|hetero|✗|✗|ADD|
|GIN+Ego+HMP<sup>_𝑤𝑡_</sup>|✓|hetero|feature|feature|ADD|
|LAS-GNN|✓|signed|mult.|sorted|LSTM|
|- Random|✓|signed|✗|✗|LSTM|
|- HMP|✓|hetero|✗|sorted|LSTM|



**Table 1: Overview of the models used in our experiments.** 

component, since we work with simple graphs. Specifically, the model is a GIN architecture with edge features [21], ego IDs and heterogeneous message passing, and we call it _GIN+Ego+HMP_ . The ability of the model to represent edge features is important as this provides us with the means of seamlessly inputting the edge weights and timestamps as edge features. We refer to the weighted, temporal model as _GIN+Ego+HMP_<sup>_𝑤𝑡_</sup> . The timestamp feature is _𝐿_ 2 normalized since using absolute timestamp values results in worse performance. The MLP used is a two-layer neural network with 64 nodes in each layer. The _𝜖_ -parameter is set to 0. 

We refer to Table 1 for an overview of all compared models. 

For implementation we use PyTorch Geometric [15]. All experiments were run on a server running Ubuntu 22.04 LTS with Intel Xeon Gold 6238R CPU @ 2.20GHz, Nvidia A100 PCIe 40 GB GPU, and 256GB of RAM. We run each experiment for 100 epochs. We assume convergence and apply early stopping once the validation loss does not improve for 20 epochs. All reported results are averaged over five networks; each has a different seed. 

## **5 Preliminary Experiments** 

Before evaluating LAS-GNN on temporal, weighted networks, we report on our experiments conducted to confirm that ego IDs, signed-message passing, and the LSTM aggregator are working in line with state-of-the-art on unweighted, non-temporal networks. We perform these experiments on our WSM graphs. 

We first isolate the effectivity of the employed message passing approach and ego IDs (see Table 2). These experiments partially reproduce results of Egressy et al. [11], confirming that heterogeneous message passing combined with ego IDs to the node features (GNN-Hetero+ego ID) is essential to correctly identify the target motifs, offering a substantial improvement over bidirectional and directional message passing. The poor performance of bidirectional message-passing is explained by the fact that it does not account for directionality, while all the target motifs are directed. Importantly, we observe that our alternative approach of signed message passing seems to be very effective. Echoing the results for heterogeneous message passing, it achieves perfect detection results in directed graphs with reciprocating edges. 

Next, to ensure that using an LSTM does not harm detection results, we compare GIN+Ego+HMP to LAS-GNN-Random. From the perfect results in Table 2, we conclude that using LSTM aggregators instead of MLPs does not deteriorate motif detection capabilities, even when learning a sequential order is not required. 

260 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

Stan Verlaan, Ioana Hulpus, , and Erik Jan van Leeuwen 



<!-- Start of picture text -->
1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>LAS-GNN (AP = 1.00) LAS-GNN (AP = 0.97) LAS-GNN (AP = 1.00)<br>0.2 0.2 0.2<br>GIN+Ego+HMP wt  (AP = 0.86) GIN+Ego+HMP wt  (AP = 0.86) GIN+Ego+HMP wt  (AP = 0.86)<br>LAS-GNN-HMP (AP = 0.48) LAS-GNN-HMP (AP = 0.66) LAS-GNN-HMP (AP = 0.50)<br>0.0 LAS-GNN-Random (AP = 0.42) 0.0 LAS-GNN-Random (AP = 0.33) 0.0 LAS-GNN-Random (AP = 0.28)<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Recall Recall Recall<br>Precision Precision Precision<br><!-- End of picture text -->

**Figure 5: PR curves for temporal motif detection on WSM graphs with** 10 **K nodes. The filled area marks the standard deviation.** **_Left:_ GS motif, imb.** ≈ 9 _._ 0% **.** **_Middle:_ SG motif, imb.** ≈ 7 _._ 0% **.** **_Right:_ C3 motif, imb.** ≈ 9 _._ 8% **.** 



<!-- Start of picture text -->
1.0<br>1.0<br>Train LAS-GNN<br>Val LAS-GNN<br>0.8 0.8 Train GIN+Ego+HMP t<br>Val GIN+Ego+HMP t<br>0.6 0.6<br>0.4 0.4<br>LAS-GNN (AP = 0.99)<br>0.2 0.2<br>GIN+Ego+HMP t  (AP = 0.88)<br>LAS-GNN-HMP (AP = 0.72)<br>0.0 LAS-GNN-Random (AP = 0.49) 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0 20 40 60 80 100<br>Recall Epoch<br>1.0<br>1.0<br>Train LAS-GNN<br>Val LAS-GNN<br>0.8 0.8 Train GIN+Ego+HMP t<br>Val GIN+Ego+HMP t<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>LAS-GNN (AP = 1.00)<br>0.0 GIN+Ego+HMP t  (AP = 0.98) 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0 20 40 60 80 100<br>Recall Epoch<br>Loss<br>Precision<br>Loss<br>Precision<br><!-- End of picture text -->

**Figure 6:** **_Top Left:_ PR curves for unweighted temporal motif detection, simultaneously for GS, SG, and C3 motifs, on WSM graphs with** 10 000 **nodes. Imb.:** ≈ 23 _._ 5% **. The filled area marks the standard deviation.** **_Top right:_ Training and validation loss of a single run in this setting.** **_Bottom left:_ PR curves for unweighted temporal motif detection, simultaneously for GS, SG, and C3 motifs, on LFR graphs with** 100 000 **nodes. Imb.:** ≈ 16 _._ 0% **.** **_Bottom right:_ Training and validation loss of a single run in this setting.** 

## **6 Results** 

In this section, we evaluate LAS-GNN on the weighted and temporal motif detection tasks that it has been designed for. Our focus is on the motifs illustrated in Figure 3, specifically: GS, SG, and C3. 

_Unweighted, Temporal._ Figure 5 shows the results from our experiments on WSM graphs when combining the LSTM aggregator and signed message passing, as formalized in Equation 3. LAS-GNN can almost perfectly detect these temporal motifs. 

The failure of LAS-GNN-Random is inevitable: it does not use temporal information. Still, this shows that LSTM aggregation only works when the edge ordering is respected. LAS-GNN-HMP does make use of temporal information, but its lacking performance shows that its separate aggregation is incompatible with learning the sequential ordering of temporal motifs. GIN+Egp+HMP<sup>_𝑤𝑡_</sup> manages to learn from the temporal motif examples provided during training where timestamps are provided as edge features. However, 

261 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection 



<!-- Start of picture text -->
1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>LAS-GNN (AP = 1.00) LAS-GNN (AP = 0.98) LAS-GNN (AP = 0.99)<br>0.0 GIN+Ego+HMP wt  (AP = 0.95) 0.0 GIN+Ego+HMP wt  (AP = 0.83) 0.0 GIN+Ego+HMP wt  (AP = 0.86)<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>Recall Recall Recall<br>Precision Precision Precision<br><!-- End of picture text -->

**Figure 7: PR curves for weighted temporal motif detection on WSM graphs with** 100 000 **nodes. The filled area marks the standard deviation.** **_Left:_ GS motif, imb.** ≈ 4 _._ 5% **.** **_Middle:_ SG motif, imb.** ≈ 3 _._ 5% **.** **_Right:_ C3 motif, imb.** ≈ 5 _._ 1% **.** 

|**Model**|**FI**|**FO**|**GS**|**SG**|**BC**|**C3**|**C4**|
|---|---|---|---|---|---|---|---|
|GNN-Bidirect.|0_._64|0_._67|0_._73|0_._48|0_._47|0_._71|0_._64|
|+ego ID|0_._64|0_._65|0_._73|0_._49|0_._49|0_._73|0_._68|
|GNN-Direct.|1_._0|0_._63|0_._78|0_._57|0_._55|0_._67|0_._61|
|+ego ID|1_._0|0_._68|0_._80|0_._62|0_._58|1_._0|0_._75|
|GNN-Hetero.|1_._0|1_._0|1_._0|0_._63|0_._64|0_._85|0_._70|
|+ego ID|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|
|GNN-Signed|1_._0|1_._0|1_._0|0_._64|0_._63|0_._86|0_._71|
|+ego ID|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|
|GIN+Ego+HMP|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|
|LAS-GNN-Random|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|1_._0|
|_Imb._ (%)|_42.5_|_42.5_|_45.2_|_23.8_|_23.7_|_45.7_|_40.7_|



**Table 2: Average** _𝐹_ 1 **scores for the unweighted, non-temporal setting, on WSM graphs with** 10 000 **nodes, after 10 epochs. Standard deviations are all** ≤ 0 _._ 03 **. Perfect detection results are in bold.** 

|**Model**|**C3**|**C4**|**C5**|**C6**|
|---|---|---|---|---|
|GIN+Ego+HMP|0_._93±|_._006<br>0_._81±|_._126<br>0_._73±_._094|0_._49±_._055|
|LAS-GNN|0_._99±|_._002<br>0_._98±|_._001<br>0_._95±_._004|0_._87±_._010|



**Table 3: F1 scores for temporal motif detection of longer cycles on WSM graphs. This task is balanced: we sampled 2000 temporal and 2000 non-temporal cycle motifs at random.** 

besides achieving a lower average precision (0 _._ 86), the PR curve shows that performance can be inconsistent across multiple runs. 

It is important to observe that our GNN architecture is not specialized to detect a specific motif. To investigate this, we also consider networks where we mix the GS, SG, and C3 motifs. That is, we label all key nodes in any of these three motifs by 1. Figure 6 (top left) indeed shows that the LAS-GNN model achieves near perfect performance in this complex setting and can handle multiple motifs at the same time, in contrast to the other models. The comparison to LAS-GNN-HMP and LAS-GNN-Random confirms that it is the 

combination of signed message passing and LSTM in the Aggregator function of LAS-GNN that accounts for its performance. As alluded to earlier, Figure 6 (top right) shows that the training and validation loss of the GIN+Ego+HMP<sup>_𝑤𝑡_</sup> model suffers considerable instability and it takes considerable longer for it to converge. 

We also consider the performance when detecting longer cycles (up to length 6). We use _𝐾_ layers for cycles of length _𝐾_ . We use much larger networks: up to 4M nodes for C6. This leads to a very large imbalance; therefore, we report on the _balanced_ task, where we sampled 2000 suspicious and 2000 non-suspicious nodes from the graph at random. Table 3 shows that the average F1 score of both models decreases as the cycle length increases. It seems that having the timestamps as edge features is much less robust when detecting longer cycles than utilizing LSTM aggregation as done in LAS-GNN. This could mean that the latter is a more promising approach for larger subgraph motifs that require message passing at further depth. 

_Weighted, Temporal._ Finally, we add weights to our WSM graphs. We experimented with various weight threshold _𝑟_ values, and the performance sees a relative decay as _𝑟_ increases. In the interest of brevity, we report results obtained with a weight threshold _𝑟_ = 0 _._ 8. Recall that we suspiciously weigh around half of the occurring motifs. To ensure a fair task, we set the weights of an equal number of non-temporally ordered motifs to be suspicious. We train, validate, and test on networks with 100K nodes, using a batch size of 512, to accommodate the more severe class imbalance of this task. 

Figure 7 shows our results for the GS, SG, and C3 motifs. For the GS motif, both models perform almost equally well. Note that when we inspect the training and validation loss progression of a single run, GIN+Ego+HMP<sup>_𝑤𝑡_</sup> needs more training rounds: 40 versus just 15 epochs for LAS-GNN. Also, the validation loss of the GIN+Ego+HMP<sup>_𝑤𝑡_</sup> model exhibits considerable instability from that point onward. The F1-score is 1 _._ 000 for LAS-GNN and 0 _._ 908 for GIN+Ego+HMP<sup>_𝑤𝑡_</sup> . Similar behavior occurs for the C3 motif. The SG motif, however, seems more difficult to detect for both models. Still, the F1-score is 0 _._ 934 for LAS-GNN and 0 _._ 839 for GIN+Ego+HMP<sup>_𝑤𝑡_</sup> . 

## **7 Discussion** 

Compared to the GIN+Ego+HMP<sup>_𝑤𝑡_</sup> model of Egressy et al. [11], the LAS-GNN model offers clear advantages in temporal settings. 

262 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

Stan Verlaan, Ioana Hulpus, , and Erik Jan van Leeuwen 

Our results suggest that, even though all temporal information is available through the edge features, GIN+Ego+HMP<sup>_𝑤𝑡_</sup> struggles to learn from this information effectively. Note that GIN+Ego+HMP<sup>_𝑤𝑡_</sup> must independently determine that the timestamps represent a sequential order, whereas LAS-GNN explicitly models this in its aggregation mechanism. LAS-GNN also seems more robust than GIN+Ego+HMP<sup>_𝑤𝑡_</sup> when detecting larger cyclic motifs. 

It may also be interesting to compare LAS-GNN against further GNNs, such as done in the extensive benchmarking by Egressy et al. [11]. However, the GIN+Ego+HMP<sup>_𝑤𝑡_</sup> model of Egressy et al. [11] was shown to significantly outperform other available models in the non-temporal setting, making this the most relevant baseline. Similarly, while a comparison to temporal GNNs could be interesting, existing temporal GNNs do not fit our chronological setting, as far as we are aware, as already discussed in the introduction. 

The experiments we performed may not yet showcase the full potential of LAS-GNN. A limitation is that our experiments focused on simple directed graphs (although we allow bidirectional edges). Using port numbering [40], we may be able to extend our model to directed multigraphs and demonstrate the potential of LAS-GNN on the extended testbed of Egressy et al. [11]. In particular, they used a variant of the Watts-Strogatz model with parallel edges and directed multigraphs derived from the AMLworld simulator [1]. In future work, we aim to test LAS-GNN further on data sets based on AMLSim [43], AMLworld [1], and Elliptic [47]. 

In the same vein, a limitation of our use of the Watts-Strogatz model is that it produces a Poisson degree distribution [3], while real networks are often scale-free networks with a power-law degree distribution [2], enabling the presence of _hubs_ – nodes with high degree. In real financial networks, hubs often correspond to nonsuspicious companies [42], so this may not be a substantial issue. Still, we did perform some initial experiments on our LFR dataset. As shown in Figure 6 (bottom), LAS-GNN detects motifs very well for mild power-law exponents and small maximum degree. While our LFR dataset uses _𝛾_ = 3, we noticed the same excellent results down to _𝛾_ = 2 _._ 5, but ran out of memory for lower values of _𝛾_ . Moreover, performance seems to degrade as the exponent of the power-law decreases and hubs become more pronounced. When increasing the maximum degree to 100 (keeping _𝛾_ = 3), using four layers required too much memory. To detect C3, we require only three layers, but notice performance issues already in the non-temporal setting: while ADD aggregation works well, LSTM aggregation performs poorly and the model does not seem to learn. There is some evidence in the literature that all GNNs suffer from similar performance issues, although this can sometimes be mitigated [35, 56]. Moreover, we seem to run into issues with LSTM aggregation specifically. We hope to extend LAS-GNN to still perform well in these situations. 

## **8 Conclusion** 

We summarize our work: (i) we provided a new definition of temporal graph motifs that aligns with the flow of funds in money laundering; (ii) we proposed a directed GNN message passing mechanism that achieves state-of-the-art performance while being compatible with edge timestamps; (iii) we proposed a new GNN model for temporal motif finding that shows consistent superiority to straightforward adaptations of static state-of-the-art motif finding 

solutions to temporal graphs; (iv) our model is to the best of our knowledge the first to leverage the permutation sensitivity of an LSTM aggregator in GNNs to capture temporal interaction of nodes. 

## **References** 

- [1] Erik Altman, Jovan Blanuša, Luc Von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. 2023. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. In _Proc. NeurIPS 2023_ . 

- [2] Albert-László Barabási and Réka Albert. 1999. Emergence of Scaling in Random Networks. _Science_ 286 (1999), 509–512. Issue 5439. 

- [3] Alain Barrat and Martin Weigt. 2000. On the properties of small-world network models. _Eur. Phys. J. B_ 13 (2000), 547–560. 

- [4] Jovan Blanusa, Maximo Cravero Baraja, Andreea Anghel, Luc von Niederhäusern, Erik R. Altman, Haris Pozidis, and Kubilay Atasu. 2024. Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection. In _Proc. ICAIF 2024_ . ACM, 222–230. 

- [5] Jovan Blanusa, Paolo Ienne, and Kubilay Atasu. 2022. Scalable Fine-Grained Parallel Cycle Enumeration Algorithms. In _Proc. SPAA ’22_ . ACM, 247–258. 

- [6] Kevin Buehler. 2019. Transforming approaches to AML and financial crime. McKinsey. 

- [7] Z. Chai, Y. Yang, J. Dan, S. Tian, C. Meng, W. Wang, and Y. Sun. 2023. Towards Learning to Discover Money Laundering Sub-network in Massive Transaction Network. In _Proc. AAAI 2023_ . AAAI Press, 14153–14160. 

- [8] Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. 2020. Can Graph Neural Networks Count Substructures?. In _Proc. NeurIPS 2020_ . 

- [9] Zhiyuan Chen, Le Dinh Van Khoa, Ee Na Teoh, Amril Nazir, Ettikan Kandasamy Karuppiah, and Kim Sim Lam. 2018. Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: a review. _Knowl. Inf. Syst._ 57, 2 (2018), 245–285. 

- [10] Dawei Cheng, Yujia Ye, Sheng Xiang, Zhenwei Ma, Ying Zhang, and Changjun Jiang. 2023. Anti-Money Laundering by Group-Aware Deep Graph Learning. _IEEE Trans. Knowl. Data Eng._ 35, 12 (2023), 12444–12457. 

- [11] Béni Egressy, Luc Von Niederhäusern, Jovan Blanuša, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. 2024. Provably Powerful Graph Neural Networks for Directed Multigraphs. In _Proc. AAAI 2024_ . AAAI Press, 11838–11846. 

- [12] Béni Egressy and Roger Wattenhofer. 2022. Graph neural networks with precomputed node features. _arXiv:2206.00637_ (2022). 

- [13] European Parliament and Council of European Union. 2024. Regulation (EU) 2024/1624. http://data.europa.eu/eli/reg/2024/1624/oj 

- [14] ZhengZhao Feng, Rui Wang, TianXing Wang, Mingli Song, Sai Wu, and Shuibing He. 2024. A Comprehensive Survey of Dynamic Graph Neural Networks: Models, Frameworks, Benchmarks, Experiments and Challenges. _CoRR_ abs/2405.00476 (2024). 

- [15] Matthias Fey and Jan Eric Lenssen. 2019. Fast graph representation learning with PyTorch Geometric. _arXiv:1903.02428_ (2019). 

- [16] Oscar M Granados and Andrés Vargas. 2022. The geometry of suspicious money laundering activities in financial networks. _EPJ Data Science_ 11, 1 (2022), 6. 

- [17] László Hajdu and Miklós Krész. 2020. Temporal Network Analytics for Fraud Detection in the Banking Sector. In _Proc. ADBIS, TPDL and EDA 2020 Common Workshops and Doctoral Consortium (CCIS, Vol. 1260)_ . Springer, 145–157. 

- [18] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In _Proc. NeurIPS 2017_ . 1024–1034. 

- [19] Jing He, Jiao Tian, Yuanyuan Wu, Xinyi Cia, Kai Zhang, Mengjiao Guo, Hui Zheng, Junfeng Wu, and Yimu Ji. 2021. An efficient solution to detect common topologies in money launderings based on coupling and connection. _IEEE Intell. Syst._ 36, 1 (2021), 64–74. 

- [20] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. _Neural Comput._ 9, 8 (1997), 1735–1780. 

- [21] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. 2020. Strategies for pre-training graph neural networks. In _Proc. ICLR 2020_ . OpenReview.net. 

- [22] Zhongyu Huang, Yingheng Wang, Chaozhuo Li, and Huiguang He. 2022. Going deeper into permutation-sensitive graph neural networks. In _Proc. ICML 2022 (PMLR, Vol. 162)_ . 9377–9409. 

- [23] Guillaume Jaume, An-phi Nguyen, María Rodríguez Martínez, Jean-Philippe Thiran, and Maria Gabrani. 2019. edGNN: a Simple and Powerful GNN for Directed Labeled Graphs. _CoRR_ abs/1904.08745 (2019). 

- [24] Ali Jazayeri and Christopher C. Yang. 2020. Motif discovery algorithms in static and temporal networks: A survey. _J. Complex Networks_ 8, 4 (2020). 

- [25] Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In _Proc. ICLR 2015_ . 

- [26] Eren Kurshan, Hongda Shen, and Haojie Yu. 2020. Financial Crime & Fraud Detection Using Graph Computing: Application Considerations & Outlook. In _Proc. TransAI 2020_ . IEEE, 125–130. 

- [27] Dattatray Vishnu Kute, Biswajeet Pradhan, Nagesh Shukla, and Abdullah M. Alamri. 2021. Deep Learning and Explainable Artificial Intelligence Techniques 

263 

ICAIF ’25, November 15–18, 2025, Singapore, Singapore 

LAS-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection 

Applied for Detecting Money Laundering-A Critical Review. _IEEE Access_ 9 (2021), 82300–82317. 

- [28] Nevine Makram Labib, Mohammed Abo Rizka, and Amr Ehab Muhammed Shokry. 2020. Survey of machine learning approaches of anti-money laundering techniques to counter terrorism finance. In _Proc. ITAF 2019 (LNNS, Vol. 114)_ . 73–87. 

   - [55] Yanping Zheng, Lu Yi, and Zhewei Wei. 2025. A survey of dynamic graph neural networks. _Frontiers of Computer Science_ 19, 6 (2025), 196323. 

   - [56] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. 2020. Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs. In _Proc. NeurIPS 2020_ . 

- [29] Andreas Lancichinetti, Santo Fortunato, and Filippo Radicchi. 2008. Benchmark graphs for testing community detection algorithms. _Phys. Rev. E_ 78 (2008), 046110. 

- [30] Jiantao Li, Jianpeng Qi, Yueling Huang, Lei Cao, Yanwei Yu, and Junyu Dong. 2024. MoTTo: Scalable Motif Counting with Time-aware Topology Constraint for Large-scale Temporal Graphs. In _Proc. CIKM 2024_ . ACM, 1195–1204. 

- [31] Xujia Li, Yuan Li, Xueying Mo, Hebing Xiao, Yanyan Shen, and Lei Chen. 2023. Diga: Guided diffusion model for graph recovery in anti-money laundering. In _Proc. SIGKDD 2023_ . ACM, 4404–4413. 

- [32] Xiangfeng Li, Shenghua Liu, Zifeng Li, Xiaotian Han, Chuan Shi, Bryan Hooi, He Huang, and Xueqi Cheng. 2020. FlowScope: Spotting Money Laundering Based on Graphs. In _Proc. AAAI 2020_ . AAAI Press, 4731–4738. 

- [33] Yuchen Li, Zhengzhi Lou, Yu Shi, and Jiawei Han. 2018. Temporal Motifs in Heterogeneous Information Networks. In _Proc. MLG 2018_ . 

- [34] Antonio Longa, Veronica Lachi, Gabriele Santin, Monica Bianchini, Bruno Lepri, Pietro Lio, Franco Scarselli, and Andrea Passerini. 2023. Graph Neural Networks for Temporal Graphs: State of the Art, Open Challenges, and Opportunities. _Trans. Mach. Learn. Res._ (2023). 

- [35] Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. 2022. Is Homophily a Necessity for Graph Neural Networks?. In _Proc. ICLR 2022_ . OpenReview.net. 

- [36] Ashwin Paranjape, Austin R. Benson, and Jure Leskovec. 2017. Motifs in Temporal Networks. In _Proc. WSDM 2017_ . ACM, 601–610. 

- [37] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael M. Bronstein. 2020. Temporal Graph Networks for Deep Learning on Dynamic Graphs. _arXiv:2006.10637_ (2020). 

- [38] Emanuele Rossi, Bertrand Charpentier, Francesco Di Giovanni, Fabrizio Frasca, Stephan Günnemann, and Michael M Bronstein. 2023. Edge directionality improves learning on heterophilic graphs. In _Proc. LoG 2023 (PMLR, Vol. 231)_ . 25. 

- [39] Ahmet Erdem Sarıyüce. 2025. A powerful lens for temporal network analysis: temporal motifs. _Discover Data_ 3 (2025), 14. 

- [40] Ryoma Sato, Makoto Yamada, and Hisashi Kashima. 2019. Approximation ratios of graph neural networks for combinatorial problems. _Proc. NeurIPS 2019_ , 4083– 4092. 

- [41] Joakim Skarding, Bogdan Gabrys, and Katarzyna Musial. 2021. Foundations and Modeling of Dynamic Networks Using Dynamic Graph Neural Networks: A Survey. _IEEE Access_ 9 (2021), 79143–79168. 

- [42] Michele Starnini, Charalampos E Tsourakakis, Maryam Zamanipour, André Panisson, Walter Allasia, Marco Fornasiero, Laura Li Puma, Valeria Ricci, Silvia Ronchiadin, Angela Ugrinoska, et al. 2021. Smurf-based anti-money laundering in time-evolving transaction networks. In _Proc. ECML PKDD 2021 (LNCS, Vol. 12978)_ . Springer, 171–186. 

- [43] Toyotaro Suzumura and Hiroki Kanezashi. 2021. Anti-Money Laundering Datasets: InPlusLab Anti-Money Laundering DataDatasets. http://github.com/ IBM/AMLSim/. 

- [44] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In _Proc. ICLR 2018_ . OpenReview.net. 

- [45] Duncan J Watts and Steven H Strogatz. 1998. Collective dynamics of ‘smallworld’networks. _Nature_ 393, 6684 (1998), 440–442. 

- [46] Mark Weber, Jie Chen, Toyotaro Suzumura, Aldo Pareja, Tengfei Ma, Hiroki Kanezashi, Tim Kaler, Charles E Leiserson, and Tao B Schardl. 2018. Scalable graph learning for anti-money laundering: A first look. _arXiv:1812.00076_ (2018). 

- [47] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom Robinson, and Charles E Leiserson. 2019. Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. _arXiv:1908.02591_ (2019). 

- [48] Zhihao Wen and Yuan Fang. 2022. TREND: TempoRal Event and Node Dynamics for Graph Representation Learning. In _Proc. WWW 2022_ . ACM, 1159–1169. 

- [49] Jiajing Wu, Jieli Liu, Weili Chen, Huawei Huang, Zibin Zheng, and Yan Zhang. 2022. Detecting Mixing Services via Mining Bitcoin Transaction Network With Hybrid Motifs. _IEEE Trans. Syst. Man Cybern. Syst._ 52, 4 (2022), 2237–2249. 

- [50] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. 2021. A Comprehensive Survey on Graph Neural Networks. _IEEE Trans. Neural Networks Learn. Syst._ 32, 1 (2021), 4–24. 

- [51] Leshanshui Yang, Clément Chatelain, and Sébastien Adam. 2024. Dynamic Graph Representation Learning With Neural Networks: A Survey. _IEEE Access_ 12 (2024), 43460–43484. 

- [52] Rex Ying, Zhaoyu Lou, Jiaxuan You, Chengtao Wen, Arquimedes Canedo, and Jure Leskovec. 2020. Neural Subgraph Matching. _arXiv:2007.03092_ (2020). 

- [53] Jiaxuan You, Jonathan M Gomes-Selman, Rex Ying, and Jure Leskovec. 2021. Identity-aware graph neural networks. In _Proc. AAAI 2021_ , Vol. 35. 10737–10745. 

- [54] Yichao Yuan, Haojie Ye, Sanketh Vedula, Wynn Kaza, and Nishil Talati. 2023. Everest: GPU-Accelerated System For Mining Temporal Motifs. _Proc. VLDB Endow._ 17, 2 (2023), 162–174. 

264 

