# **Towards Collaborative Anti-Money Laundering Among Financial Institutions** 

Zhihua Tian Zhejiang University Hangzhou, China zhihuat@zju.edu.cn 

Xiang Yu Alibaba Group Hangzhou, China shaseng.yx@antgroup.com 

Yuan Ding Zhejiang University Hangzhou, China dy1ant@zju.edu.cn 

Enchao Gong Alibaba Group Hangzhou, China enchao.gec@antgroup.com 

Wenjie Qu National University of Singapore Singapore wenjiequ@u.nus.edu 

Jian Liu<sup>∗</sup> Zhejiang University Hangzhou, China liujian2411@zju.edu.cn 

## Kui Ren 

Zhejiang University Hangzhou, China kuiren@zju.edu.cn 

### **Abstract** 

Money laundering is the process that intends to legalize the income derived from illicit activities, thus facilitating their entry into the monetary flow of the economy without jeopardizing their source. It is crucial to identify such activities accurately and reliably in order to enforce anti-money laundering (AML). 

Despite considerable efforts to AML, a large number of such activities still go undetected. Rule-based methods were first introduced and are still widely used in current detection systems. With the rise of machine learning, graph-based learning methods have gained prominence in detecting illicit accounts through the analysis of money transfer graphs. Nevertheless, these methods generally assume that the transaction graph is centralized, whereas in practice, money laundering activities usually span multiple financial institutions. Due to regulatory, legal, commercial, and customer privacy concerns, institutions tend not to share data, restricting their utility in practical usage. In this paper, we propose the _first_ algorithm that supports performing AML over multiple institutions while protecting the security and privacy of local data. 

To evaluate, we construct Alipay-ECB, a real-world dataset comprising digital transactions from Alipay, the world’s largest mobile payment platform, alongside transactions from E-Commerce Bank (ECB). The dataset includes over 200 million accounts and 300 million transactions, covering both intra-institution transactions and those between Alipay and ECB. This makes it the largest real-world transaction graph available for analysis. The experimental results demonstrate that our methods can effectively identify 

∗Corresponding author 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _WWW ’25, Sydney, NSW, Australia._ 

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1274-6/25/04 https://doi.org/10.1145/3696410.3714576 

cross-institution money laundering subgroups. Additionally, experiments on synthetic datasets also demonstrate that our method is efficient, requiring only a few minutes on datasets with millions of transactions. Our code and dataset are available on https: //github.com/zhihuat/Collaborative-AML. 

### **CCS Concepts** 

• **Security and privacy** → **Privacy-preserving protocols** ; • **Computing methodologies** → **Distributed algorithms** . 

### **Keywords** 

Anti-money Laundering; Collaborative Learning 

##### **ACM Reference Format:** 

Zhihua Tian, Yuan Ding, Wenjie Qu, Xiang Yu, Enchao Gong, Jian Liu, and Kui Ren. 2025. Towards Collaborative Anti-Money Laundering Among Financial Institutions. In _Proceedings of the ACM Web Conference 2025 (WWW ’25), April 28–May 2, 2025, Sydney, NSW, Australia._ ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3696410.3714576 

### **1 Introduction** 

Money laundering is a process that attempts to conceal or disguise the origins of dirty money derived from illicit activities, making it appear as if the funds have been obtained through legitimate means [15]. It typically consists of three primary steps: a _placement_ step first introduces the dirty money into existing financial systems; a _layering_ step then carries out complex transactions to hide the source of the funds; and a _integration_ step withdraws the fund from a destination bank account before using it for legitimate activities [14]. The transaction relationship of accounts can be represented as a graph, where an individual account is denoted as a node, and transactions between two accounts are denoted as edges. Due to the distinctive nature of money laundering activities, the transaction graph associated with money launderers exhibits a unique pattern known as **scatter-gather** [3, 5, 13], as illustrated in Fig. 1a. 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Zhihua Tian et al. 



<!-- Start of picture text -->
Institution A<br>Institution B<br>Source Intermediaries Destination Source Intermediaries Destination<br>(a) (b)<br><!-- End of picture text -->

**Figure 1: (a) Scatter-gather pattern money laundering; (b) Scatter-gather distributed across two institutions.** 

It is the responsibility of financial institutions to conduct _antimoney laundering_ (AML): diligently monitor transactions, take necessary actions like shutting down or imposing restrictions on suspicious accounts, and promptly report any suspicious activities through to law enforcement agencies. To detect money laundering activities, a common idea is to identify the ultimate beneficiary, which refers to the individual or entity that ultimately receives the funds, even if those funds have been obscured through multiple layers of transactions [15]. To achieve that, a simple approach is to calculate the ratio to which funds in one account originate from another account [13]. If the ratio exceeds a predefined threshold, it indicates a potential association between the two accounts, raising suspicions of money laundering activities with one account being the source and the other the destination. 

However, money laundering has evolved into a highly sophisticated process, spanning across multiple financial institutions s.t. the subgraph within one institution appears to be normal (Fig. 1b). As a result, relying solely on the transaction graph within a single institution for AML is no longer sufficient. A straightforward solution is to combine the transaction graphs from multiple institutions. However, due to regulatory, legal, commercial, and customer privacy concerns, institutions tend not to share data. 

**Our contribution.** In this paper, we make the _first_ step towards collaborative AML, which allows multiple institutions to jointly conduct AML without exposing their individual transaction graphs. 

Our primary contribution lies in the introduction of a novel algorithm for scatter-gather subgraph mining, specifically tailored to suit the collaborative setting. In more detail, this algorithm first employs a breadth-first search (BFS) approach for each node to identify a set of cross-institution transactions associated with that node, which can be either scattered from or gathered towards the node. If two nodes, belonging to different institutions, share the same set of cross-institution transactions, it indicates a potential scatter-gather relationship within a money laundering subgraph, with one node being the source and the other being the destination. Building upon this observation, the algorithm considers two institutions, denoted by P _𝐴_ and P _𝐵_ , and iterates through their respective nodes ({ _𝑁_ 1<sup>_𝐴, 𝑁_</sup> 2<sup>_𝐴, . . . , 𝑁_</sup> _𝑛_<sup>_𝐴_}and{</sup><sup>_𝑁_</sup> 1<sup>_𝐵, 𝑁_</sup> 2<sup>_𝐵, . . . , 𝑁𝐵𝑛_}) to identify the</sup> sets of cross-institution transactions: S<sup>_𝐴_</sup> = { _𝑆_ 1<sup>_𝐴,𝑆_</sup> 2<sup>_𝐴, . . . ,𝑆𝐴𝑛_}and</sup> S<sup>_𝐵_</sup> = { _𝑆_ 1<sup>_𝐵,𝑆_</sup> 2<sup>_𝐵, . . . ,𝑆𝐵𝑛_}, where e.g.,</sup><sup>_𝑆_</sup> _𝑖_<sup>_𝐴_is the set of cross-institution</sup> transactions associated with node _𝑁_<sup>_𝐴_</sup> _𝑖_<sup>. If two sets</sup><sup>_𝑆_</sup> _𝑖_<sup>_𝐴_and</sup><sup>_𝑆𝐵_</sup> _𝑗_<sup>ex-</sup> hibit a high degree of similarity, it suggests that _𝑁𝑖_<sup>_𝐴_and</sup><sup>_𝑁_</sup> _𝑗_<sup>_𝐵_are</sup> potentially involved in scatter-gather activities within a money laundering subgraph. 

This approach requires P _𝐴_ and P _𝐵_ to exchange S<sup>_𝐴_</sup> and S<sup>_𝐵_</sup> , and measure the similarity between each pair (e.g., _𝑆𝑖_<sup>_𝐴_and</sup><sup>_𝑆𝐵_</sup> _𝑗_<sup>). This is</sup> costly in terms of both communication and computation. To solve the problem, we use locality-sensitive hashing (LSH) [31] and Bloom filter [27] to minimize the amount of information to be exchanged between P _𝐴_ and P _𝐵_ . LSH enables the estimation of similarity between two sets by comparing the minimum hash values of their elements. Combined with Bloom filters, the approach transforms pairwise comparisons into a process of testing the presence of an element within a Bloom filter. The Bloom filter is memory-efficient, and this testing process is computationally efficient. 

Specifically, an LSH is computed for each set, resulting in { _lsh_<sup>_𝐴_</sup> 1<sup>_, lsh𝐴_</sup> 2<sup>_,_</sup> _. . . , lsh𝑛_<sup>_𝐴_</sup> } and { _lsh_<sup>_𝐵_</sup> 1<sup>_, lsh𝐵_</sup> 2<sup>_, . . . , lsh𝐵𝑛_}. Notice that</sup><sup>_lsh_</sup> _𝑖_<sup>_𝐴_=</sup><sup>_lsh𝐵_</sup> _𝑗_<sup>if</sup><sup>_𝑆_</sup> _𝑖_<sup>_𝐴_</sup> and _𝑆_<sup>_𝐵_</sup> _𝑗_<sup>exhibit a high degree of similarity. Next, one institution,</sup> say P _𝐴_ , inserts { _lsh_<sup>_𝐴_</sup> 1<sup>_, lsh𝐴_</sup> 2<sup>_, . . . , lsh𝐴𝑛_} into a bloom filter</sup><sup>_𝐵𝐹𝐴_, and</sup> transfers _𝐵𝐹𝐴_ to P _𝐵_ ; P _𝐵_ iterates through { _lsh_<sup>_𝐵_</sup> 1<sup>_, lsh𝐵_</sup> 2<sup>_, . . . , lsh𝐵𝑛_} to</sup> check if each _lsh_<sup>_𝐵_</sup> is present in _𝐵𝐹𝐴_ . If _lsh_<sup>_𝐵_</sup> _𝑗_<sup>is found in</sup><sup>_𝐵𝐹𝐴_,P</sup><sup>_𝐵_</sup> learns that _𝑁 𝑗_ is one end node in the scatter-gather activity. At this stage, P _𝐵_ reveals the corresponding _lsh_<sup>_𝐵_</sup> _𝑗_<sup>to P</sup><sup>_𝐴_, enabling P</sup><sup>_𝐴_</sup> to identify the other end node in the scatter-gather activity. By leveraging this optimization, the communication overhead is significantly reduced as it only requires the transfer of a bloom filter. Moreover, by comparing against a bloom filter, the computational complexity is reduced to _𝑂_ ( _𝑛_ ), rather than _𝑂_ ( _𝑛_<sup>2</sup> ) when comparing each pair individually. 

To evaluate whether our methods can detect money laundering activities across multiple institutions in a real-world setting, we construct Alipay-ECB, a multi-institution transaction dataset that includes digital currency transactions from Alipay and E-Commerce Bank (ECB) users. The dataset contains over 200 million accounts and 300 million transactions. To the best of our knowledge, it is the largest real-world transaction dataset available. 

By analyzing the dataset, we find that money laundering groups possess a much more intricate structure in real-world settings, encompassing multiple simple patterns such as fan-in, fan-out, cycles, random, and bipartite, etc. However, our method can effectively identify money laundering subgroups. Experiments on synthetic datasets also demonstrate our methods can effectively and efficiently identify money laundering subgroups. 

### **2 Preliminaries** 

This section provides the necessary background and preliminaries for understanding this paper. The frequently used notations are presented in Table 2. 

### **2.1 Scatter-Gather Mining** 

In order to detect money laundering transaction subgraphs of scatter-gather patterns, a simple approach is to examine cases where a significant amount of money flows out of one account and gets aggregated in other accounts [13]. We refer to this method as _Centralized Scatter-Gather Mining_ . To illustrate, let’s consider an example where there’s a node _𝑗_ that receives 80% of the money flowing out from node _𝑖_ . In this case, it’s possible that both nodes _𝑖_ and _𝑗_ , along with the nodes in-between, are involved in a potential money 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Towards Collaborative Anti-Money Laundering Among Financial Institutions 



<!-- Start of picture text -->
100/100 40/40 70/140<br>40/40 40/40<br>a b e<br>60/60 30/100<br>0/140 30/100<br>d c f<br>0/100 60/200 30/100<br><!-- End of picture text -->

**Figure 2: Illustration of centralized scatter-gather mining. Consider** _𝑎_ **as the source and designate all the funds flowing out from** _𝑎_ **as illicit money, visually represented in red. The enclosed numbers within boxes indicate the money possessed by the nodes, while numbers above the line represent the money involved in a transaction. The orange nodes represent the detected money laundering nodes, which have an illicit funds ratio greater than 0.3.** 

laundering activity, with _𝑖_ acting as the source within the subgraph, and _𝑗_ serving as the destination. 

To determine how much of the money received by node _𝑗_ comes from node _𝑖_ , the method utilizes a tracking mechanism based on the transaction graph. This involves marking the outflow money from node _𝑖_ as suspected money and tracing their movement within the graph. When node _𝑖_ sends money to another node _𝑣_ , the marked money is transferred to _𝑣_ . Similarly, if node _𝑣_ subsequently sends money to node _𝑗_ , the marked money is also transferred to node _𝑗_ . In the context of the method, two principles govern the flow of marked money in downstream nodes, considering that money is divisible. Denote _𝑀𝑖𝑛_<sup>_𝑗, 𝑀_</sup> _𝑜𝑢𝑡_<sup>_𝑗_as total inflow and outflow of node</sup><sup>_𝑗_</sup> separately, and _𝑚𝑖𝑛_<sup>_𝑗,𝑚_</sup> _𝑜𝑢𝑡_<sup>_𝑗_as marked inflow and outflow included</sup> in _𝑀𝑖𝑛_<sup>_𝑗, 𝑀_</sup> _𝑜𝑢𝑡_<sup>_𝑗_that satisfy</sup><sup>_𝑚_</sup> _𝑖𝑛_<sup>_𝑗_≤</sup><sup>_𝑀_</sup> _𝑖𝑛_<sup>_𝑗,𝑚_</sup> _𝑜𝑢𝑡_<sup>_𝑗_≤</sup><sup>_𝑀_</sup> _𝑜𝑢𝑡_<sup>_𝑗_.</sup> We have the following principles to calculate _𝑚𝑖𝑛_<sup>_𝑗_:</sup> 



(2) The marked inflow money of a node is the sum of marked money received from other nodes. 

After getting the value of _𝑚𝑖𝑛_<sup>_𝑗_,wecalculatetheratioofinflow</sup> money from _𝑖_ to _𝑗_ as _𝑟𝑖𝑗_ = _𝑚_<sup>_𝑚_</sup> _𝑜𝑢𝑡_<sup>_𝑖_</sup> _<u>𝑖𝑛</u>_<sup>_𝑗_.</sup> 

Figure 2 illustrates an example of applying the method by considering node _𝑎_ as the source node and discovering the scatter-gather pattern it is involved in. By setting the threshold to 40%, we identify three suspected money laundering nodes _𝑏_ , _𝑐_ , and _𝑒_ , which contain 40%, 60% and 70% of marked money, respectively. 

intersection and the number of elements of their union: 



Let _𝐻_ denote the minhash function that maps a set to a real number; it has the property 



That is, the probability that _𝐻_ ( _𝐴_ ) = _𝐻_ ( _𝐵_ ) is true is equal to the similarity _𝐽_ ( _𝐴, 𝐵_ ). 

The details of the MinHash algorithm is following: Given a hash function _ℎ_ that maps the members of a set _𝑈_ to real numbers, and _perm_ which is a random permutation of the elements of _𝑈_ . For any set _𝑆_ ⊂ _𝑈_ , _𝐻_ is defined as the minimum value of _ℎ_ ( _𝑝𝑒𝑟𝑚_ ( _𝑥_ )), i.e., 



Let _𝑟_ be a random variable that is 1 when _𝐻_ ( _𝐴_ ) = _𝐻_ ( _𝐵_ ) and 0 otherwise, _𝑟_ is the unbiased estimator of _𝐽_ ( _𝐴, 𝐵_ ), i.e., _𝐸_ ( _𝑟_ ) = _𝐽_ ( _𝐴, 𝐵_ ). 

The MinHash scheme reduces this variance by averaging together several variables constructed in the same way, such as by applying multiple hash functions. To estimate _𝐽_ ( _𝐴, 𝐵_ ), let _𝑛_ be the number of hash functions for which _𝐻_ ( _𝐴_ ) = _𝐻_ ( _𝐵_ ), _𝐾_<sup>_<u>𝑛</u>_is the esti-</sup> mate, where _𝐾_ is the total number of hash functions used. This estimate is the average of _𝐾_ random variables _𝑟_ s, each of which is the unbiased estimator of _𝐽_ ( _𝐴, 𝐵_ ). Hence, the average is also unbiased. By standard deviation for sums of the variables, the similarity estimation error is O(1/ ~~√~~ _𝐾_ ). 

### **2.3 Bloom filter** 

A Bloom filter [27] is a memory-efficient data structure that is used to test whether an element is present in a set. The price paid for the efficiency is that Bloom filter is a probabilistic data structure: It tells us that the element either _definitely_ is not in the set or _may be_ in the set. In other words, false positive matches are possible, but false negatives are not. 

A Bloom filter is an array of _𝑚_ bits with all positions set to 0 when it is empty. There are also _𝑘_ hash functions, each of which maps or hashes each element in a set to one of the _𝑚_ positions uniformly. To _add_ an element, we simply feed it to each of the _𝑘_ hash functions to get _𝑘_ array positions and set the bits at all these positions to 1. To _query_ an element (test whether it is in the set), hash it using the identical _𝑘_ hash functions to get _𝑘_ array positions. If any of the _𝑘_ positions are 0, the element is _definitely_ not in the set. If all are 1, then the element is either in the set or the bits were set to 1 when inserting other elements by chance, resulting in a false positive. The false positive error _𝜖_ , the size of Bloom filter _𝑚_ , and the number of hash functions _𝑘_ are related in the following way: 



With _𝑚_ increases, the false positive probability _𝜖_ decreases. 

### **2.2 MinHash** 

MinHash [29] is a technique to estimate how similar two sets are, where the similarity is defined in terms of the Jaccard similarity coefficient. Specifically, let _𝐴_ and _𝐵_ are two sets. The Jaccard index is defined to be the ratio of the number of elements of their 

### **3 Problem Statement** 

Let G = (V _,_ E _,_ X) be a money transaction graph, where V is the vertex set represents accounts, E is the edge set represents transactions, and X ∈ R<sup>_𝑑_</sup> is the feature matrix of all edges. An edge 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Zhihua Tian et al. 



<!-- Start of picture text -->
Transaction Sets Banding Matrix Bloom Filters<br>Institution Cross-institution<br>transaction Insert into<br>set discovery LSH Bloom filters<br>Naive approach (Section 4.1): Optimized approach (Section 4.2):<br>Pairwise similarity computation Estimate similarity using LSH and BF<br>Cross-institution<br>transaction Insert into<br>set discovery LSH Bloom filters<br>Institution<br><!-- End of picture text -->

**Figure 3: Workflow of CSGM. The dotted lines on the graphs indicate cross-institution transactions.** 

( _𝑖, 𝑗_ ) ∈E indicates that the account _𝑖_ transfers money to _𝑗_ and the corresponding x ∈X indicates the attributes of the transaction, such as the amount of money, the time, to name a few. In this paper, we mainly focus on two attributes: the amount of money and whether the transaction is an external transaction, denoted as _𝑎_ and _𝑐_ separately. Specifically, x = [ _𝑎,𝑐_ ]<sup>⊤</sup> . For ease of presentation, we denote x _𝑖_ → _𝑗_ the attributes for the transaction from _𝑖_ to _𝑗_ . 

In our setting of collaborative learning, we consider two institutions P _𝐴_ and P _𝐵_ ; each holds a subgraph G _𝐴_ = (V _𝐴,_ E _𝐴,_ X _𝐴_ ) and G _𝐵_ = (V _𝐵,_ E _𝐵,_ X _𝐵_ ), where V _𝑖,_ E _𝑖,_ X _𝑖_ are subsets of V _,_ E _,_ X, separately. In the rest of the paper, we use the notations _𝑝_ and _𝑞_ to denote the indices of the two institutions. Specifically, P _𝑝_ refers to one institution and P _𝑞_ to the other. 

To comply with Know Your Customer (KYC) standards [28], financial institutions are required to gather basic information about both the initiator and recipient of each transaction. This rule remains applicable even when accounts are held across different institutions. Based on this requirement, we assume an overlap between V _𝐴_ and V _𝐵_ . The overlapping nodes represent accounts involved in cross-institution transactions between P _𝐴_ and P _𝐵_ . 

We further assume that the overlapping accounts are recorded with identical identifiers by both institutions. This identification can be performed privately through multi-party private set intersection methods [10], which is orthogonal to our paper. 

Given the above setting, we aim to discover money laundering groups of typologies presented in figure 1a based on two subgraphs V _𝐴_ and V _𝐵_ . 

### **4 Methods** 

In this section, we present in detail how our collaborative AML algorithm, named collaborative scatter-gather mining (CSGM), is designed. We begin by transforming the centralized scatter-gather mining method into the one that can be applied to two subgraphs as defined in Section 3 owned by different institutions. The method enables the detection of money laundering nodes distributed across multiple institutions, particularly when the source and destination nodes belong to different institutions. 

We further enhance the method by making use of Localitysensitive hashing (LSH) [31] and Bloom filter [27] to minimize 

communication costs and improve efficiency. Figure 3 presents the workflow of CSGM. 

### **4.1 Collaborative Scatter-Gather Mining** 

In the scatter-gather pattern of money laundering, money is transferred from a source to a destination through multiple transactions involving many adversarial middle nodes. When the source and the destination are located in different institutions, it implies that money laundering activities transfer money to another institution via cross-institution transactions, as shown in Figure 1b. 

The key idea behind our method is that the set of cross-institution transactions scattered from the source is identical to the set of crossinstitution transactions gathered at the destination when the source and destination are involved in the same money laundering subgraph. Therefore, by comparing the sets of transactions identified by both institutions, we can effectively detect money laundering subgraphs in which both source and destination are implicated. 

Specifically, let S _𝑝_ ←[ _𝑆𝑖_ | _𝑖_ ∈V _𝑝_ ] and D _𝑝_ ←[ _𝐷𝑖_ | _𝑖_ ∈V _𝑝_ ] for _𝑝_ ∈{ _𝐴, 𝐵_ } denote the sets of all cross-institution transactions associated with P _𝑝_ , where _𝑆𝑖_ and _𝐷𝑖_ represent the sets obtained through scattering from or gathering to node _𝑖_ , respectively. P _𝑝_ transmits both S _𝑝_ and D _𝑝_ to institution P _𝑞_ , ensuring that both P _𝐴_ and P _𝐵_ possess all relevant sets. By independently comparing the similarity between any two sets _𝑆𝑖_ ∈S _𝑝_ and _𝐷 𝑗_ ∈D _𝑞_ , each institution can identify sources or destinations involved in money laundering activities. Specifically, P _𝑝_ can detect sources by comparing sets from S _𝑝_ with those from D _𝑞_ , and similarly, identify destinations by comparing sets from D _𝑝_ with those from S _𝑞_ . Note that we filter out the discovered sets of small size (setting the threshold to 4-7 in our experiments), considering that money laundering groups are typically huge to conceal substantial amounts of money. Once all suspicious sources and destinations are identified, intermediate nodes can be readily located by tracing the transactions that are scattered from sources or gathered to destinations within the local subgraph. 

**Cross-institution Transaction Set Discovery.** To find the set of cross-institution transactions, each institution employs the BFS approach for each node to find transactions scattered or gathered 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Towards Collaborative Anti-Money Laundering Among Financial Institutions 

from the node and determine if they are cross-institution transactions. Specifically, it starts from a specific node and loops all neighbor nodes to identify cross-institution transactions originating from the node until either all relevant transactions are found or the maximum depth is reached. Let F represent the algorithm, and we denote the discovery process as _𝑆𝑖_ ←F ( _𝑖,_ G _,𝑇_ ). Here, _𝑆𝑖_ is the set of cross-institution transactions scattered from node _𝑖_ , G is the local transaction graph, and _𝑇_ denotes the maximum depth allowed. When aiming to discover the gathered transaction sets, we can simply transform G into a new graph G<sup>′</sup> with the inverse direction. Specifically, G<sup>′</sup> = (V _,_ E<sup>′</sup> _,_ X), where E<sup>′</sup> = {( _𝑗,𝑖_ )| ( _𝑖, 𝑗_ ) ∈E}. By performing the same algorithm on G<sup>′</sup> , we construct the reversed transaction set as _𝐷𝑖_ ←F ( _𝑖,_ G<sup>′</sup> _,𝑇_ ). Algorithm 3 in Appendix presents the procedure. 

### **4.2 Optimization for Distributed Scatter-Gather Mining** 

Applying the distributed scatter-gather mining algorithm directly is both communication- and computation-intensive, as it requires institutions to exchange multiple transaction sets (O( _𝑛_ ), where _𝑛_ is the number of nodes) and perform pairwise comparisons among them, which is O( _𝑛_<sup>2</sup> ). To address the challenge, we propose an optimized algorithm using LSH [31] and Bloom filters [27]. LSH enables the estimation of similarity between two sets by comparing the minimum hash values of their elements, and by inserting the results of all sets (either from S or D) into a bloom filter, we transform pairwise comparisons into a more efficient process of testing whether an element exists within the Bloom filter. 

Specifically, institution P _𝑝_ first performs LSH on all sets. The results are then inserted into _𝐾_ Bloom filters, where _𝐾_ is determined by the length of the LSH value. The Bloom filters are then shared with another institution, P _𝑞_ . By querying the Bloom filter with the LSH of P _𝑞_ ’s local set, which is likely to match those of other sets with high similarity, P _𝑞_ can efficiently detect the existence of a similar set, thereby determining whether the corresponding node is involved in potential money laundering activities. As it requires only the transfer of Bloom filters, the optimization significantly reduces communication overhead, Moreover, the computational complexity is reduced to O( _𝐾𝑛_ ) _,𝑘 << 𝑛_ , as opposed to O( _𝑛_<sup>2</sup> ) when performing pairwise comparisons. 

Next, we provide a detailed explanation of how sets are inserted into Bloom filters. We then introduce two methods, namely Probability-Based Similar Set Detection and Similarity-Based Similar Set Detection, to detect similar sets using Bloom filters. 

_4.2.1 Inserting sets into Bloom filters._ We adopt the MinHash algorithm (cf. Section 2.2) as the approach to implement LSH. Take S<sup>_𝑝_</sup> as an example, P _𝑝_ first employs _𝑚_ distinct minhash functions _𝐻_ (cf. Equation 3) on each set _𝑆𝑖_ ∈S _𝑝_ resulting in a signature matrix _𝑀_<sup>_𝑝_</sup> _𝑆_<sup>with</sup><sup>_𝑚_rows and |S</sup><sup>_𝑝_| columns, where | · | denotes the number</sup> of sets in S _𝑝_ . Each row of the matrix represents applying the same minhash function to all sets in S _𝑝_ , and each column represents applying all minhash functions to the same set. 

A banding technique then be applied to the matrix. Specifically, we divide the matrix into bands, each containing _𝑟_ rows of the matrix, resulting in a total of _𝐾_ = _𝑚_ / _𝑟_ bands. Each column of a band, which is composed of the result of applying _𝑟_ minhash 



<!-- Start of picture text -->
Banding<br>1 1 1 1 1<br>Indices 0 1 2 3 4 5 6 7 8 9 10<br>1 1 1 1 1<br>Indices 0 1 2 3 4 5 6 7 8 9 1 0<br>1 1 1 1 1<br>Indices 0 1 2 3 4 5 6 7 8 9 10<br>Signature Matrix Banding Matrix Bloom Filters<br><!-- End of picture text -->

**Figure 4: Example of inserting transaction sets into Bloom filters. We consider three sets** _𝑆_ 1 **,** _𝑆_ 2 **, and** _𝑆_ 3 **and 6 MinHash functions. The band width** _𝑟_ = 2 

functions to one set, can be treated as a result of applying LSH on the set. If two sets have the Jaccard similarity of _𝑠_ , then the probability that their columns within the same band are equal is _𝑠_<sup>_𝑟_</sup> . By mapping each column to a distinctive signature, for example, by utilizing the MD5 function [32], each band can be treated as one row of the banding matrix. We denote it as _𝐵𝑆_<sup>_𝑝_. We then insert</sup> each band into a Bloom filter (cf. Section 2.3), resulting in _𝐾_ Bloom filters _𝐵𝐹_<sup>_𝑝_</sup> _𝑆_<sup>[1]</sup><sup>_, ..., 𝐵𝐹_</sup> _𝑆_<sup>_𝑝_[</sup><sup>_𝐾_]. When the context is clear, we omit the</sup> superscripts and subscripts, and represent each Bloom filter as _𝐵𝐹𝑘,𝑘_ ∈{1 _, ..., 𝐾_ }. 

We note that to guarantee the LSH of two similar sets are equal with high probability, P _𝑝_ and P _𝑞_ are required to use the same MinHash functions on S _𝑝_ and D _𝑞_ . Figure 4 presents an example of inserting three sets _𝑆_ 1, _𝑆_ 2, and _𝑆_ 3 into three Bloom filters, with the band with _𝑟_ = 2. 

_4.2.2 Probability-based similar set detection._ With the received Bloom filters, institutions can detect whether the node is involved in money laundering activities by querying the existence of the band values in corresponding Bloom filters. Specifically, for a set _𝑆𝑖_ ∈S<sup>_𝑝_</sup> , denote its band values as _𝐵𝑖_ , which are a column in the banding matrix. P _𝑝_ query the existence of each _𝐵𝑘𝑖_ in the corresponding Bloom filter _𝐵𝐹𝑘_ . Theoretically, if there exists a set _𝐷 𝑗_ ∈D<sup>_𝑞_</sup> that exhibits a similarity of _𝑠_ with _𝑆𝑖_ , the probability that at least one Bloom filter contains _𝐵𝑘𝑖_ is: 



where _𝐾_ = _𝑚_ / _𝑟_ . As shown in Figure 5, by appropriately selecting values for _𝑚_ and _𝑟_ , this probability can be adjusted to be close to 1 or 0, depending on the level of similarity. For example, when the threshold is 0 _._ 4, we set _𝑟_ = 4 and _𝑚_ = 400 so the probability is about 0 _._ 92. Consequently, if at least one _𝐵𝑘𝑖_ tested exists in _𝐵𝐹𝑘_ , we treat the corresponding node of _𝑆𝑖_ as a potential source within a money laundering subgroup. 

With the probability-based similar set detection, we denote our AML method as **Prob-CSGM** and Algorithm 1 presents the pseudocode of the method. 

_4.2.3 Similarity-based similar set detection._ While the probabilitybased method enables the detection of source/destination nodes involved in money laundering activities, it suffers from a high false 



<!-- Start of picture text -->
1-(1-s')”™"<br>1.0<br>0.8 r=3,m=120<br>> r=3,m=180<br>= 0.6 r=3,m=240<br>2 r=3,m=300<br>2Q —— r=4,m=400<br>2 0.4 — r=5,m=500<br>a. I— r=6,m=600<br>0.2 I— r=7,m=700<br>i—— r= 8,m=800<br>0.0 I—— r=9,m=900<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Ss<br><!-- End of picture text -->



<!-- Start of picture text -->
><br><!-- End of picture text -->



<!-- Start of picture text -->
U<br><!-- End of picture text -->

( 

) 

x 

~~-~~ x 



<!-- Start of picture text -->
AMLSim — bal<br>0.8 —r=1<br>—— r=2<br>v a. — r=3<br>S Hh r=3<br>20.4 —7 \=5<br>a — r=6<br>0.2 \<br>00, 0.2 0.4 0.6 0.8<br>Threshold<br><!-- End of picture text -->



<!-- Start of picture text -->
AMLSim — unb<br>0.8<br>—r=1<br>06 — r=2<br>v —- r=3r=<br>S — 1r=3<br>a4 —— te<br>a — r=6<br>0.2 \<br>0.0<br>0.2 0.4 0.6 0.8<br>Threshold<br><!-- End of picture text -->



<!-- Start of picture text -->
AMLWorld— HI<br>0.8<br>—r=1<br>06 — r=2<br>v —- r=3r=<br>S — r=3<br>B04 — r=5<br>a — r=6<br>0.2<br>0.0<br>0.2 0.4 0.6 0.8<br>Threshold<br><!-- End of picture text -->



<!-- Start of picture text -->
AMLWorld— LI<br>0.8<br>—r=1<br>06 — r=2<br>v —- r=3r=<br>Soa — r=3<br>a” — r=5<br>a — r=6<br>0.2<br>0.0<br>0.2 0.4 0.6 0.8<br>Threshold<br><!-- End of picture text -->

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Zhihua Tian et al. 

**Table 1: Experiments on AMLSim and AMLWorld datasets. "-" represents that the metric is unsuitable for the method. We bold the best experimental results and underline the second-best results.** 

||||_AMLSim_bal_|||||_AMLSim_unb_|||
|---|---|---|---|---|---|---|---|---|---|---|
|**Methods**|ACC|Precision|Recall|F1-score|AUC|ACC|Precision|Recall|F1-score|AUC|
|SGM|0.9761|0.8627|0.9047|0.8832|**0.9743**|0.9805|0.8665|0.9678|0.9144|0.9876|
|GIN [7, 33]|0.9497±0.0021|0.8397±0.0096|0.8992±0.0033|0.8684±0.0047|0.9301±0.0016|0.8978±0.0064|0.6926±0.0159|0.9045±0.0034|0.7844±0.0109|0.9003±0.0049|
|GAT [24]|0.8332±0.0123|0.5285±0.0201|0.9173±0.0021|0.6704±0.0159|0.8657±0.0071|0.8235±0.0181|0.5424±0.0288|0.9251±0.0035|0.6835±0.0225|0.8611±0.0113|
|PNA [25]|0.9533±0.0017|0.8508±0.0078|0.9061±0.0018|0.8776±0.0039|0.9351±0.0011|0.9177±0.0043|0.7470±0.0125|0.9066±0.0022|0.8190±0.0079|0.9136±0.0032|
|LaundroGraph [4]|0.936±0.0014|0.7870±0.0048|0.8961±0.0034|0.8380±0.0031|0.9206±0.0018|0.9136±0.0043|0.7350±0.0126|0.9070±0.0077|0.812±0.0077|0.9112±0.0029|
|MultiGIN [5]|0.9827±0.0003|**0.9949**±**0.001**|0.9108±0.0016|0.9510±0.0008|0.9549±0.0008|0.9809±0.0005|**0.9955**±**0.0012**|0.9110±0.0018|0.9514±0.0012|0.9550±0.0009|
|Prob-CSGM|0.9858±0.0007|0.9926<br>±0.0003|0.8638±0.0072|0.9237±0.0041|-|0.9880±0.0019|0.9928±0.0010|0.8943±0.0176|0.9409±0.0096|-|
|Sim-CSGM|**0.9908**±**0.0006**|0.9833±0.0012|**0.9231**±**0.0068**|**0.9522**±**0.0033**|0.9607±0.0034|**0.9964**±**0.0012**|0.9930±0.0002|**0.9737**±**0.0114**|**0.9833**±**0.0058**|**0.9865**±**0.0057**|
||||_AMLWorld_HI_|||||_AMLWorld_LI_|||
|**Methods**|ACC|Precision|Recall|F1-score|AUC|ACC|Precision|Recall|F1-score|AUC|
|SGM|0.9992|0.5187|0.6501|0.5770|0.8250|0.9989|0.0314|0.1765|0.0533|0.5878|
|GIN [7, 33]|0.9984±0.0004|0.2938±0.0781|0.5526±0.0892|0.3811±0.0828|0.7757±0.0447|0.9997±0.0001|0.1791±0.0350|0.1647±0.0738|0.1598±0.0474|0.5823±0.0369|
|GAT [24]|0.9992±0.0001|0.5572±0.1188|0.2143±0.0136|0.3081±0.0326|0.6071±0.0068|0.998|0.0|0.0|0.0|0.5|
|PNA [25]|0.9985±0.0001|0.3565±0.0208|**0.9380**±**0.0097**|0.5165±0.0234|**0.9683**±**0.0049**|0.9997±0.0001|0.3321±0.0052|0.8873±0.0069|0.4833±0.0065|0.9435±0.0035|
|LaundroGraph [4]|0.9992±0.0001|0.5412±0.0840|0.6193±0.0613|0.5710±0.0299|0.8094±0.0306|0.9998±0.0043|0.3846±0.0126|0.0490±0.0077|0.0870±0.0077|0.5245±0.0029|
|MultiGIN [5]|0.9996±0.0002|0.6945±0.0959|0.9366±0.0173|0.7943±0.0658|0.9681±0.0086|0.9996±0.0001|0.1746±0.0365|0.3353±0.2252|0.2104±0.1101|0.6675±0.1125|
|Prob-CSGM|0.9996±0.0001|**0.8747**±**0.0242**|0.6413±0.0643|0.7392±0.0499|-|0.9998±0.0001|0.4370±0.0684|0.3529±0.1038|0.3878±0.086|-|
|Sim-CSGM|**0.9997**±**0.0001**|0.7718±0.0191|0.8292±0.0136|**0.7995**±**0.0128**|0.9145±0.0068|**0.9999**±**0.0009**|**0.6458**±**0.0008**|**0.9118**±**0.0001**|**0.7561**±**0.0041**|**0.9558**±**0.0002**|



low, such as in the AMLWorld-HI dataset, MultiGIN suffers from low precision, leading to a high false positive rate. In contrast, our methods maintain strong performance on the AMLWorld datasets, highlighting the generalizability of our approach. 

### **5.4 Ablation study.** 

To explore the impact of different parameters on the performance of our methods, we conducted experiments by varying the number of hash functions used in MinHash _𝑚_ , the number of rows _𝑟_ in each band as well as the threshold. 

Here, we mainly focus on Sim-CSGM. We vary the threshold from 0.2 to 0.6 and observe the change of F1-score with different _𝑟_ . The results are depicted in Figure 6. It shows that the F1-score when _𝑟 >_ 1 performs better than when _𝑟_ = 1, showing the effectiveness of the banding technique in the similarity-based method. Furthermore, when _𝑟_ = 1, the method prefers a higher threshold, illustrating that repeated elements in a band lead to an overestimation of similarity when using the bloom filter. 

Additionally, experiments in Appendix A.3 show that the banding technique could significantly reduce the number of repeated elements. We also evaluate the efficiency of our methods in terms of the communication costs as well as the running time in Appendix A.4. The results show that our methods take only a few minutes. 

### **6 Related works** 

The term money laundering was first used at the beginning of the 20th Century to label the operations that in some way intended to legalize the income derived from illicit activity, thus facilitating their entry into the monetary flow of the economy [22]. Since then, numerous methods have been proposed to identify money laundering activities [6, 11, 13, 17–19, 34]. Rule-based approaches were first widely used in the early days [13, 17]. Rajput et al. [17] propose an ontology-based expert system to detect suspicious transactions, and Michalak et al. [13] propose a method that integrates the fuzzing 

method and decision rules to detect suspicious transactions. Although easy to deploy, rule-based methods can easily be evaded by fraudsters. 

With the popularity of machine learning, learning-based methods have become an emergency. Tang et al. [21] propose to use the support vector machine method (SVM) to detect unusual behaviors in transactions. Lv et al. [12] judge whether the capital flow is involved in money laundering activities using RBF neural networks calculated from time to time. Paula et al. [16] also show some success for AML by using deep neural networks. However, these methods detect money laundering activities in a supervised manner, suffering from highly skewed labels and limited adaptability. 

Graphs have the advantage of better characterizing the association between objects. Many graph-based anomaly detection techniques have been developed for discovering structural anomalies. Zhang et al. [34] use financial transaction networks and community detection algorithms to find money laundering groups. Cardoso et al. [4] introduces a self-supervised graph representation learning method aimed at detecting money laundering. Recently, Béni et al. [5], incorporates a range of adaptations, including multigraph port numbering, ego IDs, and reverse message passing, to enhance GNNs’ ability to detect various patterns of illicit activities. 

Despite the advance of all those methods, they work based on the prerequisite that the transaction graph is centralized, while in practice, money laundering activities span across multiple institutions s.t. the transaction subgraph within one institution appears to be normal. Our methods make the _first_ steps towards collaborative anti-money laundering among institutions without exposing the transaction graphs. 

### **7 Conclusion** 

In this work, we propose the _first_ algorithm enabling collaborative anti-money laundering (AML) among institutions while preserving the privacy of their transaction graphs. We employ LSH [31] and bloom filters [27] to reduce communication costs and enhance efficiency. Experimental evaluations on two synthetic 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Towards Collaborative Anti-Money Laundering Among Financial Institutions 

datasets demonstrate the effectiveness and efficiency of the proposed algorithm. In future work, we will attempt to deploy the algorithm in real-world industrial settings to evaluate its effectiveness with realistic data. Moreover, we will enhance the algorithm to address intricate money laundering scenarios involving more institutions and more complex transaction graphs. 

### **Acknowledgments** 

This work is sponsored in part by National Key Research and Development Program of China (2023YFB2704000) and CCF-AFSG research fund. 

### **References** 

- [1] 2024. alipay. https://www.alipay.com/. 

- [2] 2024. E-Commerce Bank. https://www.mybank.cn/. 

- [3] Erik Altman, Jovan Blanuša, Luc Von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. 2024. Realistic synthetic financial transactions for anti-money laundering models. _Advances in Neural Information Processing Systems_ 36 (2024). 

- [4] Mário Cardoso, Pedro Saleiro, and Pedro Bizarro. 2022. LaundroGraph: Selfsupervised graph representation learning for anti-money laundering. In _Proceedings of the Third ACM International Conference on AI in Finance_ . 130–138. 

- [5] Béni Egressy, Luc Von Niederhäusern, Jovan Blanuša, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. 2024. Provably Powerful Graph Neural Networks for Directed Multigraphs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 38. 11838–11846. 

- [6] Bryan Hooi, Hyun Ah Song, Alex Beutel, Neil Shah, Kijung Shin, and Christos Faloutsos. 2016. Fraudar: Bounding graph fraud in the face of camouflage. In _Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining_ . 895–904. 

- [7] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. 2019. Strategies for pre-training graph neural networks. _arXiv preprint arXiv:1905.12265_ (2019). 

- [8] Kaggle. 2024. IBM Transactions for Anti Money Laundering (AML). https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-antimoney-laundering-aml/data. 

- [9] Md Rezaul Karim, Felix Hermsen, Sisay Adugna Chala, Paola de Perthuis, and Avikarsha Mandal. 2023. Catch me if you can: Semi-supervised graph learning for spotting money laundering. _arXiv preprint arXiv:2302.11880_ (2023). 

- [10] Vladimir Kolesnikov, Naor Matania, Benny Pinkas, Mike Rosulek, and Ni Trieu. 2017. Practical multi-party private set intersection from symmetric-key techniques. In _Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security_ . 1257–1272. 

- [11] Nhien An Le Khac and M-Tahar Kechadi. 2010. Application of data mining for anti-money laundering detection: A case study. In _2010 IEEE international conference on data mining workshops_ . IEEE, 577–584. 

- [12] Lin-Tao Lv, Na Ji, and Jiu-Long Zhang. 2008. A RBF neural network model for anti-money laundering. In _2008 International conference on wavelet analysis and pattern recognition_ , Vol. 1. IEEE, 209–215. 

- [13] Krzysztof Michalak and Jerzy Korczak. 2011. Graph mining approach to suspicious transaction detection. In _2011 Federated conference on computer science and_ 

_information systems (FedCSIS)_ . IEEE, 69–75. 

- [14] United Nations. 2020. Tax abuse, money laundering and corruption plague global finance. 

- [15] United Nations. 2024. Money Laundering. https://www.unodc.org/unodc/en/ money-laundering/overview.html. 

- [16] Ebberth L Paula, Marcelo Ladeira, Rommel N Carvalho, and Thiago Marzagao. 2016. Deep learning anomaly detection as support fraud investigation in brazilian exports and anti-money laundering. In _2016 15th ieee international conference on machine learning and applications (icmla)_ . IEEE, 954–960. 

- [17] Quratulain Rajput, Nida Sadaf Khan, Asma Larik, and Sajjad Haider. 2014. Ontology based expert-system for suspicious transactions detection. _Computer and Information Science_ 7, 1 (2014), 103. 

- [18] Reza Soltani, Uyen Trang Nguyen, Yang Yang, Mohammad Faghani, Alaa Yagoub, and Aijun An. 2016. A new algorithm for money laundering detection based on structural similarity. In _2016 IEEE 7th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON)_ . IEEE, 1–7. 

- [19] Michele Starnini, Charalampos E Tsourakakis, Maryam Zamanipour, André Panisson, Walter Allasia, Marco Fornasiero, Laura Li Puma, Valeria Ricci, Silvia Ronchiadin, Angela Ugrinoska, et al. 2021. Smurf-based anti-money laundering in time-evolving transaction networks. In _Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part IV 21_ . Springer, 171–186. 

- [20] Toyotaro Suzumura and Hiroki Kanezashi. 2021. Anti-Money Laundering Datasets: InPlusLab Anti-Money Laundering DataDatasets. http://github.com/ IBM/AMLSim/. 

- [21] Jun Tang and Jian Yin. 2005. Developing an intelligent data discriminating system of anti-money laundering based on SVM. In _2005 International conference on machine learning and cybernetics_ , Vol. 6. IEEE, 3453–3457. 

- [22] Rodolfo Uribe. [n. d.]. Changing Paradigms on Money Laundering. http://www. cicad.oas.org/oid/new/information/observer/observer2_2003/mlparadigms.pdf. 

- [23] Atif Usman, Nasir Naveed, and Saima Munawar. 2023. Intelligent Anti-Money Laundering Fraud Control Using Graph-Based Machine Learning Model for the Financial Domain. _Journal of Cases on Information Technology_ 25, 1 (2023). 

- [24] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. _arXiv preprint arXiv:1710.10903_ (2017). 

- [25] Petar Velickovic, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio, and R Devon Hjelm. 2019. Deep graph infomax. _ICLR (Poster)_ 2, 3 (2019), 4. 

- [26] Mark Weber, Jie Chen, Toyotaro Suzumura, Aldo Pareja, Tengfei Ma, Hiroki Kanezashi, Tim Kaler, Charles E Leiserson, and Tao B Schardl. 2018. Scalable graph learning for anti-money laundering: A first look. _arXiv preprint arXiv:1812.00076_ (2018). 

- [27] Wikipedia. 2023. Bloom Filter. https://en.wikipedia.org/wiki/Bloom_filter. 

- [28] Wikipedia. 2023. Know Your Customer. https://en.wikipedia.org/wiki/Know_ your_customer. 

- [29] Wikipedia. 2023. MinHash. https://en.wikipedia.org/wiki/MinHash. 

- [30] Wikipedia. 2024. Alipay. https://en.wikipedia.org/wiki/Alipay. 

- [31] Wikipedia. 2024. Locality-sensitive hashing. https://en.wikipedia.org/wiki/ Locality-sensitive_hashing. 

- [32] Wikipedia. 2024. MD5. https://en.wikipedia.org/wiki/MD5. 

- [33] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How powerful are graph neural networks? _arXiv preprint arXiv:1810.00826_ (2018). 

- [34] Zhongfei Zhang, John J Salerno, and Philip S Yu. 2003. Applying data mining in investigating money laundering crimes. In _Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data mining_ . 747–752. 

WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia. 

Zhihua Tian et al. 

### **A Appendix** 

**Table 2: Summary of notations** 

|**Notation**|**Description**|
|---|---|
|G|transaction graph|
|G<sup>′</sup><br>_𝑚_|transaction graph with inversed direction<br>number of hash functions used in MinHash|
|_𝑟_|number of rows of each band|
|_𝐾_|number of bloom flters|
|_𝑆_|the set of cross-institution transactions fnd with G|
|_𝐷_|the set of cross-institution transactions fnd with G<sup>′</sup>|
|S_,_D|list of sets_𝑆_and_𝐷_|
|_𝑀_|Signature matrix|
|_𝐵_|Banding matrix|



### **A.1 Theoretical Analysis** 

Theorem 1 (restated). _Suppose that 𝑋_ 1 _, ...,𝑋𝑁 are a sequence of real values with_ 0 ≤ _𝑋𝑁_ ≤ _..._ ≤ _𝑋_ 1 ≤ 1 _. Then_ ∀ _𝜀 >_ 0 _, there_ ∃ _𝛿, s.t. when 𝑟 > 𝛿,_ 



Proof. We have 



To prove the inequality, we only need to prove 



It is easy to prove that 





where _𝑝_ = _𝑋_ 2/ _𝑋_ 1 

□ 

### **A.2 Dataset Statistics** 

**Alipay-ECB.** Alipay Mobile Payment [1] is the world’s largest mobile payment platform, allowing users to pay for a wide range of daily needs, including money transfers, online shopping, salary deposits, investments, and more. As of June 2020, Alipay serves over 1.3 billion users and 80 million merchants [30], making it an invaluable resource for studying money laundering activities that may be concealed within its vast volume of transaction records. Meanwhile, E-Commerce Bank, operating entirely online, serves tens of millions of users and merchants across China, facilitating numerous financial transactions between Alipay and E-Commerce Bank daily. The AlipayECB dataset captures these transactions, with the majority of records originating from Alipay. These include transactions between Alipay users as well as between users and various bank accounts, with ESB being one of the many banks involved. 

**Time span.** The AlipayECB dataset is constructed using transactions that occurred on Alipay and E-Commerce Bank (ECB) within a single day. Unlike synthetic datasets such as AMLSim [20] and AMLWorld [3], where transactions within a single money laundering group can span several days [8], money laundering transactions on digital platforms tend to occur rapidly. Funds are moved in and out quickly to minimize the risk of losses due to account monitoring and censorship. Based on this observation, we focus exclusively on transactions occurring within a single day. 

**Data processing.** To facilitate the experiments on the dataset, we process the data as follows: 

- **Account Segregation:** A user may link deposits or credit cards from different banks to their Alipay account. As a result, numerous transactions occur between Alipay and the user’s cards (e.g., through withdrawal services). To facilitate the tracing of fund flows between different banks, we treat each card as a separate account, even if they belong to the same user. 

- **Transaction Aggregation:** Transactions between two accounts may occur multiple times. However, as we mainly focus on a set of transactions, we consolidate these transactions into a single entry. This is different from MultiGIN [5], which treats transactions between the same accounts as distinct entities. 

- **Transaction Filtering:** Given that money laundering often involves large sums, we filter out transactions with small amounts (around ¥100 in our experiments). 

After applying these steps, we obtain a transaction graph with 48.95 million accounts and 34.45 million transactions. Detailed statistics are presented in Table 3. 

#### **Table 3: Statistics of Alipay-ECB after processing.** 

||Accounts|||Transa|ctions||
|---|---|---|---|---|---|---|
|Alipay|ECB|Others|Alipay→Alipay|ECB→ECB|Alipay→ECB|ECB→Alipay|
|23.46M|3.99M|21.50M|30.84M|5.51M|0.25M|1.65M|



**Examples of discovered subgraphs** Figure 8 presents examples of detected groups. Money laundering groups are identified when more than half of their accounts are classified as illicit. Grey groups are those in which only a small number of accounts have been reported for suspected money laundering activities. The normal group is associated with school financial collections. 



<!-- Start of picture text -->
10 3 mm yr =1<br>L mm r=2<br>oO<br>2 40° Ea ;=3<br>° r=4<br>oO<br>r= 1<br>10° | | F | 1<br>123 45 6 7 8 9 10 11 12<br>Repeat times<br><!-- End of picture text -->



<!-- Start of picture text -->
—_—<br>|<br>|<br>|<br>|<br><!-- End of picture text -->



<!-- Start of picture text -->
|<br>—<br>a<br>Aa<br>LE GER<br><!-- End of picture text -->



<!-- Start of picture text -->
Aa<br>LE<br><!-- End of picture text -->



<!-- Start of picture text -->
a<br>GER<br><!-- End of picture text -->









