# **BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining** 

Yichao Yuan University of Illinois Urbana-Champaign yichaoy2@illinois.edu 

Arjun Laxman 

Haojie Ye University of Michigan yehaojie@umich.edu 

University of Michigan arlx@umich.edu 

Krisztian Flautner 

Nishil Talati 

University of Michigan manowar@umich.edu 

University of Illinois Urbana-Champaign nishil@illinois.edu 



<!-- Start of picture text -->
Src Dst Curr Amt Type Streaming Graph<br>Financial  A D USD 9,857 ACH Representation<br>Transactions<br>Data R Z Euro 35 Wire<br>T X USD 365 Check +<br>Scope of BlazingAML: …<br>>100x higher throughput<br>pattern matching without  Pattern Mining Pattern Matching<br>losing accuracy Contributions:<br>Multi-stage framework for expressing patterns<br>Domain-specific CPU+GPU compiler for mining<br>Financial  Src Dst Curr Amt Type Graph Mining Features<br>Transactions  A D USD 9,857 ACH<br>Data with<br>Augmented  R Z Euro 35 Wire<br>Features T X USD 365 Check<br>Downstream AI-based<br>Classifier + Prediction<br><!-- End of picture text -->

## **ABSTRACT** 

Money laundering detection faces challenges due to excessive false positives and inadequate adaptation to sophisticated multi-stage schemes that exploit modern financial networks. Graph analytics and AI are promising tools, but they struggle with the fuzziness of laundering patterns, which exhibit structural and temporal variations. Conventional data mining techniques require the detailed enumeration of pattern variants, which not only complicates the analyst’s task to specify them, but also leads to large run-time overheads and difficulty training accurate AI models. We present BlazingAML, a scalable Anti Money Laundering (AML) system design that introduces a novel _multi-stage framework_ for expressing fuzzy money laundering patterns and a _domain-specific compiler_ that transforms high-level pattern descriptions into highperformance code for multiple hardware back-ends: CPU and GPU. Our multi-stage abstraction decomposes complex laundering schemes into logical stages connected by graph operations, enabling diverse patterns to be expressed using unified primitives while capturing structural and temporal fuzziness. The compiler applies sophisticated optimizations, eliminating manual parallel programming requirements for financial analysts. Evaluation in IBM AML data sets shows that BlazingAML achieves the same F1 score as stateof-the-art approaches while delivering a significant 210× and 333× higher speedup on CPU and GPU, with superior scalability. 

**Figure 1: Overview and contributions of BlazingAML.** 

due to the _fuzzy_ nature of laundering patterns. Money laundering schemes have structural and temporal fuzziness [4]: structural fuzziness involves varying numbers of intermediate accounts with the same topology, while temporal fuzziness allows flexible timing ordering between transactions. These characteristics create two challenges for scalable AML systems: expressing fuzzy patterns and efficiently mining them at high throughput for real-time financial crime detection across massive transaction graphs. 

## **1 INTRODUCTION** 

Money laundering poses a severe threat to global financial stability, enabling organized crime and terrorism by disguising illicit proceeds [8, 27]. Current Anti-Money Laundering (AML) systems rely on rule-based techniques that combine customer due diligence, transaction monitoring thresholds, and suspicious activity reporting to detect criminal activity [12–14]. However, these conventional approaches face limitations: they generate excessive false positives, struggle to adapt to evolving laundering strategies, and fail to detect sophisticated multi-stage schemes exploiting modern financial networks [5, 9]. Combining graph analytics and Artificial Intelligence (AI) offers a promising alternative that can identify complex patterns that traditional rule-based systems cannot capture. 

Although there is research on exact pattern matching and subgraph mining [21, 23, 28, 35], little work has been done on systematically expressing fuzzy patterns or developing high-performance AML algorithms using them. Mining fuzzy patterns by reducing them to exact patterns causes a combinatorial explosion, which makes the approach of using exact mining for fuzzy patterns, at a minimum, impractical. 

In this paper, we propose BlazingAML: a scalable AML system that systematically expresses fuzzy money laundering patterns and compiles them into high-performance mining code. As shown in Fig. 1, our design follows a typical AML pipeline that combines graph pattern matching with AI techniques. The first stage employs specialized graph mining to detect known laundering patterns, such 

Graph and AI-driven techniques show promise for real-world money laundering detection [1, 4, 13, 19, 32] but face challenges 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 

as scatter-gather schemes and multi-hop cycles. It augments each transaction edge with pattern occurrence counts as additional features. A downstream gradient boosting classifier [6] then leverages these enriched features to classify individual transactions as legitimate or illicit. While a more sophisticated AI-based classifier ( _e.g.,_ graph neural network [17, 20]) can improve AML accuracy, this paper focuses on improving graph pattern mining expressibility and throughput. Trying new classifiers is a compelling direction for future work. 

The _main contribution of this work is the graph mining pipeline_ optimized for AML: 

- (1) A _multi-stage specification_ technique that captures structural and temporal fuzziness in laundering patterns, enabling analysts to describe variable-topology schemes and flexible timing constraints in a unified framework. 

- (2) A _domain-specific compiler_ that transforms high-level pattern descriptions into optimized C++ and CUDA kernels, handling graph-specific optimizations. 

Our multi-stage specification technique introduces a unified abstraction that decomposes complex money laundering patterns into logical stages connected by fundamental graph operations. This allows diverse laundering schemes to be expressed using the same high-level primitives while naturally capturing both structural and temporal fuzziness inherent in real-world patterns. 

This approach establishes a description language for AML patterns that allows domain experts to specify only the logical structure of detection algorithms while automatically generating optimized implementations for different hardware architectures, effectively separating pattern logic from performance optimization concerns. 

The framework’s modular design enables scalable pattern mining on graphs with millions of nodes through automatic parallelization and incremental processing, while supporting extensibility through simple pattern library modifications rather than fundamental algorithmic reimplementation. 

The specifications are compiled by a domain-specific compiler from high-level declarative pattern specifications into optimized C++ and CUDA kernels. The optimizations performed by the compiler include: 

- Power-law-aware memory access pattern generation, 

- Degree-based workload balancing, and 

- Pipelined CPU-GPU execution based on pattern structure. 

Our system enables rapid development, specification, and deployment of optimized fuzzy graph mining algorithms by providing a seamless flow from specification to optimized CPU-GPU code. Combined with a gradient-boosted classifier, we can create an endto-end AML pipeline, forming a system we refer to as BlazingAML. 

To evaluate the effectiveness of BlazingAML, we compare both accuracy and speed using AML datasets released by IBM [1]. Our evaluation demonstrates that BlazingAML achieves significantly higher throughput. By mining graph patterns and using them as augmented features, identical in value to those used by GFP [4], BlazingAML attains the same level of F1 score as GFP while being substantially faster. 

In particular, our experiments show average speedups of 210× on CPU and 333× on GPU compared to GFP. Moreover, BlazingAML exhibits superior scalability, maintaining performance advantages 

as the size of the input transaction graph increases. We further benchmark against a graph transformer–based approach, FraudGT [19]. While FraudGT achieves a higher F1 score through computationally expensive model training and inference, BlazingAML delivers 4.9× higher throughput, making it far more practical for large-scale, real-time AML workloads. BlazingAML makes the following novel contributions. 

- Design of a _multi-stage framework_ to flexibly express money laundering patterns. 

- Design of a _compiler_ that outputs pattern-specific high-performance code for multiple hardware back-ends. 

- BlazingAML: an _end-to-end AML system design_ that delivers 4.9× higher throughput compared to the state-of-the-art with the same level of accuracy. 

## **2 BACKGROUND AND RELATED WORK 2.1 Money Laundering** 

Money laundering is the process by which criminals disguise the illicit origin of funds to integrate them into the legitimate financial system. If undetected, it fuels organized crime, enabling drug cartels, human trafficking rings, and terrorist organizations that cause immense human suffering. For example 150,000 murders are linked to Mexican drug cartels since 2006 and an estimated 40 million people have been enslaved through trafficking [32]. The UN estimates up to 5% of global GDP, or roughly 2 trillion USD, is laundered each year, with global financial crimes totaling 3.5 trillion USD in 2020 [34]. The fast growth of digital transaction businesses also gives rise to more sophisticated financial crimes. 

Money laundering typically unfolds in three stages: (1) **Placement** , where illicit proceeds are broken into smaller deposits to avoid detection [25], (2) **Layering** , involving complex fund transfers among shell companies and accounts to obscure origins, and (3) **Integration** , where cleaned funds are reintroduced into the economy through assets like real estate or securities. Money laundering detection is primarily focused on the **Layering** stage, where illicit funds are circulated through the financial system in various forms and transactions to obscure their origin. Despite tens of billions of dollars spent annually on compliance [18], and severe penalties such as the 530 million USD fine levied on the Commonwealth Bank of Australia in 2018 [2], Europol estimates only 1% of illicit funds are recovered [11]. 

Detecting money laundering, called Anti Money Laundering (AML), is an extremely challenging technical problem. It is a _needlein-a-haystack_ problem in massive, noisy, fragmented, and constantly changing transaction datasets, often spread across multiple institutions and jurisdictions [32]. Below, we outline the current real-world practices of AML and present a set of advanced graphbased techniques designed to automate detection and achieve scalable AML monitoring. 

## **2.2 Rule-Based AML** 

Today, financial institutions primarily rely on rule-based techniques to detect and prevent money laundering. These approaches combine regulatory compliance requirements, structured monitoring, and risk assessment procedures to identify suspicious behavior, flag potential illicit activity, and ensure adherence to AML standards. 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 



<!-- Start of picture text -->
Fan-out Fan-in Scatter-gather Gather-scatter Cycle Bipartite Stack<br><!-- End of picture text -->

**Figure 2: Representative graph patterns illustrating layering strategies in money laundering within financial transaction graphs [1].** 

The following sections outline key components of conventional AML practices [13]. 

- **Customer due diligence.** Financial institutions must verify the identities of their clients through comprehensive know-yourcustomer (KYC) procedures. This process entails gathering personal identification information, performing background checks, and evaluating risk profiles based on factors such as geographic location, transaction patterns, and credit history. 

- **Transaction monitoring.** Continuous surveillance of customer transactions is essential to detect irregular or suspicious activity. This includes monitoring for unusually large deposits, structuring of transactions to avoid reporting thresholds, and transfers involving high-risk individuals, entities, or jurisdictions. 

- **Suspicious activity reporting (SAR).** When potential money laundering is identified, institutions are required to submit SARs to local financial intelligence units [12]. These reports enable authorities to investigate suspicious transactions and support law enforcement in taking appropriate action. 

- **Internal audits and compliance programs.** Regular internal audits are necessary to ensure AML programs remain compliant with evolving regulations, address operational gaps, and counter increasingly sophisticated laundering methods [14]. These audits are vital for evaluating the effectiveness of existing controls and identifying vulnerabilities, particularly given the complexity and scale of transactions facilitated by digital and mobile platforms. Rule-based techniques for combating money laundering struggle 

- to keep up with the increased sophistication of financial transactions, particularly due to the widespread adoption of mobile payments and interconnected networks. This complexity leads to high false-positive rates, burdening AML teams and allowing criminal activities to go undetected, underscoring the need for advanced detection mechanisms. 

- These structural patterns are often linked to established money laundering strategies, including circular layering and smurfing [32]. More recent approaches extend this idea by employing subgraph mining and filtering techniques to expose anomalous relational structures that may signal laundering flows [10]. 

- **AI-based techniques.** Pure AI methods represent transactions as sequences or features and apply neural models for classification or anomaly detection. CNNs can extract spatial-temporal correlations in transaction matrices [26]; RNNs such as GRU/LSTM capture sequential dependencies in mobile or banking transaction streams [3]; transformers model long-range dependencies across transaction histories [29, 36]; and deep reinforcement learning has been explored by formulating AML detection as a sequential decision task [31]. These methods can adapt to evolving laundering strategies, though they often face challenges of imbalance and limited real-world labels. 

- **Hybrid graph and AI-based techniques.** Hybrid methods leverage both structural and learned features. For example, InspectionL combines self-supervised graph embeddings with a downstream Random Forest classifier [20]. Graph neural networks and their extensions directly capture relational dependencies, with methods such as EvolveGCN [24], LaundroGraph [5], FraudGT [19], and GAGNN [7]. GFP [4] incorporates pattern counts as edge features, which are then provided to a lightweight downstream classifier ( _e.g.,_ XGBoost [6]). These approaches enrich traditional AI models with graph context, leading to state-of-the-art performance on datasets like Elliptic [33] and IT-AML [1]. 

A key challenge with purely graph pattern matching–based techniques is that money laundering strategies continuously evolve, making static motif detection insufficient. Moreover, as highlighted in prior work [1, 4], the presence of a particular pattern does not necessarily indicate fraudulent activity. For instance, fan-in and fanout structures are common in legitimate account activity and are therefore not inherently suspicious. Conversely, purely AI-based approaches often struggle to capture the rich structural interactions between nodes and edges in transaction graphs. To address these limitations, this work proposes a hybrid system that integrates graph-based and AI-based methods. Following the design philosophy of GFP [4], our approach employs graph pattern matching in the front end to extract known suspicious patterns (see Fig. 2). The counts of these patterns are then incorporated as node and edge features, which are subsequently used by a downstream AI classifier to detect potential money laundering behavior. 

## **2.3 Graph and AI-based AML** 

Financial transaction data can be naturally represented as graphs, where a node represents a bank account and an edge between a pair of nodes represents a financial transaction between two accounts. Attributes on nodes and edges represent the amount of money transferred, currency, bank account details, etc. Converting a financial transaction database into a graph representation presents an opportunity to use graph analytics, AI, or a combination of these two techniques for detecting money laundering. Below, we discuss three broad categories of techniques used for AML. 

- **Graph pattern matching-based techniques.** Graph pattern matching methods identify suspicious motifs in transaction graphs: such as cycles, cliques, or hub–spoke topologies [1, 4] (see Fig. 2). 

## **3 CHALLENGES IN AML SYSTEM DESIGN** 

This section discusses the challenges of designing a scalable detection system for AML. 

**High data volumes.** Detecting money laundering is particularly challenging due to the immense volume of financial transactions that must be monitored. The proliferation of digital and mobile payment platforms, as well as smart IoT devices, has further exacerbated this problem by dramatically increasing both the scale and velocity of transactional data [13]. These systems generate millions of heterogeneous records daily, often across multiple payment channels, making rapid detection increasingly complex. Moreover, criminals exploit this high-volume environment by structuring 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 



<!-- Start of picture text -->
Intermediate Structural Fuzziness Temporal Fuzziness<br>nodes<br>t1 t2 ti<br>t3 t4<br>t5 t6<br>Scatter-gather pattern #Intr. nodes: 2 #Intr. nodes: 4 t2>t1; t4>t3; t6>t5 max(ti)-min(ti)<delta<br><!-- End of picture text -->

**Figure 3: Fuzziness illustrated in the scatter-gather money laundering pattern: (1) structural fuzziness in terms of different numbers of intermediate nodes, and (2) temporal fuzziness in terms of partial time ordering and time window.** 

illicit flows into smaller, inconspicuous digital transactions ( _e.g.,_ smurfing), which are easily obscured within legitimate traffic [13]. Consequently, modern AML systems must not only scale to vast transaction streams but also remain adaptive to the evolving complexity of digital ecosystems. 

**High-throughput requirement.** An effective AML system must achieve high throughput to process massive transaction streams in real time and prevent illicit funds from being integrated into the financial system. The ability to sweep through large volumes of data quickly is critical, as delays in detection can allow criminals to move assets across borders or convert them into untraceable forms, severely limiting recovery and enforcement efforts. However, designing high-throughput AML systems is challenging due to the need to balance scalability with accuracy; processing data at speed often increases false positives and risks overlooking subtle laundering patterns. Furthermore, heterogeneous data sources and evolving transaction structures demand architectures that are both computationally efficient and adaptable to new laundering strategies. 

**Complex transaction patterns.** As shown in Fig. 2, graph pattern matching in AML requires mining highly complex structures, such as cycles, cliques, and multi-hop motifs—that capture subtle laundering behaviors [4]. Algorithms that can identify these patterns at scale is computationally expensive, as the search space grows combinatorially with the size of the pattern and transaction graph. This makes it particularly challenging to achieve high throughput when processing massive financial datasets, where billions of edges must be examined for rapid detection. Consequently, as shown in our results, scalable graph mining for AML remains a core bottleneck in building efficient detection systems. 

**Fuzzy patterns.** While there is a large body of work designing algorithms that work for mining a fixed pattern, money laundering patterns exhibit fuzziness across two axes, as shown in Fig. 3 as detailed below. 

- _Structural fuzziness:_ Although a pattern has a fixed structural shape, the number of nodes and edges involved can vary significantly. For instance, in a scatter–gather pattern, the number of intermediate _placement_ accounts between the source and destination may range from just a few to dozens. Laundering actors can freely choose the number of intermediaries and the connectivity structure to evade detection. Enumerating and matching all such variants would require mining a combinatorial number of exact patterns, which is computationally expensive. 

- _Temporal fuzziness:_ While many temporal mining frameworks [21, 23, 35] impose a strict global temporal ordering of edges within a time window, money laundering patterns often do not adhere to such rigid structures. In practice, transactions may not follow a global sequence. For example, the scatter and gather phases can be temporally decoupled, with only local or partial ordering constraints ( _e.g., 𝑡𝑖 < 𝑡𝑖_ +1) holding. As a result, even minor deviations in temporal order can cause overly rigid pattern-matching systems to miss true positives. 

Structural and temporal fuzziness is challenging to capture for a neural network, given the amount of training data that’s available. The "XGB Only" column of Table 2 illustrates the low F1 scores without graph-pattern-derived features. The rightmost column in the table shows the significantly better results after the graph has been annotated with the mined patterns, thereby supporting the argument that pattern-mining should be used to generate features that increase model accuracy. 

Existing commercial graph query languages, like Cypher [22], are limited in their ability to express the structural and temporal fuzziness of money laundering patterns. Cypher can handle basic fuzzy structures like variable-length paths, but it lacks syntax to express complex topological variations within a single query. Moreover, Cypher lacks support for temporal fuzziness, where edge orderings may follow partial temporal constraints and time windows may overlap. This forces analysts to decompose fuzzy patterns into rigid subqueries, losing the holistic view necessary for effective pattern mining and requiring complex post-processing. 

Recent graph-mining systems have introduced “anti-edge” and “anti-vertex” constructs to express absence constraints in pattern queries. For example, Peregrine [15] explicitly provides anti-edge/antivertex operators to forbid certain vertices or edges from matching, and [16] defines an anti-vertex to exclude specific neighbors in subgraph queries declaratively. These constructs are designed for general graph-pattern tasks (e.g., filtering maximal cliques or anomalies by excluding specific **single** or **a few** unwanted connections, typically enumerated manually by the authors), emphasizing expressive pattern constraints rather than any AML-specific logic. By contrast, BlazingAML targets fundamentally different challenges: analyzing massive, noisy transaction graphs with inherently “fuzzy” structural and temporal patterns. AML patterns do not follow fixed anti-vertex structures but instead describe the procedural logic of how money flows in a laundering strategy. Effective AML detection must tolerate both structural and temporal ambiguity (e.g., **any number** of approximate or missing edges and flexible time windows), which rigid anti-edge/anti-vertex semantics cannot capture. In short, while anti-edge/anti-vertex constructs are conceptually interesting, they address a distinct problem space and do not overlap with BlazingAML ’s core contributions in automated, highperformance, and scalable anti-money-laundering pattern mining over high-volume real-time financial transactions. 

**Distinguishing between innocent vs. fraudulent transactions.** Many transaction patterns also appear in benign scenarios, necessitating joint reasoning over structural, temporal, and contextual features to reduce false positives. 

While rule-based and purely AI-driven AML approaches provide partial solutions, the most effective approaches increasingly 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 

adopt a hybrid paradigm [4] that integrates graph pattern mining with machine learning classifiers. This combination captures structural irregularities and adaptive behavioral cues, improving robustness against evolving laundering strategies. However, existing pattern mining systems [1, 4, 21, 23, 35] lack expressive frameworks to flexibly model real-world laundering patterns, especially those with fuzzy structural or temporal variations. They also fail to scale when applied to massive, streaming transaction graphs required for practical AML deployment. For instance, the IBM system [4], despite strong detection accuracy, demonstrates a steep performance decline as transaction volumes increase (as shown in our evaluation), highlighting the limitations of current graph engines in high-throughput settings. These gaps motivate the BlazingAML system design, which unifies flexible pattern specification with scalable execution to realize hybrid AML detection at a practical scale. 

## **4** BLAZINGAML **DESIGN GOALS** 

Prior graph mining systems [21, 23, 35] assume fixed shapes and strict edge orders for patterns that are infeasible to employ in a large-scale AML setting due to pattern fuzziness. 

Detecting scatter-gather money laundering patterns with rigid pattern definitions for each variant leads to multiple distinct algorithm implementations with different graph traversal logic and temporal validation rules. The computational complexity becomes prohibitive as pattern variations increase: for a 3-size scatter-gather pattern, existing frameworks must enumerate 6! = 720 distinct temporal constraint combinations, making exhaustive enumeration computationally infeasible for large-scale or real-time detection systems. Even with identical structural topology, exact algorithms must enumerate all possible combinations of partial temporal constraints, creating a combinatorial explosion that scales as _𝑂_ ( _𝑛_ !) where _𝑛_ is the number of participating edges. This factorial growth renders traditional approaches impractical for realistic money laundering scenarios. 

Furthermore, each exact algorithm must be separately optimized and maintained while redundantly scanning the same graph regions multiple times, resulting in enormous implementation complexity and computational overhead. The proliferation of specialized algorithms creates a maintenance burden that grows quadratically with the number of supported pattern variants. To address these fundamental limitations of exact pattern matching, we designed a unified fuzzy pattern matching compiler with the goals discussed below, which captures all structural and temporal variations in a single multi-stage algorithm. 

**Expressivity beyond fixed templates.** Analysts need to articulate domain rules that go beyond rigid motifs. For instance, in the scatter–gather (smurfing) and cycled laundering scenario, a bank may require that a lower bound on the number of placement edges ( _e.g.,_ “at least N”), but not mining a specific size, as shown in Fig. 3. 

**Flexible temporal semantics.** Financial behaviors frequently obey only partial orders and window constraints: “integration occurs after placement” within a time horizon _𝛿_ , but events inside each phase are mutually interchangeable. Similarly, in cycled laundering, funds may traverse a cycle with edges that are _not_ observed in strict cycle order ( _e.g.,_ account A credits B earlier as camouflage, 

before illicit funds arrive at A). Fixed-shape, exact-order miners cannot directly express these variants without enumerating all permutations. A practical pattern language must (i) represent steps as _logical time advances_ with optional per-step partial orders; (ii) allow cross-step constraints ( _e.g.,_ causality, min/max fanout, node inequality); and (iii) support out-of-order evidence (anticipatory edges) provided eventual consistency constraints are satisfied. Prior frameworks that pin every edge to a global strict order miss these semantics. 

**Separation of concerns for analysts.** Most AML users are not graph systems experts. They should specify _what_ constitutes suspicious behavior, not _how_ to implement neighbor enumeration, set intersections, or pruning strategies. A flexible compiler decouples high-level intent from low-level execution, enabling non-experts to encode complex constraints safely and audibly. The compiler can map declarative stages to optimized kernels and loops (neighbor ordering, early exits on temporal violations, degree-aware intersections, workload balancing across threads/warps/cores), and target multiple backends (CUDA for GPUs, OpenMP for CPUs). This preserves scalability on power-law transaction graphs while freeing analysts from hand-tuning. 

**Why BlazingAML?** These requirements directly motivate our design in §5. We introduce _Temporal Segment Composition_ , which represents patterns as sequences of logical time steps with interchangeable operations inside a step and explicit cross-step constraints. This abstraction simultaneously captures structural and temporal fuzziness while remaining compiler-friendly: it admits static checks, enables aggressive pruning, and maps cleanly to specialized CUDA and OpenMP code. In the next section, we detail this representation and how it underpins our code generation pipeline. 

## **5 MULTI-STAGE FRAMEWORK FOR EXPRESSING MONEY LAUNDERING GRAPH PATTERNS** 

A central challenge in detecting money laundering lies in the diversity and structural complexity of illicit transaction patterns, as discussed in §3. To address this, we present a novel **multi-stage framework** for expressing and detecting money laundering patterns in large-scale financial transaction graphs. The core innovation lies in decomposing complex laundering patterns into a series of logical stages, each representing a distinct phase in the money laundering process. A **_stage_** captures how money flows from one entity to another or how previously discovered transaction chains can be systematically extended. Each stage is produced by applying specific **_operations_** (such as neighbor expansion, set intersection, or union) on one or multiple previous stages. These operations act on fundamental graph elements called **_operands_** , which can be nodes, edges, or the outputs from preceding stages. By chaining these stages together, analysts can describe sophisticated laundering patterns in a systematic and computationally tractable manner. 

**Illustrative examples: scatter-gather and cycle patterns.** Consider the **scatter-gather pattern** , a common technique where funds are dispersed through multiple accounts before being reconsolidated. In our proposed multi-stage framework, this pattern is decomposed as follows as shown in Fig. 4(a): Stage 0 to Stage 1 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 



<!-- Start of picture text -->
Legend Graph nodes Symbolic nodes Set of nodes Operator<br>Intermediatenodes N1 Stage0 N0 N0.out() Stage0 N0 N0.in()<br>out() N1 N2 out()<br>N0 N2 Stage1 N1 Stage1 N1<br>Stage2out() N2 N2.in() N0 N3 Stage2 out() N2 N2.out()<br>TriggerEvent N0 N1 Stage3Intermediate node set∩ TriggerEvent N0 N1 Stage3 N3 set ∩<br>(a) Scatter-gather pattern (b) 4-Cycle pattern<br>…<br>…<br><!-- End of picture text -->

**Figure 4: Example of (a) scatter-gather and (b) 4-cycle patterns expressed in the proposed multi-stage framework. Note the same stage-based structure for the two patterns, with different in- and out-neighbor sets at stages 0 and 2.** 

represents the initial trigger transaction from node N0 to N1, initiating the mining process. Stage 1 to Stage 2 involves traversing the out-neighbors of N1 (denoted as N2), effectively capturing potential layering accounts where funds are dispersed. Stage 2 to Stage 3 performs an intersection operation between N2’s in-neighbors and N0’s out-neighbors, identifying candidate gathering accounts where funds reconverge. Symbolic nodes in the figure represent either a single graph node or a node set within each stage. 

Notably, multiple intermediate accounts may be discovered in parallel, which naturally captures _structural fuzziness_ in laundering patterns (Fig. 3). Beyond structure, _temporal fuzziness_ , not shown in Fig. 4 for simplicity, can also be expressed by introducing temporal ordering or window constraints as needed. For instance, a temporal ordering between outgoing and incoming edges of intermediate nodes ( _i.e.,_ the first option illustrated under temporal fuzziness in Fig. 3) can be encoded by adding an ordering constraint between edges spanning Stage 2 to Stage 1 and Stage 1 to Stage 0. 

Similarly, **cycle** detection, another fundamental laundering structure, can be expressed using identical primitives as shown in Fig. 4(b). The cycle pattern begins with Stage 0 to Stage 1 representing the trigger transaction N0 to N1, followed by Stage 1 to Stage 2 involving expansion to N2. Finally, Stage 2 to Stage 3 performs an intersection of N2’s out-neighbors with N0’s in-neighbors, effectively closing the loop to yield a complete cycle. Despite being structurally different from scatter-gather patterns, cycles are expressed using the same basic operations: neighborhood expansion and intersection, highlighting the generality and unifying power of our proposed framework. 

**Integration with streaming analytics and machine learn-** 

**ing.** Real-world financial transactions arrive in a streaming fashion, making it crucial to design mining algorithms that can operate over continuously arriving edges in financial transaction graphs. Our framework seamlessly integrates with modern streaming analytics and machine learning pipelines. Each incoming transaction edge updates a time-windowed feature representation, while the mining process maintains comprehensive feature lists across all pattern instances. When a new edge arrives in the transaction stream, it automatically increments counts for all pattern instances it participates in, such as scatter-gather patterns of various sizes or different cycle configurations. These continuously updated feature vectors are then fed into machine learning models such as XGBoost [6], 

enabling a hybrid approach that combines rule-based pattern detection with statistical learning. This tight integration ensures that the framework transcends traditional symbolic pattern matching, supporting sophisticated feature-driven decision-making that can adapt to evolving laundering techniques. 

**Rationale and advantages of the multi-stage framework.** The decomposition of money laundering detection into discrete stages provides several compelling advantages that address fundamental challenges in AML. 

- (1) The stage abstraction naturally captures the logical flow of money laundering operations, with each stage representing a unique transaction flow in the network. This alignment mirrors established financial investigation practices, making the framework intuitive for domain experts. 

- (2) The modular nature of stages enables analysts to construct complex patterns from simple, reusable building blocks, promoting both code reuse and pattern library development. 

- (3) The framework inherently supports parallelism and node interchangeability: operations within a stage can usually be executed simultaneously across multiple candidate nodes, enabling efficient utilization of modern GPU and multi-core CPU architectures. 

The proposed approach introduces several novel dimensions to AML research. First, the framework provides a _unified abstraction_ where diverse laundering patterns, including scatter-gather, cycles, and chain structures, can all be expressed using the same high-level primitives. This standardization establishes a common descriptive language for AML patterns, facilitating pattern library development and cross-institutional benchmarking. Furthermore, the proposed abstraction allows analysts to specify only the logical structure of patterns while the system automatically determines optimal backend implementations (as discussed in the next section), whether using OpenMP for CPU parallelization, CUDA for GPU acceleration, or specialized graph-optimized routines. Complex optimization decisions, such as exploiting power-law graph structures or dynamically selecting smaller neighborhoods at runtime, are entirely encapsulated within the compiler, hiding implementation complexity from domain experts such as bankers and analysts. 

The framework’s design directly addresses the scalability challenges inherent in real-world financial crime detection. By leveraging automatic parallelization and stage-wise decomposition, the system scales effectively to financial graphs containing millions of nodes and edges, representative of major financial institutions’ transaction volumes. The modular stage architecture also enables incremental processing, where new transactions can trigger localized pattern updates rather than requiring full graph recomputation. Additionally, the extensibility of the framework means that new laundering patterns can be incorporated by simply modifying stage definitions, without requiring fundamental algorithmic reimplementation. This combination of performance optimization, scalability, and adaptability positions the multi-stage framework as a significant advancement in computational approaches to financial crime detection, offering both theoretical elegance and practical utility for large-scale anti-money laundering systems. 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 

## **6 A DOMAIN-SPECIFIC COMPILER FOR MULTI-STAGE AML PATTERN MINING** 

This section presents the design of a compiler that takes multi-stage AML pattern description (§5) and outputs high-performance code with CPU and GPU back-ends. 

**Design goals and architecture.** Our system introduces a domainspecific compiler that transforms high-level declarative AML pattern specifications into highly optimized C++ and CUDA kernels. The primary design goal is to enable AML analysts to focus purely on the logical description of suspicious transaction patterns, without requiring any furtner manual generation and optimization of complex parallel graph-processing code. This abstraction layer bridges the gap between domain expertise in financial crime detection and the technical complexity of high-performance graph computing. 

The compiler accepts declarative input specifications that contain sequences of logical stages describing target patterns in terms of fundamental graph traversal primitives. These primitives include **for_all** operations for iteration over all edges or neighbors of a node, **intersection** operations for simultaneous matching of neighbor sets between nodes, **differentiate** operations for filtering with conditional logic such as eliminating self-connections, and **operands** representing graph entities including nodes, edges, and their attributes like source, destination, and transaction timestamps. Input files ( _e.g.,_ YAML configurations) serve as one example format for expressing these specifications, though the compiler architecture supports multiple input representations. 

**Compilation of any general patterns.** The fundamental uniqueness of BlazingAML’s framework is that it allows users to express any general AML pattern as a logical procedure, rather than forcing them into a rigid motif shape defined by a fixed sequence of temporal edges. In our formulation, the analyst specifies the desired laundering logic using a set-operation language that precisely describes how money laundering entities evolve over the course of the laundering pattern. This design isolates the user’s conceptual pattern definition from the details of BlazingAML’s optimized execution flow, which must account for graph size, power-law distributions, hardware heterogeneity (CPU/GPU), and production deployment concerns such as data movement and memory layout. 

Under this abstraction, BlazingAML defines any mined pattern as a composition of set operations, embedding within these operations all forms of structural and temporal fuzziness that naturally arise in AML investigations. To support this, BlazingAML breaks down the logical flow of a laundering scheme into a sequence of stages. Each stage represents a logical advancement of the flow—e.g., layering, scattering, reconvergence—rather than a single observed transaction. A stage declares its input set, output set, and the operation that transforms one into the other, along with node constraints (e.g., account type, currency type) and temporal constraints (e.g., time window or ordering). 

The set operation at a stage can be exact—enumerating all neighbors and thus matching the semantics of traditional temporal motifs [23]—or it can be defined as any expression over sets emitted by earlier stages, such as union, intersection, difference, or general set differentiation. This flexibility lets the user express both tightly constrained patterns and highly fuzzy behaviors where multiple 

branches of any number of accounts or non-strict temporal orderings are allowed. The input sets for a stage may be newly defined or may reference any prior stage’s outputs in any time order, enabling broad expressive power. In contrast, classical temporal motifs restrict each edge to depend solely on the immediately previous edge and require monotonically increasing timestamps; this becomes just a special case of BlazingAML’s general template. 

By filling in this template, an analyst can describe an arbitrary AML pattern—simple or complex—using concise logical steps. Once the pattern is specified, BlazingAML converts each stage into optimized CPU/GPU code independently. Each stage is compiled into a nested-loop structure: the compiler generates an outer loop over the stage’s input set, and inner logic that performs the required set operations using operands drawn either from freshly constructed sets or from any previously materialized stage outputs. The compiler tracks data dependencies to determine whether a stage’s results should feed forward into later stages or whether an earlier stage must be revisited to supply the operands needed by a downstream set operation (e.g., to fulfill a required union, intersection, difference, or differentiate over future stages). 

To make the compilation process more concrete, Algorithm 1 shows the generic pseudo-code template for a single cell of the nested loop constructed for each user-specified stage. Each cell is instantiated according to its configuration file, including operation type, source node set, destination node set, time window, and skip/break constraints. The compiler then automatically assembles these cells into a full nested-loop structure, respecting the logical order of stages and applying hardware-aware optimizations. 

**Algorithm 1** Generic compiled pseudo-code for one stage of a user-specified AML pattern (form_nested_loop). 

|1:|// Stage confguration|
|---|---|
|2:|// cfg.op∈{for_all, union, intersect, diferentiate}|
|3:|// cfg.src: input node set (graph-based or prior stage variable)|
|4:|// cfg.dst_var: output node-set variable for this stage|
|5:|// cfg.constraints: {skip_if, break_if (incl. time window)}|
|6:|_𝑒𝑑𝑔𝑒_𝑠𝑡𝑎𝑟𝑡_←Find_Starting_Edge(_𝑡,𝑐𝑓𝑔.𝑠𝑟𝑐_)|
|7:|**for each**_𝑒_**in**IterateEdges(_𝑐𝑓𝑔.𝑠𝑟𝑐,𝑒𝑑𝑔𝑒_𝑠𝑡𝑎𝑟𝑡,𝑐𝑓𝑔.𝑜𝑝_) **do**|
|8:|// iterate according to op|
|9:|**if**_𝑒_satisfes_𝑐𝑓𝑔.𝑐𝑜𝑛𝑠𝑡𝑟𝑎𝑖𝑛𝑡𝑠.𝑏𝑟𝑒𝑎𝑘_𝑖𝑓_**then**|
|10:|// including time-window overfow|
|11:|**break**|
|12:|**if**_𝑒_satisfes_𝑐𝑓𝑔.𝑐𝑜𝑛𝑠𝑡𝑟𝑎𝑖𝑛𝑡𝑠.𝑠𝑘𝑖𝑝_𝑖𝑓_**then**|
|13:|**continue**|
|14:|_𝑛_←ExtractNode(_𝑒_)|
|15:|// output of each stage is nodes, not edges|
|16:|cfg.dst_var.add(_𝑛_)|
|17:|// surviving nodes feed the next stage of the nested loop|
|18:|**if** is_final_stage**then**|
|19:|Postprocess(cfg.dst_var)|
|20:|// fnal stage result assembly|



Beyond direct translation, BlazingAML treats all optimization decisions holistically. Using information such as neighborhood size, expected cardinality of intermediate sets, and temporal-window 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 

selectivity, the compiler decides how each stage should be evaluated, which set operations should be reordered, and whether the stage is better executed on CPU, GPU, or as a hybrid pipeline. Optimizations include GPU load balancing, memory-coalesced neighbor iteration, ordering set operations based on estimated cost, and avoiding unnecessary materialization through on-demand propagation. Because these optimizations are performed automatically, AML analysts can write patterns at a high level without needing expertise in parallel programming or hardware management. This compiler-based approach thus enables BlazingAML to support a wide range of AML patterns—exact, fuzzy, or hybrid—within one unified abstraction. 

**Input specification and transformation process.** Each stage in the input specification defines five key components: 

- (1) Input operands (such as **N0.out_neigh** or **N1.in_neigh** ), 

- (2) Operation types ( **for_all** or intersection) 

- (3) Optional temporal constraints (like **Find_Start(t-** _𝛿_ **)** for time window boundaries) 

- (4) Filtering conditions for skipping nodes or early termination based on timestamp thresholds, and 

- (5) Output variables containing intermediate results passed to subsequent stages. 

The compiler parses these specifications, validates operand dependencies between stages to ensure logical consistency, and maps the abstract operations onto highly optimized code templates specifically designed for graphs exhibiting power-law degree distributions common in financial networks. 

The generated code templates incorporate sophisticated optimizations tailored for real-world financial transaction graphs. These optimizations include memory access patterns specifically designed for skewed degree distributions, intelligent workload balancing across CPU threads and GPU warps, efficient set intersection algorithms using degree-based ordering to minimize comparison operations, and overlapping CPU-GPU execution strategies that pipeline data transfer with mining operations across temporal windows. The compiler automatically selects and applies these optimizations based on the pattern structure and target hardware architecture. Next, we discuss examples of two patterns to showcase how the compiler outputs high-performance code. 

**Scatter-gather pattern compilation example.** As shown in Fig. 5(b), the input specification defines this pattern through two logical stages: a scatter phase and a gather phase. The scatter phase uses a **for_all** operation to enumerate neighbors from **N0.in_neigh** , storing results in variable **N2** , with constraints to skip cases where **N2 == N1** and break if the edge timestamp **e0.t** exceeds the time threshold **t** . The gather phase employs an intersection operation between **N2.out_neigh** and **N1.in_neigh** , incorporating temporal constraints to break if either **e1.t** or **e2.t** exceeds the time limit, and outputs results to the **sg_tx** variable. 

The compiler transforms this specification into optimized kernels that exploit the parallel nature of neighbor enumeration and set intersection operations. For GPU execution, the compiler generates CUDA kernels that assign individual threads to process different candidate nodes in parallel, utilizing coalesced memory access patterns and shared memory for efficient neighbor list processing. The temporal constraints are compiled into conditional branches that 

enable early termination, reducing unnecessary computation when time windows are exceeded. 

**4-Cycle pattern compilation example.** As illustrated in Fig. 5(c), the four-node cycle pattern demonstrates the compiler’s versatility in handling different topological structures using the same primitive operations. The specification defines two stages: a first leg that uses **for_all** to traverse from **N0.out_neigh** to variable **N2** , with constraints to skip self-loops ( **N2 == N0** ) and respect temporal boundaries, and a second leg that performs an intersection between **N2.out_neigh** and **N0.in_neigh** to identify closing nodes **N3** , filtering out cases where **N3 == N1** and enforcing temporal constraints on both **e1.t** and **e2.t** . 

The compiler recognizes that cycle detection requires different optimization strategies compared to scatter-gather patterns. For the cycle pattern, the generated code prioritizes memory locality for sequential neighbor traversals and implements efficient cycle completion checks. The intersection operation in the second leg is optimized using sorted neighbor lists and binary search techniques, taking advantage of the typically sparse connectivity in financial transaction graphs to minimize intersection complexity. 

**Optimization strategies and code generation.** The compiler’s sophistication lies in its ability to automatically select and apply pattern-specific optimizations without manual intervention. For scatter-gather, fan, and cycle patterns, the system generates hybrid CPU/GPU code that assigns different traversal stages to the most suitable architecture, thereby achieving additional speedup and leveraging the strengths of both. BlazingAML maps shallow traversals to the GPU, mitigating load imbalance from skewed workloads while exposing massive parallelism through GPU-friendly primitives for efficient pattern matching. A small number of deep traversals are delegated to a CPU post-processing stage, which benefits from the CPU’s high frequency and latency-optimized memory hierarchy. Finally, BlazingAML produces fully integrated CPU–GPU code and manages intermediate data movement across the two architectures transparently. 

This design philosophy enables flexibility in pattern expression and optimization. Many common AML patterns, including fanin, fan-out, multi-hop cycles, and their variants, can be expressed as combinations of the same fundamental primitives. Even minor changes in operand selection or stage ordering within the input specification can lead the compiler to generate entirely different optimized execution plans, automatically adapting to the structural characteristics of each pattern without requiring manual code modification. This adaptability, combined with the high-level declarative interface, positions the compiler as a powerful tool for both AML researchers developing new detection algorithms and practitioners deploying production-scale financial crime detection systems. 

## **7 EVALUATION METHODOLOGY** 

## **7.1 Benchmarked dataset** 

We use the state-of-the-art synthetic money laundering transaction dataset generated by IBM research [1]. The dataset (Table 1) consists of a set of realistic, standardized AML datasets that have also been evaluated in prior works [4]. 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 



<!-- Start of picture text -->
(a) Example patterns N2N2N0 Scatter-gather e:N0 4-Cycle àN0N1 @ te:N0@ tN1N3 N1àN1 patterntime_windowstages --  id idopsrcdst_varconstraintsopsrcdstconstraintsoutput ----  skip_if break_if break_if break_if : scatter_phase: gather_phase: for_all: intersection: N0.in_neigh: N2.out_neigh: N1.in_neigh:: scatter_gather: sg_tx: N2: N2 == N1: e0.t > t: e1.t > t: e2.t > t: delta:: Stage 3:Stage 2:Stage 0:Stage 1: (b) Complication example for scatter-gather pattern Edge Feat. UpdateNN02.in.out Filter: len(Place. Acct) > 2Filter: N∀∀2 != N∩ 1 N1.in efind_Start(t – δ, Ne2.start =find_Start(t – δ, Nfor eif len(sg_tx) > 2:efind_Start(t – δ, Nfor e1.start = 0if ( efor e.start = Nif ( eif N1if ( eif e2Update sg_tx features in N0 = e in N122sg_tx.add( (e.t > t)   0 == N in N1.t > t )   .dst == e20.t > t)   2.src.out_neigh: (0.in_neigh: (11.in_neigh:  breakcontinuebreakbreak221.src:.out_neigh).in_neigh)0.in_neigh)1, efor_allintersect2) ) )  )  patterntime_windowstages --  id idopsrcdst_varconstraintsopsrcdstdst_varconstraintsoutput -----  skip_if break_if skip_if break_if break_if : first_leg: second_leg: for_all: intersection: N0.out_neigh: N2.out_neigh: N0.in_neigh:: 4_cycle: cycle_tx: N2: N3: N2 == N0: N3 == N1: e0.t > t: e1.t > t: e2.t > t: delta:: Stage 0:Stage 1:Stage 2:Stage 3: (c) Complication example for 4-cycle pattern NEdge Feat. UpdateN1.out2.out Filter: NFilter: len(Place. Acct) > 0Filter: N∩∀∀23 != N != N01N0.in efind_Start(t – δ, Ne2.start =find_Start(t – δ, Nfor eif len(cycle_tx) > 2:1efind_Start(t – δ, Nfor e.start = 0Nif ( eif Nfor e.start = 1Nif ( eif N3 in N = e1.dst02if ( eif Nif e1.dst == e2.src:3Update cycle_tx features in N = e == N1cycle_tx.add((e,e22.t > t)    in N == N03.t > t )   2 == N2.out_neigh: (00.t > t)   .dst.out_neigh: (10.in_neigh:  0 breakcontinue1continuebreak20.out_neigh).in_neigh)breakcontinue1.out_neigh)intersectfor_all0,e1,e2,e3))) )<br><!-- End of picture text -->

**Figure 5: Compiler-generated pseudo-code for scatter-gather and 4-cycle pattern mining.** 

**Table 1: IBM AML Datasets [1].** 

|**Category**|**Num. of Vertex**|**Num. of Edges**|
|---|---|---|
|LI-Small|705,907|6,924,055|
|LI-Medium|2,032,095|31,251,483|
|LI-Large|2,070,980|176,066,557|
|HI-Small|515,088|5,078,345|
|HI-Medium|2,077,023|31,898,238|
|HI-Large|2,116,168|179,702,229|



## **7.2 State-of-the-art Baselines** 

We compare against two state-of-the-art baselines: 

- GFP [4] from IBM, which mines the money laundering patterns, augments edge features, and uses a gradient boost-based classifier downstream, similar to our pipeline. 

- FraudGT [19], which uses a graph transformer network to detect money laundering instances. 

## **7.3 Hardware Platform Configuration** 

We run CPU baselines on a dual-socket server with two Intel Xeon Platinum 8380 processors, each with 40 physical cores (80 SMT threads) and 8 memory channels with a total of 1TB main memory. We use up to four NVIDIA A40 GPUs to evaluate our design, each with 48GB GDDR6 memory. 

## **8 EVALUATION RESULTS** 

## **8.1 F1 Score Comparison** 

Table 2 shows the F1 score in classifying transactions into moneylaundering or normal transactions. The features extracted from each transaction (edges) are through pattern matching mined from the graph, _, i.e._ , the value of each feature corresponds to the number of instances of each pattern the transaction participates in. Then, the additional features are concatenated and passed to XGB to generate hidden dimensions for training for the first 80% of the timestamped transactions. The inference testing is conducted on the last 20% of the transactions to infer whether a transaction is fraudulent. Table 2 clearly shows that the addition of structural features (Fan → Degree → Cycle → Scatter-Gather) correlates with consistent performance improvements across all datasets, with the full combination (Fan+Degree+Cycle+SG) achieving peak scores in every case. This suggests cumulative benefits from incorporating local node characteristics (Fan/Degree) and global structural patterns (Cycle/SG). Notably, the HI (High-illicit) datasets substantially outperform their LI (Low-illicit) counterparts (e.g., HI-Large:58.1 vs LI-Large:17.8), indicating feature effectiveness scales with the 

density in fraudulent transactions. The clear performance hierarchy (Scatter-Gather > Cycle > Degree > Fan) establishes feature contribution weights that guide future model development. 

**Table 2: F1 scores of different features on datasets. XGB is the baseline with source and destination account ID. Additional features include the number of instances of each pattern (Fan, Degree, Cycle, and Scatter-Gather) each transaction participates in.** 

|**Dataset**|**XGB Only**|**Fan**|**Fan+Degree**|**Fan+Degree+**<br>**Cycle**|**Fan+Degree+**<br>**Cycle+SG**|
|---|---|---|---|---|---|
|LI-Small<br>|10.1|8.1|10.3|12.7|18.4|
|HI-Small<br>|11.1|23.2|38.4|42.6|46.6|
|LI-Med<br>|3.1|4.0|11.0|14.0|21.4|
|HI-Med|10.4|29.3|47.1|50.4|51.1|
|LI-Large|5.4|9.1|16.0|17.7|17.8|
|HI-Large|20.2|39.1|55.7|57.5|58.1|



## **8.2 Mining Performance Comparison** 

Fig. 6, 7, 8, 9 shows the performance study of BlazingAML framework vs. GFP library [4]. BlazingAML is implemented with OpenMP parallelism and evaluated from 1 to 256 threads, as well as CUDA implementation on a single A40 GPU. **Scatter-Gather Pattern Performance** : Compared to 64-thread GFP library implementation in mining Scatter-Gather pattern, BlazingAML already achieves performance comparable to GFP even with CPU single thread, underscoring the performance potentials in legacy python-based libraries. The mining performance scales almost linearly with thread count up to 64 threads, reaching 219–220× speedup over GFP (64 threads) depending on dataset. At 128 and 256 threads, scalability continues, achieving peak improvements of 333× over GFP on the largest datasets. 

**Cycle Pattern Characteristics** : The Cycle pattern demonstrates high performance advantages, with single-thread performance already achieving 10.9× speedup over GFP’s 64-thread baseline. This pattern exhibits a high scaling behavior, reaching 159× improvement with GPU acceleration. The sustained linear scaling up to 64 threads (127× speedup) followed by a performance plateau at higher thread counts suggests that cycle detection benefits significantly from the optimized graph traversal algorithms. 

**Fan Pattern** : For less complex logic such as Fan-in and Fan-out, the GFP library 64-thread achieves comparable performance with 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 



<!-- Start of picture text -->
GFP (64 threads) BlazingAML (4 threads) BlazingAML (32 threads) BlazingAML (256 threads)<br>BlazingAML (1 thread) BlazingAML (8 threads) BlazingAML (64 threads) BLazingAML GPU<br>BlazingAML (2 threads) BlazingAML (16 threads) BlazingAML (128 threads)<br>256<br>64<br>16<br>4<br>1<br>LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large GM<br>Figure 6: BlazingAML Scatter-Gather pattern mining end-to-end throughput normalized to GFP [4].<br>GFP (64 threads) BlazingAML (4 threads) BlazingAML (32 threads) BlazingAML (256 threads)<br>BlazingAML (1 thread) BlazingAML (8 threads) BlazingAML (64 threads) BlazingAML GPU<br>BlazingAML (2 threads) BlazingAML (16 threads) BlazingAML (128 threads)<br>256<br>64<br>16<br>4<br>1<br>LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large GM<br>Figure 7: BlazingAML Cycle pattern mining end-to-end throughput normalized to GFP [4].<br>GFP (64 threads) BlazingAML (4 threads) BlazingAML (32 threads) BlazingAML (256 threads)<br>BlazingAML (1 thread) BlazingAML (8 threads) BlazingAML (64 threads) BlazingAML GPU<br>BlazingAML (2 threads) BlazingAML (16 threads) BlazingAML (128 threads)<br>64<br>16<br>4<br>1<br>0.25<br>LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large GM<br>Figure 8: BlazingAML Fan-in and Fan-out pattern mining combined end-to-end throughput normalized to GFP [4].<br>Blazing AML (1 thread) BlazingAML (8 threads) BlazingAML (64 threads) BlazingAML (256 threads )<br>Blazing AML (2 threads) BlazingAML (16 threads) BlazingAML (128 threads) BlazingAML (GPU)<br>Blazing AML (4 threads) BlazingAML (32 threads)<br>40<br>30<br>20<br>10<br>0<br>LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large<br>333<br>184 219 220 201<br>103<br>54.0<br>27.3<br>13.6<br>7.0<br>SG Perf. vs.<br>1.0<br>GFP 64-thread (x)<br>113 127 122 122 156<br>68.4 86<br>42.1<br>22.8<br>10.9<br>Cycle Perf. vs. 1.0<br>GFP 64-thread (x)<br>26.7<br>7.5 10.9 11.4 8.2 6.4<br>4.4<br>2.3<br>1.0 0.6 1.2<br>Fan Perf. vs.<br>GFP 64-thread (x)<br>33.5<br>22.9 25.6 25.8 24.2<br>12.9<br>7.1<br>Normalized Stack Perf. 1.0 1.8 3.6<br><!-- End of picture text -->

**Figure 9: BlazingAML Stack pattern mining end-to-end throughput normalized to single-thread CPU.** 

BlazingAML less than 8 threads. BlazingAML demonstrates consistent improvement up to 32 threads (11.4×) before experiencing performance degradation at higher thread counts (8.2× at 128 threads). For GPU, the data structure of neighborhood search is further optimized in CUDA to achieve a better speedup for basic patterns such as Fan. 

**Stack Pattern Baseline Comparison** : The Stack pattern evaluation uses a different baseline (1-thread vs. GFP 64-thread), making direct comparison challenging, but reveals excellent parallel scalability up to 64 threads (25.8×) with slight degradation at higher core counts. The GPU achieves 33.5× improvement, demonstrating consistent acceleration across all evaluated patterns. 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 



<!-- Start of picture text -->
GFP (64 threads) BlazingAML (4 threads) BlazingAML (32 threads) BlazingAML (256 threads)<br>BlazingAML (1 thread) BlazingAML (8 threads) BlazingAML (64 threads) BlazingAML GPU<br>BlazingAML (2 threads) BlazingAML (16 threads) BlazingAML (128 threads)<br>64<br>16<br>4<br>1<br>0.25<br>Trovares-10K Trovares-100K Trovares-1M Trovares-10M Trovares-100M GM<br>14.4 23.3 27.5 24.2 21.3 43.7<br>8.2<br>4.4<br>2.3<br>1.0 1.2<br>SG Perf. vs.<br>GFP 64-thread (x)<br><!-- End of picture text -->

**Figure 10: Scalability study of BlazingAML Scatter-Gather pattern mining throughput normalized to GFP [4] on Trovares [30] 10K – 100M edge dataset.** 

The pattern mining results demonstrate both strong parallel scalability and effective load distribution in the generated code across diverse graph mining workloads. The results confirm that BlazingAML’s domain-specific compiler produces significantly more efficient code than hand-optimized existing baseline implementations. BlazingAML scales to hundreds of cores, though with pattern-specific behaviors that suggest intelligent workload-aware optimization strategies. The performance benefits hold consistently across all dataset categories (low vs. high interconnectivity, small vs. large), demonstrating the generality of the compiler design. While the GPU backend provides strong performance, high-core-count CPUs deliver competitive speedups, giving users flexibility in choosing the hardware platform based on the specific pattern mining requirements. 

## **8.3 Scalability Study** 

Figure 10 evaluates the scalability of BlazingAML against the baseline GFP implementation using synthetic graphs generated by Trovares [30] spanning five orders of magnitude in size, from 10K to 100M edge. On the smallest dataset (Trovares-10K), BlazingAML achieves a 21.8× speedup with 32-thread execution. On larger datasets, BlazingAML achieves a 40.8× and 27.8× speedup in Trovares1M and Trovares-10M. The average speedup of BlazingAML GFP reaches a remarkable 27.5× speedup using the same 64 threads, indicating excellent parallel scalability of our scatter-gather pattern implementation. Our multi-threading analysis reveals consistent scaling behavior across different thread counts. The performance improvements scale nearly linearly from 1 to 64 threads, with the 64-thread configuration achieving speedups of 19.5× and 27.5× on Trovares-10M and 100M. 

BlazingAML GPU implementation demonstrates a 24.4× speedup on Trovares-100M compared to the baseline GFP. This substantial improvement highlights the effectiveness of our CUDA implementation in exploiting the massive parallelism inherent in scatter-gather operations on large-scale graphs. The scalability trends clearly favor BlazingAML as dataset size increases. This behavior is consistent with our design philosophy of optimizing for large-scale graph mining workloads where the overhead of our compilation framework is amortized across substantial computational work. 

## **8.4 F1 Score of Money-Laundering Predictions** 

**Why using F1 score?** Money laundering is inherently an imbalanced problem since laundering transactions only account for a 

very small fraction of the total transactions in real-world financial systems. In such an imbalanced inference setting, the F1 score is widely adopted as the primary evaluation metric [4, 27] because it balances precision and recall and prevents models from achieving deceptively high accuracy by simply predicting the majority “non-laundering” class. To further highlight the imbalance in real AML datasets, Table 3 shows the confusion matrix obtained on the HI-Small dataset after applying all mined features (Fan, Degree, Cycle, and SG) and training the XGB classifier. Even in this relatively high–laundering-occurrence dataset, the TN count remains overwhelmingly dominant. This concrete example reinforces why the F1 score is the most appropriate metric for evaluating laundering prediction performance in such imbalanced conditions. 

Figure 11 presents the F1 score of predicting whether a transaction is a laundering transaction under different feature configurations. In the baseline setting, only the raw transactional information (transaction ID, source account, and destination account) is used as features for XGB Boost. As additional features derived from pattern mining are included—such as the number of Scatter-Gather (SG) structures an edge participates in—the F1 score increases significantly. This demonstrates that structural mining features contribute strong discriminative power in identifying laundering behaviors. 

BlazingAML intentionally keeps the set of mining outputs identical to the GFP library to ensure full compatibility, while providing a more efficient and flexible execution engine. Across all datasets, Degree, Cycle, and SG features consistently lead to meaningful improvements in F1 score, confirming their effectiveness in capturing laundering-related topological signals. 

An additional trend observed in Figure 11 is that the HI datasets (with higher laundering occurrence) achieve substantially higher F1 scores than the LI datasets. This is expected: in datasets with more positive instances, the classifier receives stronger supervision and can learn laundering-related behaviors more reliably. 

## **8.5 Comparison with FraudGT** 

FraudGT [19] uses a graph transformer model and achieves state-ofthe-art performance in detecting fraudulent activities, also demonstrating high throughput compared to existing methods. We compare BlazingAML and FraudGT on their achieved F1 score and mining throughput in detecting money laundering patterns. Tab. 4 shows the F1 score of the two mining framework. Note that we have verified that BlazingAML achieves the same feature mining output as GFP [4]. Therefore, the F1 score difference roots from 

Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, and Nishil Talati 

**Table 3: Confusion matrix of laundering prediction on HI-Small after applying all BlazingAML features. The matrix demonstrates the extreme class imbalance even in a high-laundering dataset.** 



<!-- Start of picture text -->
Predicted Laundering Predicted Non-Laundering<br>Actual Laundering 558 1239<br>Actual Non-Laundering 31 1013841<br>XGB only Fan Fan + Degree Fan + Degree + Cycle Fan + Degree + Cycle + SG<br>70<br>60<br>50<br>40<br>30<br>20<br>10<br>0<br>LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large<br>Workload<br>F1 Score<br><!-- End of picture text -->

**Figure 11: F1 score of BlazingAML when using different shapes as mining features. The F1 score increases as more features (the number of participating shapes for each transactional edge) are included and then used for training and inference via the XGB Boost library. BlazingAML preserves the output quality of the GFP library while providing a faster and more flexible mining framework.** 

**Table 4: F1 scores of FraudGT [19] and BlazingAML across IBM datasets.** 

scales more efficiently than the quadratic complexity inherent in Transformer architectures. 



<!-- Start of picture text -->
Method LI- HI- LI- HI- LI- HI-<br>Small Small Medium Medium Large Large<br>FraudGT 28.6 69.6 24.0 62.3 11.0 54.3<br>BlazingAML 18.4 46.4 21.4 51.1 17.8 58.1<br>FraudGT BlazingAML (128 threads)<br>400<br>300<br>200<br>100<br>0 LI-Small HI-Small LI-Medium HI-Medium LI-Large HI-Large<br>per seconds<br>Kilo Transaction<br><!-- End of picture text -->

**Figure 12: Performance study of BlazingAML compared with FraudGT. BlazingAML processes 4.9** × **higher number of edges per second on average.** 

## **9 CONCLUSION** 

This paper presented BlazingAML, a scalable anti-money laundering (AML) system that advances the way financial institutions design and deploy pattern detection algorithms. At the core of our approach is a _multi-stage specification technique_ that captures both the structural and temporal complexities of money laundering schemes, moving beyond the limitations of traditional rigid pattern-matching techniques. A domain-specific compiler bridges AML expertise and high performance deployment, automatically generating optimized implementations without requiring low-level programming knowledge. Our evaluation demonstrates that BlazingAML achieves substantial speedups on both CPUs and GPUs while preserving high detection accuracy, making sophisticated pattern mining both practical and scalable. 

the ML framework difference (feature extension + XGB in [4] and Transformer-based model in [19]). Fig. 12 compares the mining throughput of BlazingAML with FraudGT. On average, BlazingAML 128-thread implementaion achieves 4.9× higher throughput, which corroborates with the intuition that feature mining+XGB achieving a much more efficient solution than Transformer-based models. 

The performance gap is particularly pronounced on larger datasets, where BlazingAML processes between 300-400 thousand transactions per second compared to FraudGT’s 50-100 thousand transactions per second. On average across all configurations, BlazingAML’s 128-thread implementation achieves a 4.9× higher throughput than FraudGT. This substantial performance improvement can be attributed to the fundamental algorithmic differences between the two approaches. This design choice proves particularly effective for fraud detection scenarios where rapid transaction processing is critical, as our feature extraction and gradient boosting approach 

BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining 

## **REFERENCES** 

- [1] Erik Altman, Jovan Blanuša, Luc Von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. 2024. Realistic synthetic financial transactions for anti-money laundering models. _Advances in Neural Information Processing Systems_ 36 (2024). 

- [2] BBC. 2018. Commonwealth Bank offers to pay record fine in laundering case. Accessed: 2025-08-13. 

- [3] Jiang Bian, Abdullah Al Arafat, Haoyi Xiong, Jing Li, Li Li, Hongyang Chen, Jun Wang, Dejing Dou, and Zhishan Guo. 2022. Machine learning in real-time Internet of Things (IoT) systems: A survey. _IEEE Internet of Things Journal_ 9, 11 (2022), 8364–8386. 

- [4] Jovan Blanuša, Maximo Cravero Baraja, Andreea Anghel, Luc von Niederhäusern, Erik Altman, Haris Pozidis, and Kubilay Atasu. 2024. Graph Feature Preprocessor: Real-time Extraction of Subgraph-based Features from Transaction Graphs. _arXiv preprint arXiv:2402.08593_ (2024). 

- [5] Mário Cardoso, Pedro Saleiro, and Pedro Bizarro. 2022. LaundroGraph: Selfsupervised graph representation learning for anti-money laundering. In _Proceedings of the Third ACM International Conference on AI in Finance_ . 130–138. 

- [6] Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In _Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining_ . 785–794. 

- [7] Dawei Cheng, Yujia Ye, Sheng Xiang, Zhenwei Ma, Ying Zhang, and Changjun Jiang. 2023. Anti-money laundering by group-aware deep graph learning. _IEEE Transactions on Knowledge and Data Engineering_ 35, 12 (2023), 12444–12457. 

- [8] Bruno Deprez, Toon Vanderschueren, Bart Baesens, Tim Verdonck, and Wouter Verbeke. 2025. Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation. arXiv:2405.19383 [cs.SI] https: //arxiv.org/abs/2405.19383 

- [9] Ahmad Naser Eddin, Jacopo Bono, David Aparício, David Polido, João Tiago Ascensão, Pedro Bizarro, and Pedro Ribeiro. 2022. Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs. arXiv:2112.07508 [cs.LG] https://arxiv.org/abs/2112.07508 

- [10] Béni Egressy, Luc Von Niederhäusern, Jovan Blanuša, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. 2024. Provably powerful graph neural networks for directed multigraphs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 38. 11838–11846. 

- [11] Europol. 2017. From Suspicion to Action – Converting Financial Intelligence into Greater Operational Impact. https://www.europol.europa.eu/publicationsdocuments/suspicion-to-action-converting-financial-intelligence-greateroperational-impact 

- [12] Jiani Fan, Ziyao Liu, Hongyang Du, Jiawen Kang, Dusit Niyato, and KwokYan Lam. 2024. Improving security in IoT-based human activity recognition: a correlation-based anomaly detection approach. _IEEE Internet of Things Journal_ (2024). 

- [13] Jiani Fan, Lwin Khin Shar, Ruichen Zhang, Ziyao Liu, Wenzhuo Yang, Dusit Niyato, Bomin Mao, and Kwok-Yan Lam. 2025. Deep Learning Approaches for Anti-Money Laundering on Mobile Transactions: Review, Framework, and Directions. _arXiv preprint arXiv:2503.10058_ (2025). 

- [14] FATF. 2022. _Virtual assets: red flag indicators_ . Technical Report. Financial Action Task Force (FATF). 

- [15] Kasra Jamshidi, Rakesh Mahadasa, and Keval Vora. 2020. Peregrine: a patternaware graph mining system. In _Proceedings of the Fifteenth European Conference on Computer Systems_ . 1–16. 

- [16] Kasra Jamshidi, Mugilan Mariappan, and Keval Vora. 2022. Anti-vertex for neighborhood constraints in subgraph queries. In _Proceedings of the 5th ACM SIGMOD Joint International Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA)_ . 1–9. 

- [17] Fredrik Johannessen and Martin Jullum. 2023. Finding Money Launderers Using Heterogeneous Graph Neural Networks. arXiv:2307.13499 [cs.LG] https://arxiv. org/abs/2307.13499 

- [18] KPMG. 2014. Global Anti-Money Laundering Survey 2014. 

- [19] Junhong Lin, Xiaojie Guo, Yada Zhu, Samuel Mitchell, Erik Altman, and Julian Shun. 2024. FraudGT: a simple, effective, and efficient graph transformer for financial fraud detection. In _Proceedings of the 5th ACM International Conference on AI in Finance_ . 292–300. 

- [20] Wai Weng Lo, Gayan K Kulatilleke, Mohanad Sarhan, Siamak Layeghy, and Marius Portmann. 2023. Inspection-L: self-supervised GNN node embeddings for money laundering detection in bitcoin. _Applied Intelligence_ 53, 16 (2023), 19406–19417. 

- [21] Patrick Mackey, Katherine Porterfield, Erin Fitzhenry, Sutanay Choudhury, and George Chin. 2018. A chronological edge-driven approach to temporal subgraph isomorphism. In _2018 IEEE international conference on big data (big data)_ . IEEE, 3972–3979. 

- [22] Neo4j, Inc. 2025. _Cypher Query Language_ . https://neo4j.com/docs/cyphermanual/current/introduction/ Declarative graph query language. 

- [23] Ashwin Paranjape, Austin R Benson, and Jure Leskovec. 2017. Motifs in temporal networks. In _Proceedings of the tenth ACM international conference on web search and data mining_ . 601–610. 

- [24] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao Schardl, and Charles Leiserson. 2020. Evolvegcn: Evolving graph convolutional networks for dynamic graphs. In _Proceedings of the AAAI conference on artificial intelligence_ , Vol. 34. 5363–5370. 

- [25] Stephen Schneider. 2004. Money laundering in Canada: a quantitative analysis of Royal Canadian Mounted Police cases. _Journal of Financial Crime_ 11, 3 (2004), 282–291. 

- [26] Shivani Singh, Razia Sulthana, Tanvi Shewale, Vinay Chamola, Abderrahim Benslimane, and Biplab Sikdar. 2021. Machine-learning-assisted security and privacy provisioning for edge computing: A survey. _IEEE Internet of Things Journal_ 9, 1 (2021), 236–260. 

- [27] Kiwhan Song, Mohamed Ali Dhraief, Muhua Xu, Locke Cai, Xuhao Chen, Arvind, and Jie Chen. 2024. Identifying Money Laundering Subgraphs on the Blockchain. arXiv:2410.08394 [cs.LG] https://arxiv.org/abs/2410.08394 

- [28] Nishil Talati, Haojie Ye, Sanketh Vedula, Kuan-Yu Chen, Yuhan Chen, Daniel Liu, Yichao Yuan, David Blaauw, Alex Bronstein, Trevor Mudge, et al. 2022. Mint: An accelerator for mining temporal motifs. In _2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1270–1287. 

- [29] Maria Paola Tatulli, Tommaso Paladini, Mario D’Onghia, Michele Carminati, and Stefano Zanero. 2023. HAMLET: A transformer based approach for money laundering detection. In _International Symposium on Cyber Security, Cryptology, and Machine Learning_ . Springer, 234–250. 

- [30] Trovares. 2024. Temporal Triangles xGT Datasets. https://datasets.trovares.com/ synthetic/TT/index.html#pre-generated-datasets. [Accessed 25-12-2024]. 

- [31] Aashma Uprety and Danda B Rawat. 2020. Reinforcement learning for iot security: A comprehensive survey. _IEEE Internet of Things Journal_ 8, 11 (2020), 8693–8706. 

- [32] Mark Weber, Jie Chen, Toyotaro Suzumura, Aldo Pareja, Tengfei Ma, Hiroki Kanezashi, Tim Kaler, Charles E Leiserson, and Tao B Schardl. 2018. Scalable graph learning for anti-money laundering: A first look. _arXiv preprint arXiv:1812.00076_ (2018). 

- [33] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I Weidele, Claudio Bellei, Tom Robinson, and Charles E Leiserson. 2019. Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. _arXiv preprint arXiv:1908.02591_ (2019). 

- [34] Ernst & Young. 2020. Economic crime in a digital age. https://assets.ey. com/content/dam/ey-sites/ey-com/en_gl/topics/assurance/assurance-pdfs/eyeconomic-crime-digital-age.pdf. 

- [35] Yichao Yuan, Haojie Ye, Sanketh Vedula Wynn Kaza, and Nishil Talati. 2023. Everest: GPU-Accelerated System For Mining Temporal Motifs. _arXiv preprint arXiv:2310.02800_ (2023). 

- [36] Ruichen Zhang, Hongyang Du, Yinqiu Liu, Dusit Niyato, Jiawen Kang, Zehui Xiong, Abbas Jamalipour, and Dong In Kim. 2024. Generative AI agents with large language model for satellite networks via a mixture of experts transmission. _IEEE Journal on Selected Areas in Communications_ (2024). 

