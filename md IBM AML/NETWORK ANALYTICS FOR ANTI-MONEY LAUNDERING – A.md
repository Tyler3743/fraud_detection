# NETWORK ANALYTICS FOR ANTI-MONEY LAUNDERING – A SYSTEMATIC LITERATURE REVIEW AND EXPERIMENTAL EVALUATION 

A PREPRINT 

**Bruno Deprez**<sup>_∗_</sup> 

**Toon Vanderschueren** 

KU Leuven KU Leuven University of Antwerp - imec University of Antwerp - imec 

**Bart Baesens** 

KU Leuven University of Southampton 

**Tim Verdonck** 

**Tim Verdonck Wouter Verbeke** University of Antwerp - imec KU Leuven KU Leuven 

## **ABSTRACT** 

Money laundering presents a pervasive challenge, burdening society by financing illegal activities. The use of network information is increasingly being explored to effectively combat money laundering, given it involves connected parties. This led to a surge in research on network analytics for antimoney laundering (AML). The literature is, however, fragmented and a comprehensive overview of existing work is missing. This results in limited understanding of the methods to apply and their comparative detection power. This paper presents an extensive and unique literature review, based on 97 papers from _Web of Science_ and _Scopus_ , resulting in a taxonomy following a recently proposed fraud analytics framework. We conclude that most research relies on expert-based rules and manual features, while deep learning methods have been gaining traction. This paper also presents a comprehensive framework to evaluate and compare the performance of prominent methods in a standardized setup. We compare manual feature engineering, random walk-based, and deep learning methods on two publicly available data sets. We conclude that (1) network analytics increases the predictive power, but caution is needed when applying GNNs in the face of class imbalance and network topology, and that (2) care should be taken with synthetic data as this can give overly optimistic results. The open-source implementation facilitates researchers and practitioners to extend this work on proprietary data, promoting a standardized approach for the analysis and evaluation of network analytics for AML. 

**_Keywords_** Fraud Analytics _·_ Anti-Money Laundering _·_ Network Analytics _·_ Literature Review 

## **1 Introduction** 

Money laundering is the process of concealing illegally obtained funds by passing them through a complex sequence of transactions, so the money can be used to fund further criminal activities, e.g., drug trafficking, and terrorism [75]. The United Nations Office on Drugs and Crime [126] estimates that around 2% to 5% of global GDP is laundered worldwide, amounting yearly to USD 2 trillion. 

In contrast to credit card fraud, money laundering occurs over a longer duration and requires analysis of multiple transactions for detection. It is actively concealed, since sustained laundering is needed to process continuous money streams. Credit card fraud, on the other hand, aims to get as much money out as fast as possible. Hence, the specific characteristics of money laundering merit dedicated research attention. 

> _∗_ Correspondening author: `bruno.deprez@kuleuven.be` 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Since this money needs to enter the financial system at some point [75], legislators have put stringent rules on financial institutions [15] for reporting suspicious transactions. Despite record fines and mounting pressure by regulators, money laundering represents a growing problem [135]. 

Money laundering typically involves three stages [126, 75]: placement, where illicit funds enter the financial system; layering, where transactions obscure their origin; and integration, where the funds re-enter the economy through seemingly legitimate means, completing the laundering process. 

Anti-money laundering practices require constant monitoring of clients and transactions. These practices are, however, largely ineffective [35], with Europol estimating only 1.1% of criminal profits in the EU being confiscated [38]. 

Effective money laundering involves multiple parties, with individual transactions appearing normal [15]. Therefore, network analytics is essential [15, 10]. This need was already mentioned three decades ago by Senator et al. [113], where the authors used visualisations to support expert investing. 

The increasing attention of the past years has resulted in a growing body of literature on network analytics for anti-money laundering, but simultaneously resulted in a lack of comprehensive overviews. Therefore, as a first step, this paper aims to supplement the literature by providing a comprehensive literature review of network analytics (NA) for anti-money laundering, covering 97 papers analysed according to multiple dimensions. 

Due to the fragmented nature of the literature there is limited insight into which methods perform best. This fragmentation and lack of comparison is apparent in the fact that many of the introduced methods are not tested against any baselines. Furthermore, researchers are slow to adopt the latest network analytics methods and graph neural network in anti-money laundering. Therefore, the second goal of this work is to present a structured experimental set-up to evaluate the most prominent methods. 

In summary, our main contributions are four-fold: 

- We provide an extensive and critical literature review on network analytics for anti-money laundering, to gain comprehensive and deep insights, and provide directions for future research. 

- We benchmark a range of state-of-the-art network analytics methods for anti-money laundering, comparing the performance on two open-source data sets. 

- We provide insights into the specific challenges of network methods based on the topology of the network and the data generating process. 

- We implement the methods in a uniform manner and facilitate replication of the presented results by providing public access to the code at `https://github.com/B-Deprez/AML_Network` , aiming to promote a standardized approach towards the analysis and adoption of NA for AML and to encourage further research. 

This paper is structured as follows. We present the methodology for the literature review in Section 2, with Section 3 presenting the results and analysis. Section 4 describes the the empirical evaluation, with the results presented in Section 5. Overall conclusions and directions for future work are presented in Section 6. 

## **2 Literature Review** 

The first part of this work covers the current literature. A summary is given according to multiple categories. Additional analyses are provided for the top-cited papers. The aim is to provide answers to the following research questions: 

- What are the most prominent learning methods in the literature and how have these evolved over time? 

- How are methods evaluated, using what data sets and evaluation metrics? 

- What is the setting, objectives and challenges discussed in the top-cited papers? 

### **2.1 Methodology** 

We conducted the literature search using the queries _“graph analy*” AND “money launder*”_ and _“network analy*” AND “money launder*”_ on WoS and Scopus, chosen for their high completeness and research credibility [96, 14]. The search is limited to English-language papers published before 2023, excluded recent publications to ensure adequate time for community evaluation. 

The selection process included a title and abstract scan, followed by a full-text review. Papers were excluded if they focused on legal aspects or lacked substantial application of network analytics to anti-money laundering (AML). The 

2 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

final set of papers is classified according to seven criteria as summarised in Table 1: publication data, method type, modelling approach, evaluation metrics, main objectives, data type, and network characteristics. 

Our review provides both a broad analysis of trends across all selected papers and an in-depth focus on top-cited works (top 10% annually and overall, per Google Scholar). This dual approach highlights general trends and the state-of-the-art (SOTA) in the field. Finally, we examine relevant review papers to contextualize our findings and demonstrate how our work extends existing literature. 

Table 1: The (sub-)categories with explanation for the classification of the literature. 

|**Category**|**Sub-Category**|**Defnition**|
|---|---|---|
|_Publication Data_|Title<br>Journal/Conference<br>Year<br>Review Paper<br>Bitcoin/Crypto|The full title of the paper<br>Where the paper is published<br>The year of publication<br>Indicating if it is a review paper<br>Indicating if it deals with crypto currency|
|_Method Type_|Supervised<br>Unsupervised<br>Semi-Supervised<br>Mixed<br>Visualisation|Application of supervised method<br>Application of unsupervised method<br>Application of semi-supervised method<br>Methods presented across different learning methods<br>Application of visualisation method|
|_Modelling Method_|Rule-based<br>Manual features<br>Walk-based<br>Shallow representation<br>Deep representation<br>Correlation-based<br>Logistic regression<br>Tree-based<br>SVM-based<br>Neural networks<br>Anomaly detection<br>Clustering|Application of specifc rules or cut-offs<br>Network features having a exact defnition<br>Network analysis based on (random) walks<br>Embeddings typically through matrix factorization or random<br>walks, without employing deep neural networks<br>Embeddings based on deep neural networks, e.g., GNNs<br>Analysis of correlation of network features with target<br>Application of logistic regression<br>Application of tree-based methods<br>Application of support vector machines<br>Learning using neural networks<br>Application of anomaly or outlier detection<br>Features based on community detection or clustering of features|
||Accuracy<br>Precision<br>Recall<br>|Accuracy<br>The precision<br>The recall<br>|
||F1|The F1 or micro-F1 score|
|_Evaluation Metric_|TPR<br>FPR|The true positive rate<br>The false positive rate|
||AUC-ROC<br>AUC-PR<br>Time|The area under the ROC curve<br>The area under the precision-recall curve<br>The execution time|
||Client classifcation|Detection of suspicious entities|
|_Objective_|Transaction classifcation<br>Community detection<br>Flow/Chain detection|Detection of suspicious transactions<br>Detection of suspicious groups of clients<br>Detection of suspicious combination of transactions|
||Proprietary|Usage of confdential data|
|_Data_|Open-source<br>Synthetic|Usage of freely available data<br>Usage of synthetically generated data|
|_Network_|Multiple<br>Inverse/undirected|Usage of multiple networks<br>Additional use of network with directions removed or reverted|



3 













Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Number of papers per year<br>Full Search<br>40<br>Relevant Papers<br>35<br>30<br>25<br>20<br>15<br>10<br>5<br>0<br>1995 2000 2005 2010 2015 2020<br><!-- End of picture text -->

Figure 2: The number of papers incorporated in the full query and after filtering per year. 

Table 4: Journals/conferences that have two or more publications in scope of this review. 

|**Journal/Conference Title**|**Number**|
|---|---|
|Journal of Money Laundering Control|7|
|International Conference on Advances in Social Network Analysis and Mining, ASONAM|3|
|CEUR Workshop Proceedings|2|
|IEEE International Conference on Data Mining Workshops|2|
|IEEE International Conference on Big Data (Big Data)|2|
|Journal of Physics: Conference Series|2|
|Federated Conference on Computer Science and Information Systems (FedCSIS)|2|
|EPJ Data Science|2|
|Information Sciences|2|
|IEEE Access|2|
|International Conference on Machine Learning Technologies|2|



Crypto-related research accounts for 26 papers (26 _._ 5% of the total), analysed separately in Appendix A due to its distinct AML setting, data characteristics, and anonymity challenges. These differences are evident in the corresponding plots. 

Appendix B presents results for the 20 top-cited papers, summarized in Table 6. The category distribution aligns with the overall literature, with 25% focusing on cryptocurrency, closely matching the full-scope percentage. 

## **3 Discussion on the Literature** 

### **3.1 Review Papers** 

We identified 11 review papers, which we discuss first to illustrate how our work complements and extends the existing literature. 

Ngai et al. [99], Kurshan and Shen [73] and Lokanan [82] cover general financial fraud, with less emphasis on antimoney laundering or graph learning. Ngai et al. [99] construct a classification framework, but include only one paper on money laundering, which is the only paper on network analytics. Kurshan and Shen [73] give a high-level overview of the general difficulties of machine learning solutions for fighting financial crime. They noted that network-based methods have a lot of potential to process many transactions handled by financial institutions. The paper is closer to a discussion paper than to a systematic literature review. The scope of the review by Lokanan [82] is limited, covering only visualisation methods for outlier detection. 

8 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Other review papers deal specifically with AML [74, 112, 42]. Kute et al. [74] give an overview of the methods applied in the literature, but their main topic is interpretability. Semenov et al. [112] discuss the different methods applied to AML. It touches upon different network methods, but it is not a systematic literature review. The framework by Gao and Ye [42] again deals with a wider array of methods, with the authors concluding that network analytics is the most promising. However, the number of papers on network analytics for AML included is limited. 

Gerbrands et al. [47] look at the effects of money laundering policies, by measuring the change in network features when a major AML law is introduced. It analyses cluster size and the degree, closeness and betweenness centrality. Hence, the scope is different, since we focus on enhancing anti-money laundering using network features. 

A significant number of review papers deal with crypto currency [63, 27, 4, 57]. Irwin and Turner [63] describe challenges in detecting money laundering on the blockchain. This view results in a narrow scope with a total of 45 unique references. These different challenges are partially addressed by Alarab et al. [4] and Han et al. [57]. The experiment by Alarab et al. [4] compares different supervised machine learning methods, with the statistics of the ego-network of the different nodes in the Bitcoin transaction network [37, 134] as features. 

Other papers cover a specific money laundering method and propose tailor-made solutions. Han et al. [57] discuss the application of anomaly detection, based on crime-specific transaction patterns. The paper also covers Ponzi schemes and blackmail campaigns. Their scope is more limited, with a total of 33 references. Similarly, Day [27] gives an overview of different characteristics of financial crime, and uses these to construct knowledge graphs for building legal cases. 

We extend on prior work by conducting the first systematic review of network analytics methods for anti-money laundering, including both supervised and unsupervised methods. Our scope spans traditional centrality measures to advanced graph neural networks, and addresses money laundering in both fiat and crypto currency. 

### **3.2 Full Scope of Papers** 

This section summarises the 83 non-review papers. Although network analytics for anti-money laundering has been researched for a long time, Figure 2 illustrates this increasing research interest only after 2010. The average increase in output is 61% year-on-year for the papers in scope. 

### **3.2.1 Methods.** 

The methods in the literature are supervised, unsupervised, semi-supervised and visualisations. The most popular method is unsupervised learning, followed by supervised learning and visualisations, as shown in Table 3. The popularity of unsupervised learning may be due to money laundering being very uncommon, resulting in highly skewed data sets with very few labelled observations [3]. Due to limited resources to investigate the massive amounts of transactions, most transactions are left unlabelled [129, 106]. In addition, criminals try to evade detection by continuously changing tactics [10, 106], making historically patterns less relevant for future predictions [15]. The performance of (supervised) models trained on historical data can therefore decrease over time [14]. 

Table 3 indicates that half of the papers rely on manual feature engineering. These manual features include classic network centrality measures (degree, betweenness etc.) and summary statistics of the transactions in the nodes’ ego-network. The second most popular method is clustering, which is used broadly in two ways. First, clustering for visualisations mitigates visual cluttering and allows the the selection detail shown. Second, clusters are used for feature engineering. Next to manual feature engineering and clustering, rule-based methods are still frequently used. This illustrates that the current literature heavily relies on expert knowledge. Hence, there is an opportunity to further develop data-driven machine learning methods in anti-money laundering. 

The evolution of methods used over time, as presented in Figure 3, shows that the earliest research relied almost exclusively on rule-based systems and manual feature engineering. These remain popular approaches to this day. From 2018 onward, we see the application of deep learning methods based on shallow representations and graph neural networks. Especially the use of GNNs has seen a spike in 2022. 

The downstream methods used for classification have also evolved. Popular methods are tree-based, clustering, anomaly detection and logistic regression. SVM-based methods were popular until 2019, while neural networks were only used in AML as of 2020. These are often recurrent neural networks to incorporate the temporal aspect [140, 149, 95, 79]. 

Key to these methods is how the authors define the network(s) used. The construction of multiple networks, e.g., same entities but using different relations, to capture richer data is present in 18 papers. Of these papers, five used the undirected network or the one with direction reversed in tandem with the original network. It allows for information propagate from, e.g., the transaction’s receiver back to the sender. 

9 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Modelling Method per Year (Detailed)<br>Rule-based Manual features Walk-based<br>10 Rule-based Manual features Walk-based<br>5<br>0<br>Shallow repr. Deep repr. Correlation-based<br>10 Shallow repr. Deep repr. Correlation-based<br>5<br>0<br>Tree-based SVM-based Neural Networks<br>10 Tree-based SVM-based Neural Networks<br>5<br>0<br>Anomaly detection Clustering Logistic regression<br>10 Anomaly detection Clustering Logistic regression<br>5<br>0<br>Year Year Year<br>199520072010201120122013201420152016201720182019202020212022 199520072010201120122013201420152016201720182019202020212022 199520072010201120122013201420152016201720182019202020212022<br><!-- End of picture text -->

Figure 3: The evolution of the model building blocks over the years. Since there are many modelling methods, the graphs are split. 

### **3.2.2 Objectives.** 

Table 2 shows that the AML method’s objective is most often detecting suspicious clients, followed by detecting suspicious transactions, money flows and communities. Client classification combines client characteristics with payments made over a longer period. Since money laundering is done over a longer period, with the aim to make the individual transaction appear normal, models trained on a series of transactions, leveraging an individual’s behaviour, can obtain higher performance. 

Figure A3 shows that methods for crypto-currency are most often introduced for suspicious transaction detection. Due to the pseudo-anonymity, it is much harder to know which wallets belong to what person [152, 145], making it less feasible to build profiles. Therefore, researchers in AML have also looked at de-anonymisation of the wallets [152, 145, 48]. They use network features to cluster wallets together to see if they belong to the same person or organization. 

In recent years, more studies have been done on flow/chain detection, tracking payments over multiple steps/people. Although this requires more computing power, it more closely resembles reality [126, 75]. Hence, these methods are becoming more widely used. 

### **3.2.3 Evaluation.** 

Table 2 shows that research mostly relies on the recall, _F_ 1-score, precision and time. Figure 4 illustrates that training/prediction time was a popular method in earlier studies, but relatively few papers use time in more recent work, which might be a consequence of increased computing power. Additionally, the popularity of the precision, recall and _F_ 1-score is a recent and increasing trend in the literature. 

We remark two surprising results. First, only one paper used the area under the precision-recall curve (AUC-PR). Due to the high label imbalanced, this metric is shown to be more appropriate than the area under the ROC curve (AUCROC) [26, 102]. Therefore, we would expect it to be more widely used. Second, some papers solely reported accuracy [31, 141, 48, 143]. Given the high class imbalance, accuracy fails in providing an appropriate performance assessment. 

### **3.2.4 Data.** 

Table 2 shows that most studies use open-source data, with proprietary data used in almost 50% of the papers. Only a minority of papers report their findings using synthetic data. There is, however, a major difference between crypto- and other research. Figure A1 shows that crypto-research is almost exclusively done on open-source data. This results from 

10 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Evaluation Metrics per Year (Detailed)<br>Accuracy Precision Recall<br>7.5 Accuracy Precision Recall<br>5.0<br>2.5<br>0.0<br>F1 TPR FPR<br>7.5 F1 TPR FPR<br>5.0<br>2.5<br>0.0<br>AUROC AUPRC Time<br>7.5 AUROC AUPRC Time<br>5.0<br>2.5<br>0.0<br>Year Year Year<br>199520072010201120122013201420152016201720182019202020212022 199520072010201120122013201420152016201720182019202020212022 199520072010201120122013201420152016201720182019202020212022<br><!-- End of picture text -->

Figure 4: The evolution of the evaluation metrics over the years. Since there are many metrics, the graphs are split. 

the blockchain’s freely accessible distributed ledger [134]. On the other hand, _classic_ transaction data cannot be shared, due to privacy reasons, resulting in half the data sets being proprietary. 

Following this last remark, one would expect to have (almost) no research done on open-source data sets for fiat currencies. Figure A1 shows that almost 30% of non-crypto related papers use open-source data. These papers can be categorised in four groups: 

- Methods that are evaluated on non-financial network data, whether or not next to proprietary/synthetic money laundering data: Zhou et al. [151], Ovelgönne et al. [101], Micale et al. [92], Prado-Romero and Gago-Alonso [108] 

- Methods evaluated on government data, assumed or stated to be open-source: Didimo et al. [29], Imanpour et al. [62], Velasco et al. [131], Bahulkar et al. [11], Malm and Bichler [87] 

- Methods constructed to analyse leaked documents/scandals, e.g., the Panama papers: Joaristi et al. [68, 69], Magalingam et al. [84], Adriaens et al. [1], Winiecki et al. [136], Helmy et al. [59], Cheong and Si [23] 

- Other: Nandhini and Das [97], Bhalerao et al. [13], Ça˘glayan and Bahtiyar [17] 

For the _Other_ category, we have the following. Nandhini and Das [97] say that the data is publicly available, without giving any more information. Bhalerao et al. [13] scraped their data from Hack Forums. Finally, Ça˘glayan and Bahtiyar [17] use a Kaggle data set based on synthetic data. We categorise it as open-source, since it is made publicly available. 

Table 5 summarises the open-source data sets. The most widely used are almost all crypto-currency data sets. Raw Bitcoin data is used by ten papers. This raw data is often enhanced using data from other platforms. One such platform is WalletExplorer, which is used by three papers. It is less relevant today, since the data has not been maintained since 2016. Another recent data set on Bitcoin is the Elliptic data sets [37, 134]. Although only published in 2019, it has already been used in six experiments. After Bitcoin, Ethereum seems to be the second most popular crytpo-currency in AML. 

The most used data set for fiat currency is the Czech Financial Data set (CFD). This data set does not provide any labels. Papers using it introduce additional, synthetic money laundering patterns to test their methods [78, 133]. This is not ideal, as this hinders comparison across research and can introduce detection bias. 

As mentioned above, AML methods are also being evaluated on leaked data like the ICIJ Offshore Leaks Database (including, e.g., the Panama-papers) and the Enron data set. 

### **3.3 Top Cited Papers** 

A final step in our discussion is an in-depth analysis of the top-cited papers. These papers are well received by the scientific community and are likely of high quality. We select the 10% most cited papers per year, extended with the 10% most cited papers overall (see Section 2.1). This results in 20 papers as listed in Table 6. The analysis is done in 

11 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 5: The open-source data sets used in the different papers. 

|**Data Set**<br>|**Number**|**References**|
|---|---|---|
|Raw bitcon|10|[88], [137], [105], [138], [125], [110], [145], [121],<br>[109], [146]|
|Elliptic Data set|6|[5], [140], [95], [119], [76], [79]|
|Ethereum XBlock.pro|4|[121], [152], [65], [66]|
|ICIJ Offshore Leaks Database|4|[69], [44], [68], [136]|
|WalletExplorer|3|[137], [125], [145]|
|Czech Financial Data set (CFD)|2|[78], [133]|
|Enron Data set|2|[84], [1]|
|RenRenDai|1|[141]|
|Bloxy<br>iCOV|1<br>1|[66]<br>[47]|
|Secret Presidential Funds scandal|1|[23]|
|Tencent QQ|1|[154]|
|Caviar data set|1|[11]|
|BitcoinTalk|1|[138]|
|Ripple Network|1|[19]|
|2015 Ashley Madison extortion scam|1|[105]|
|FinCEN|1|[29]|
|BlockCypher|1|[109]|



three stages to gain deeper insights. First, we revisit the categories as presented in Table 1. Second, we analyse the task-level objectives by discussing the data and methods present in these literature. Third, the papers are classified according to the processing steps and challenges, as introduced by Bockel-Rickermann et al. [14]. 

### **3.3.1 General Classification.** 

Five out of 11 review papers are selected. This is to be expected, since review papers have a higher likelihood of being cited [32]. Plots summarising the classification according to Table 1 are given in Appendix B. Similar results are observed for the top-cited papers as for the literature as a whole; unsupervised learning is by far the most applied method, and research relies heavily on manual and rule-based features. Only time and threshold-dependent metrics were used for evaluation. This indicates that threshold-independent metrics are not mainstream when comparing AML models using networks analytics. We observe that, compared to the wider literature, flow/chain detection methods are better cited as well as studies using open-source data. As mentioned before, flow detection better captures the intricate patterns used in money laundering. The use of open-source data makes comparison with and replication of the studies possible, fostering research adoption. 

Additionally, five papers deal with crypto-related money laundering, indicating a strong research interest. On the one hand, crypto currencies are widely used by criminals because of the absence of regulation [40] and the anonymity it provides [63, 134]. On the other hand, a full history of ownership is available on the blockchain for each _coin_ , facilitating the discovery of transaction patterns in payment networks [63, 134]. 

### **3.3.2 Task-Level Objectives.** 

Table 7 summarises the data set used with information therein, the model proposed by the authors to capture a certain money laundering phenomenon, and the baselines to which the proposed model is compared. Additionally, Table 8 summarises the network properties of these papers. 

We again see a strong reliance on _basic_ centrality measures for anti-money laundering. A couple of papers apply very basic analysis. Fronzetti Colladon and Remondi [41] and Malm and Bichler [87] purely consider correlations to extract conclusions for their AML features. 

Next to the typical objectives of money laundering detection, some research illustrates how network analytics can be used to support investigators to find novel leads and other people involved in previously-reported cases. 

Table 7 points to limitations in the literature. Many of the proposed methods are only tested on proprietary data, limiting the reproducibility of their results. The performance is often illustrated using case studies, without the inclusion of other baselines. This shows that there is a lack of a consensus on baselines and state-of-the-art methods in the field. 

12 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 6: Summary of the top-cited papers, with indication of whether it is a review and/or crypto-related. 

|Reference|Title|Year|Citations|Review|Crypto|
|---|---|---|---|---|---|
|Ngai et al. [99]|The application of data mining techniques in fnancial<br>fraud detection: A classifcation framework and an<br>academic review of literature|2011|1307|✓||
|Fronzetti Colladon and<br>Remondi [41]|Using social network analysis to prevent money laun-<br>dering|2017|200|||
|Chang et al. [22]|WireVis: Visualization of Categorical, Time-Varying<br>Data From Financial Transactions|2007|171|||
|Senator et al. [113]|Financial Crimes Enforcement Network AI System<br>(FAIS) Identifying Potential Money Laundering from<br>Reports of Large Cash Transactions|1995|152|||
|Dre˙zewski et al. [33]|The application of social network analysis algorithms<br>in a system supporting money laundering detection|2015|141|||
|McGinn et al. [88]|Visualizing Dynamic Bitcoin Transaction Patterns|2016|122||✓|
|Gao and Ye [42]|A framework for data mining-based anti-money laun-<br>dering research|2007|106|✓||
|Wu et al. [137]|Detecting Mixing Services via Mining Bitcoin Trans-<br>action Network With Hybrid Motifs|2021|94||✓|
|Malm and Bichler [87]|Using friends for money: the positional importance of<br>money-launderers in organized crime|2013|66|||
|Zhou et al. [151]|A Local Algorithm for Structure-Preserving Graph<br>Cut|2017|63|||
|Li et al. [78]|FlowScope: Spotting Money Laundering Based on<br>Graphs|2020|54|||
|Irwin and Turner [63]|Illicit Bitcoin transactions: challenges in getting to<br>the who, what, when and where|2018|47|✓|✓|
|Phetsouvanh et al. [105]|EGRET: Extortion Graph Exploration Techniques in<br>the Bitcoin Network|2018|38||✓|
|Kute et al. [74]|Deep Learning and Explainable Artifcial Intelligence<br>Techniques Applied for Detecting Money Launder-<br>ing–A Critical Review|2021|38|✓||
|Zhdanova et al. [150]|No Smurfs: Revealing Fraud Chains in Mobile Money<br>Transfers|2014|32|||
|Bhalerao et al. [13]|Mapping the underground: Supervised discovery of<br>cybercrime supply chains|2019|24|||
|Ovelgönne et al. [101]|<br>Covertness Centrality in Networks|2012|20|||
|<br>Cheong and Si [23]|<br>Event-based approach to money laundering data anal-<br>ysis and visualization|2010|12|||
|Gerbrands et al. [47]|<br>The effect of anti-money laundering policies: an em-<br>pirical network analysis|2022|11|✓||
|Zhou et al. [152]|Behavior-Aware Account De-Anonymization on<br>Ethereum Interaction Graph|2022|7||✓|



The above observations motivate the construction of the experimental set-up of Section 4. Our aim is to come to a common framework to test methods for network analytics in a uniform way. 

In addition to the objectives of the studies, Table 8 presents the network construction in these papers. It indicates that most of the papers use homogeneous networks with transactions forming the edges between accounts. Only a few use heterogeneous networks, either by having both accounts and transactions as nodes, or having nodes that represent different characteristics or feature values. 

### **3.3.3 Processing steps.** 

Next, the papers are classified according to the framework presented by Bockel-Rickermann et al. [14]. Table 9 summarises the papers over the processing steps. The most prevalent pre-processing steps are feature engineering and exploration. Feature engineering is done to define different risk profiles or to enhance GNN models. This is either based on neighbourhood and centrality metrics [41, 33, 101] or on the aggregation of local (node-specific) information [152, 150]. For exploration, some research is done on how to construct meaningful visualisations [22, 88], while other papers rely on clustering the nodes as additional support for their findings [87, 41]. 

13 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 7: Summary of the objective of the paper, where the paper uses [Data set] with [information] to introduce a <u>[model] capturing [phenomenon] and is compared to [baselines].</u> 

|Paper|Data set|Information|Model|Phenomenon|Baselines|
|---|---|---|---|---|---|
|Fronzetti Colladon and<br>Remondi [41]|Italian factoring data|Transaction invoices,<br>economic sector,<br>geographical area|Correlation of centrality<br>metrics with label|Smurfng|-|
|Chang et al. [22]|Proprietary bank data set|<br>Transactions, accounts,<br>keywords,<br>personal information|Visualisation of<br>keywords-accounts, account<br>using transactions, and<br>evolution transactions of<br>(groups of) accounts|Suspicious behaviour|-|
|Senator et al. [113]|Reports of large cash<br>transactions according to<br>Bank Secrecy Act|People, accounts, businesses,<br>locations and transactions|<br>Graphical user interface|identifying high-value leads|-|
|Dre˙zewski et al. [33]|<br>Proprietary bank statements,<br>national court register|Connections|Rule-based role defnition<br>via centrality metrics|Roles in criminal<br>organisations|-|
|McGinn et al. [88]|<br>Bitcoin|Transactions, wallets|<br>Visualisation|Unexpected high-frequency<br>transaction patterns|-|
|Wu et al. [137]|Bitcoin|Transactions and wallets|Account, transaction and<br>network feature extraction +<br>z-score based on hybrid<br>motives + logistic regression|<br>Mixing|-|
|Malm and Bichler [87]|Police intelligence reports|Person under investigation<br>in the report and relations,<br>demographic characteristics<br>of each person, nature of<br>illicit drug trade<br>involvement, and the types<br>of relationships that existed<br>among individuals|<br>Correlation of betweenness<br>and eigen-vector centrality<br>with known roles|Role of money-launderers in<br>illicit markets|-|
|Zhou et al. [151]|Proprietary bank data set|<br>Bank accounts, names,<br>emails, addresses, and phone<br>number|HOSPLOC (local graph<br>clustering)|Synthetic identities and<br>money laundering|Local clustering (Nibble,<br>NPR, LS-OSC), global<br>(NMF, TSC)|
|Li et al. [78]|Proprietary and Czech<br>Financial Data set|Bank accounts, transactions|FlowScope (objective<br>maximisation to fnd fast<br>money fow through bank in<br>multi-artite network)|Anomalous money fow<br>indicative for money<br>laundering|<br>SpokEn, D-Cube, Fraudar,<br>HoloScope, RRCF|
|Phetsouvanh et al. [105]|Bitcoin blockchain and<br>Ashley Madison blackmail|Transactions and wallets|p<br>Average length and<br>confuence analysis|Suspicious bitcoin fow and<br>discover other wallets owned<br>by suspected perpetrators|-|
|Zhdanova et al. [150]|Synthetic MMT data|Transaction between phones|PSA@R (event-driven<br>process analysis)|<br>Smurfng|PART, C4.5, Random Forest|
|Bhalerao et al. [13]|Forum threads (English and<br>Russian)|Users, Posts and replies|<br>Automatic classifcation<br>using XGBoost, LogReg,<br>SVM ans FastText.<br>Construction supply chain<br>based on interaction graph.<br>Visualisation via alluvial<br>plots|Money laundering and other<br>fraudulent behaviour|-|
|Ovelgönne et al. [101]|Youtube and Universitat<br>Rovira I Virgili data sets|Network|Covertness centrality|Hiding behaviour|-|
|Cheong and Si [23]|<br>Criminal relations database|Transactions and<br>relationships|Rule-based degree of<br>suspicion and visualisation<br>of relations|Find additional suspects in<br>money laundering cases|-|
|Gerbrands et al. [47]|iCOV|Transactions, family and<br>professional relations of<br>people investigated|<br>Clustering (Louvain and<br>temporal) + node level<br>centralities|Interactions over time|-|
|Zhou et al. [152]|Ethereum (Eth-ICO,<br>Eth-Mining, Eth-Exchange,<br>Eth-Pish&Hack)|<br>Transactions and wallets|Pipeline based on sampling,<br>Hierarchical Graph<br>ATtention Encoder and<br>sub-graph contrastive<br>learning|Account identifcation<br>(phishing, exchange etc.)|Manual, Deepwalk,<br>node2vec, trans2vec,<br>graph2vec + LR, RF, LGBM,<br>and GCN, GAT, GIN,<br>I2BGNN-A, I2BGNN-T|



Most papers apply unsupervised learning in their processing step. This involves detection of suspicious transaction flows [151, 78, 105]. Another important part is the use of centrality metrics to define specific roles for entities in the network [33, 87, 101]. Finally, some methods create heuristics to define outlying behaviour [113, 22, 23, 150]. 

For the post-processing step, most research covers statistical evaluation, implementation and interpretation. Statistical evaluation is used to obtain a quantitative evaluation of the performance. Implementation consists of analysing the efficiency of the methods. It deals solely with computation time and scalability. For interpretation, researchers use specific interpretable features. 

More detailed trends in the literature are discussed below according to three main categories, that mainly follow the processing steps; unsupervised learning, (semi-)supervised learning and visualisation methods. 

**Unsupervised learning** is divided into two main streams. The first assigns and identifies specific roles of actors in criminal networks, based on police reports of criminal cases [87, 33]. The second covers methods to track money flows through the network [151, 78, 105]. It is important to integrate these transaction chains, since money laundering is a process involving multiple steps and actors [75]. Next to these two main streams, the remaining paper [101] introduces a new covertness measure. 

14 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 8: Summary of the construction of the network. 

|**Paper**|**Nodes**|**Edges**|
|---|---|---|
|||Economic sector<br>geographical data|
|Fronzetti Colladon and<br>Remondi [41]|Factoring companies|<br>transaction amount<br>tacit link (same owners/resources)|
|Chang et al. [22]|People<br>|Transactions|
|Senator et al. [113]|Accounts<br>People, businesses, accounts and location|Connections|
|Dre˙zewski et al. [33]<br>|People (via court register)<br>|Transactions (via bank statements)<br>|
|McGinn et al. [88]|Transactions, inputs and outputs<br>|Input/output part of same transaction,<br>inputs that belong to the same address<br>|
|Wu et al. [137]|Address<br>Address and transactions|Transactions<br>If they are related|
|Malm and Bichler [87]|People (from police reports)|Co-appearance in the report (family,<br>client-lawyer etc.)|
|Zhou et al. [151]|All items of interest (bank account, name,<br>email, address, phone number)|Link bank account to other items|
|Li et al. [78]|Bank accounts|Transfers|
|Phetsouvanh et al. [105]|Bitcoin wallet address|Transactions|
|Ovelgönne et al. [101]|-|-|
|Cheong and Si [23]|People|Relations|
|Zhou et al. [152]<br>|Accounts<br>|Transactions<br>|
|Zhdanova et al. [150]|mWallets|Transactions|



Table 9: Papers per method cluster, based on Bockel-Rickermann et al. [14]. 

|**Process Step**|**Sub-Steps**|**References**|
|---|---|---|
|Pre-processing|Sampling<br>Exploration<br>Missing Value Treatment<br>Outlier Detection and Treatment<br>Categorization, standardization & segmentation<br>Feature Engineering<br>Variable Selection|[41].<br>[41],[22],[88], [87].<br>[41].<br>-<br>[33], [13].<br>[41], [33], [101],[152], [150].<br>-|
|Processing|Unsupervised Learning<br>Supervised Learning<br>Semi-Supervised Learning<br>Hybrid Learning|[22], [113], [33], [87], [151], [78], [105],<br>[101], [23],[150].<br>[41], [13].<br>[137].<br>[152].|
|Post-processing|Statistical Evaluation<br>Interpretation<br>Economical Evaluation<br>Implementation|[41], [137], [151], [78], [13], [152], [150].<br>[33], [137], [13], [152].<br>[150].<br>[33], [88], [151], [78], [101],[152], [150].|



Although detecting money laundering is in principle trying to find anomalous behaviour, very few studies deal with (unsupervised) anomaly detection [85, 108, 34], leaving possibilities for future work. 

**(Semi-)Supervised learning** methods deal with different aspects of money laundering. Their strength lies in a combination of (1) the methods used and (2) tailoring the network to the problem at hand. The obtained network features are put into downstream classifier. To maintain interpretability, network metrics are recalculated for multiple networks, each representing a specific aspect of the data, to extract predictive features [41, 137]. Promising methods are based on deep learning [152], but these are mostly black boxes. Research on the interpretability of deep network representation methods is still scarce [74]. 

15 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 10: Papers per fraud detection challenge, based on Bockel-Rickermann et al. [14]. 

||**Challenge**|**Papers**|
|---|---|---|
||Automation<br>Class Imbalance<br>Concept Drift<br>Data Availability<br>Feature Construction|[113], [23], [152].<br>[137], [13].<br>[41], [13].<br>[41], [22], [137], [87], [101], [150].<br>[101].|
|Original|Noisy Data<br>Stream Data<br>Real-Time Execution<br>Scalability<br>Unlabelled Data<br>Verifcation Latency|[113], [137], [87].<br>[113], [88], [152].<br>[41].<br>[151], [101], [152].<br>[113], [137], [13], [152].<br>[150].|
|New|Bias in data<br>Generalisation<br>Robustness|[87], [150].<br>[152].<br>[78].|
||Anonymity|[88], [137], [13].|



**Visualisation** of transaction networks is developed to support the experts in during investigations. This is either done by having a overview of the flow of money over different transactions [88, 22] or by finding specific relations among the persons involved [113, 29, 23]. Hence, these methods are mostly intended to be used after suspicion has been raised. None of them include an evaluation of the added value in money laundering detection. 

### **3.3.4 Challenges.** 

Table 10 summarises the papers according to the challenges identified by Bockel-Rickermann et al. [14]. It only contains those challenges mentioned in the literature. The main challenge is data availability, followed by unlabelled data. Due to privacy issues, only a limited number of anonymised data is available. Additionally, since transaction data sets are often huge and resources are limited, it is hard to provide all instances with a label. 

Few papers mention feature construction, real-time execution, or verification latency as a challenge. Anti-money laundering is less reliant on execution time, since the investigation is done after the fact, based on a longer history of transactions [75]. 

We introduce four additional challenges observed in the anti-money laundering literature, supplementing BockelRickermann et al. [14]. We view this as an extension of the framework since these challenges can be important to fraud research in general. 

- **Bias in data** affects the performance of the model and manifests itself in different ways. First, there is _bias in the missing values_ [87]. Second, _detection bias_ [150] can be present in synthetic data when specific patterns are present that the model is trained to look for. 

- **Generalisation** is mainly used in two ways. One way refers to the ability for methods developed for anti-money laundering to be used for other fraud domains as well [152]. Another and important use of the term refers to the need for methods to generalise to unseen data [49], often referred to as _inductive_ methods (compared to _transductive_ ones). This is an important consideration when selecting network-based methods [56, 127, 128]. 

- **Robustness** of a method is important since money laundering can be seen as an adversarial attack [78], as perpetrators try to avoid detection by the methods. 

- **Anonymity** of the account data gives rise to two main challenges. First, the identity of the account holder is unknown in crypto data sets, which makes it almost impossible to know if different accounts - also called wallets - belong to the same person [137]. Second, financial institution possess customer data, but due to privacy reasons, researchers cannot publish these. Hence, published articles avoid using Personally Identifiable Information all together [13]. 

### **3.4 Further Considerations** 

Based on the observations made above, we give an intermediate conclusion and perspective on future work. The methods in the literature still rely heavily on expert-based methods, i.e., manual feature engineering and rule-based 

16 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

systems, but deep learning methods are gaining traction. A more detailed analysis of these methods is needed to evaluate their performance properly, especially given that the current literature barely applies threshold-independent metrics meant for highly skewed data, like the AUC-PR. Moreover, the top-cited papers barely compare their methods to other baselines. This detailed analysis is the topic for the second part of this paper. 

Different gaps in the literature exist. A first gap was also identified by Kute et al. [74], namely the lack of interpretation tools. Future work should interpretation methods to try and explain the output of graph neural networks. 

A second gap concerns the under-representation of classic unsupervised methods [21, 3]. Future work should analyse the effectiveness of unsupervised learning in networks to find anomalous behaviour. This can be beneficial since many clients and transactions are effectively unlabelled, and the modi operandi for money laundering keep evolving. The second part of this paper will introduce a first way of applying such unsupervised learning methods, but a more extensive study is needed. 

A third is that most research studies static networks. Some research is being done to incorporate the time-dynamic nature of transaction networks using recurrent neural networks, but we believe that much more can still be done. 

A fourth gap is related to the inherent incompleteness of labels in AML. As criminals try to cover their tracks and financial institutions have only limited resources [129, 106], it is a given that there are still many money laundering cases that stay unnoticed. These transactions are seen as having label 0 when training the models, resulting in a positive and unlabelled (PU) problem. Specific PU-learning methods for networks should be applied when training a model. 

A fifth gap is related to the limited resources as well. Since not all clients and transactions can be investigated, the predictions should take this into account. One way of dealing with this is learning-to-rank [28, 130], which has already been shown to be effective in fraud detection. The main benefit is that the goals is to correctly rank the most suspicious clients, while less importance is given to have a correct ranking for less suspicious clients. Although this is highly relevant, it has not yet been applied to the AML literature using network analytics. 

## **4 Experimental Set-Up** 

To the best of our knowledge, a benchmark study comparing the state-of-the-art network learning methods for antimoney laundering is still missing. This section describes the experimental framework to fill this gap, including model specification (Section 4.1), data (Section 4.2), hyperparameter tuning (Section 4.3) and performance metrics (Section 4.4). The focus is on supervised learning, but unsupervised learning is also explored. Given challenges in open-source data availability, we present results on two data sets and provide code to facilitate reproducible experiments. We hope this framework will support and expand future research, and allows uniform publishing and comparing of results on proprietary data without the need to disclose sensitive information. 

### **4.1 Model Specification** 

To set up the experimental evaluation, we start from the wider literature on network representation. Extensive overviews are given by Cai et al. [18], Hamilton et al. [56] and Goyal and Ferrara [50]. A sub-branch of representation learning applies deep learning methods [153, 139]. Based on these papers, we select the methods that are most prominently used. 

We refined the selection by considering that, for AML, transaction data contains networks with millions of nodes and edges, making scalability an important selection criterion. Table 11 provides an overview of the methods selected. They are grouped into three categories [18, 56, 50]: manual feature engineering, shallow representation learning and deep representation learning. These methods will be tested against a baseline model that only includes the intrinsic features, denoted by IF. 

**Manual feature engineering** includes both local and global metrics. The first is the density of the ego-network. A node’s ego-network consists of the node, its direct neighbours and all connections between the neighbours. The density is the relative number of connections in this ego-network compared to the theoretical maximum. Summary statistics, i.e., minimum, mean and maximum values, of the density of the node’s neighbours are also included. 

A second type consists of centrality metrics, quantifying the global position of the node in the network [98]. The most popular centrality metrics are the betweenness, closeness and eigenvector centrality. The betweenness quantifies the number of shortest paths between any two nodes in the network passing through that node. The closeness, being the inverse of the average distance, measures how close a node is to all other nodes. The eigenvector centrality is high for nodes that are connected to important nodes in the network. The importance of the node itself is quantified using the PageRank [103]. 

17 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 11: The methods used to generate network features for the experimental evaluation. 

|**Category**|**Features**|**Defnition**|
|---|---|---|
|Manual feature engineering|Density<br>Centrality measures<br>PageRank|Baesens et al. [10]: Number of edges in the egonet<br>relative to the maximal number possible.<br>Closeness, betweenness and eigenvector centrality.<br>Page et al. [103]: Score indicating the importance of a<br>node.|
|Shallow representation learning|DeepWalk<br>Node2vec|Perozzi et al. [104]: Random walk through the network.<br>Embedding via word2vec.<br>Grover and Leskovec [52]: Truncated random walks<br>(breadth-frst vs. depth-frst). Embedding via word2vec.|
|Deep representation learning|GCN<br>GraphSAGE|Kipf and Welling [70]: Graph convolutional network<br>Hamilton et al. [56]: Aggregation based on fxed-size<br>sample of neighbours|
||GAT<br>GIN|Veliˇckovi´c et al. [132]: Graph attention network<br>Xu et al. [144]: Graph isomorphism network|



Next to the manual feature engineering, automatic network features are constructed using network embedding methods [56, 50, 18]. Although there are many different methods to construct a network embedding [18], we use those based on deep learning, called network representation learning [127, 129]. 

**Shallow network representation learning** can be viewed as an “embedding lookup” [56, 129]. We apply methods based on random walks. DeepWalk [104] was the first method to try to give nodes that are close in the network, i.e., that co-occur on short random walks, a similar embedding. Each random walk is seen as a sentence of words. The nodes are embedded into a Euclidean latent space using NLP methods, often skip-gram [94]. 

Node2vec [52] extends DeepWalk by introducing two hyperparameters, _p_ and _q_ . The method samples each node according to the following unnormalized transition probability, _α_ , at node _v_ , when coming from node _t_ : 



where _dtx_ denotes the distance between node _t_ and _x_ . These probabilities are defined a link exists between _v_ and _x_ . Otherwise, it is set to 0. 

The main drawbacks of shallow representation learning methods are that (1) they are transductive, so need full retraining when a new node is added to the network, and (2) retraining on the same network results in a different embedding. This second point can be understood intuitively by viewing two embeddings of the same network, with one being a rotation of the other. Both are equally good, but a downstream classifier trained on one cannot be used to make predictions on the other, as the coordinates of the nodes are different. 

In the experiments below, we construct the embedding on all nodes in the training and validation set, while only training on the labels from the training set. Afterwards, the embedding is constructed for the full network, and we train a new classifier using the labels of the train and validation set. 

This mimics the applications of these methods in reality, since a bank will have the full transaction network up to the current time, with some historical labels. Based on this, they need to determine which accounts to investigate next for money laundering. 

**Deep network representation learning** adapts deep learning methods to be directly applicable on networks. We apply different graph neural network (GNN) architectures. 

The most popular methods for node embedding calculation are graph convolutional networks (GCNs), based on convolutional neural networks [70]. It relies on neighbourhood aggregation to update the embedding as follows: 



with _H_<sup>(</sup><sup>_l_)</sup> the embedding at step _l_ , _σ_ a non-linear (activation) function, _A_<sup>˜</sup> = _A_ + _I_ the adjacency matrix with self-loops added, the diagonal matrix _D_<sup>˜</sup> _ii_ =<sup>�</sup> _j_<sup>_A_˜</sup><sup>_ij_, and</sup><sup>_W_(</sup><sup>_l_)the learnable weights in layer</sup><sup>_l_.The node embedding for the</sup> 

18 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

individual node is updated via 



Neighbourhood aggregation has seen multiple extensions, two of which this paper will evaluate. The first is GraphSAGE [56], which samples a fixed number of neighbours for each node, to make the calculations scalable to large graphs. The representations of the sampled neighbours are aggregated, with the authors proposing the mean, LSTM and pooling aggregators [56]. Then, the aggregated neighbourhood representation is concatenated with the node’s representation. Finally, the representation is updated using the weights and activation function. For a given node _v_ , this is calculated as: 



where _∥_ represents the concatenation operator. 

The second extension is graph attention network (GAT) [132], which adds an attention mechanism. Our work uses the modified version of the attention mechanism, introduced to fix the static attention problem [16]: 



As mentioned by Veliˇckovi´c et al. [132], having multiple mechanisms—also called heads—can improve stability. We follow Veliˇckovi´c et al. [132], where the intermediate steps concatenate the results over the heads, while the final layer averages the results. 

The final method is the Graph Isomorphism Network (GIN) [144]. The authors used the Weisfeiler-Lehman graph isomorphism test to define the necessary conditions a GNN must satisfy to achieve maximal discriminative power. This discriminative power is denoted by how well the training data is fitted. 

Using the universal approximation theorem [61, 60], GIN updates the representation using a multilayer perceptron (MLP). 



Although this increases the risk of overfitting, the authors show that GIN has satisfactory generalisation capabilities. The main problem with other GNNs is that they tend to underfit the data [144]. Recent research showed GIN’s potential to outperform other GNN architectures [36, 123]. 

There are some key differences between the three categories presented in Table 11. The manual features need to be specifically defined, while representation learning tries to find meaningful embeddings automatically. Additionally, the manual features and the shallow representations are added to the intrinsic features, which are used in a down-stream classifier (as illustrated in Figure 5). The GNNs incorporate the intrinsic features and directly learn the embeddings on the classification task. 

Different decoders are used for the final classification in the supervised setting. A neural network with two hidden layers, each of dimension ten is used for the manual features and shallow representation methods. For the graph neural networks, the GNN layers are followed by a single linear layer for making predictions. 

For the unsupervised model, we apply isolation forest [80], as it achieves state-of-the-art performance for outlier/anomaly detection [122, 64]. This will be applied to the intrinsic and manual features, and the shallow representations, since it requires a tabular input. A larger experiment comparing unsupervised methods for network analytics is left for future work. 

One of the objectives of this research is to provide the code for all methods in a clean and simple way and make these methods easily accessible to researchers and practitioners. The methods are implemented in Python. The manual 

19 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

|||A|B<br>C<br>D<br>E<br>Ne|twork|Embed|ding|||||
|---|---|---|---|---|---|---|---|---|---|---|
|**ID**|**F1**|**F2**|_. . ._<br>**Fn**<br>**ID**|**F1**|**F2**|_. . ._|**Fn**|**NF1**|_. . ._|**NFm**|
|A|f11|f21|_. . ._<br>fn1<br>A|f11|f21|_. . ._|fn1|nf11|_. . ._|nfm1|
|B|f12|f22|_. . ._<br>fn2<br>B|f12|f22|_. . ._|fn2|nf12|_. . ._|nfm2|
|...|...|...|...<br>...<br>...|...|...|...|...|...|...|...|
|E|f1k|f2|_. . ._<br>fnm<br>E|f1k|f2|_. . ._|fnm|nf1k|_. . ._|nfmk|



Figure 5: For the feature engineering and shallow representation methods, the obtained network features/embedding is added to the intrinsic ones. 

features are constructed using NetworkX [54] and NetworKit [7], and the representation learning is implemented in PyTorch Geometric [39]. The code is made available on GitHub<sup>2</sup> . 

### **4.2 Data** 

The methods are compared on two data sets; the Elliptic [37, 134] and IBM money laundering [6, 36] data sets. These cover the two main lines of research in the literature. The Elliptic data set concerns real-world crypto-transactions, while the IBM data set deals with simulated transactions in fiat currencies. 

### **4.2.1 Elliptic.** 

The Elliptic data set [37, 134] contains Bitcoin transactions for 49 time steps, each being around two weeks long. The network has 203 769 nodes and 234 355 edges. The nodes correspond to transactions, and an edge represents that the output of one transaction is the input of the next. Additionally, the data set contains 166 pre-calculated features. These are split into 94 transaction-specific, i.e., local features and 72 aggregated features, summarizing the local features of a node’s neighbours. 

All features are numerical and have already been standardised. The local features will be used for all methods. Aggregated features are not included for the GNNs, since these methods aggregate the neighbourhood information, including the node features. This was also addressed by Weber et al. [134], stating that graph neural networks are better to address the heterogeneity of the neighbourhoods than the aggregated features. 

Around 33% of transactions are labelled, 4 545 of which as illicit and 42 019 as licit. Hence, the classification problem is imbalanced with 2% of all nodes labelled _illicit_ . We note that an illicit label does not automatically mean that the transaction is used to launder money. However, criminals will try to obscure the source of these illicit funds, so we believe that it is still informative. Section 3.2 also illustrated that this data set is widely used in AML research. 

We use the Elliptic data set included in the Pytorch Geometric library [25, 39]. We split the periods into a train (period 1-30), validation (period 31-40) and test set (period 41-49). Only nodes having a label are used for performance evaluation. Transactions with label 2 (unknown) are included for the construction of the network, but they do not contribute to training or loss calculation [70, 147, 134]. 

### **4.2.2 IBM Money Laundering.** 

The second data set is the HI-Small data set, taken from Kaggle, consisting of simulated transactions in a virtual world between individuals, companies and banks [6, 36]. A fraction of these transactions are labelled as money laundering. Given that the labels are at transaction level, the network is constructed with the transactions as nodes. Edges are added from transaction _vi_ to _vj_ if the receiver of transaction _vi_ is the sender of transaction _vj_ , transaction _vj_ happened before _vi_ , and the difference in time is smaller than ∆ _t_ , i.e., ∆ _t ≥ tj − ti ≥_ 0. This is inspired by the work by Tariq and Hassani [120], and allows to capture the money flow. We take a 60-20-20 train-validation-test split based on transaction time. 

> 2 `https://github.com/B-Deprez/AML_Network` 

20 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 12: Network features for the Elliptic and IBM-AML data set. 

|**Network**|**Nodes**|**Edges**|**Percentage Fraud**|**Avg. Degree**|**Avg. Path Length**|**Clustering**|
|---|---|---|---|---|---|---|
|Elliptic|203,769|234,355|2%|2.3|15.96|0.0117|
|IBM-AML|500,000|1,278,952|0_._27%|5.1|3.78|0.0000|



Since this results in a massive data set and hyperparameter tuning for DeepWalk and node2vec would take too long, we only take the last 500 000 transactions, and set ∆ _t_ to four hours. This results in a network with 500 000 nodes and 1 278 952 edges. Just 1356 transactions, i.e., 0 _._ 27%, are labelled as money laundering. 

Each transaction has a couple of features, of which we use the following: ‘Amount Received’, ‘Receiving Currency’, ‘Amount Paid’, ‘Payment Currency’, ‘Payment Format’. Next to that, we extract the day of the week, hour and minute out of a transaction’s date stamp. The categorical features, i.e., ‘Receiving Currency’, ‘Payment Currency’, ‘Payment Format’, are transformed using one-hot-encoding. In the remainder of this paper, we denote this data set as IBM-AML. 

An analysis of the topology of both networks is given in Table 12, based on well-known network features. IBM-AML seems to be much more connected, compared to the Elliptic data set, having higher average degree and lower average path length. This is corroborated by the degree distribution and the distributions of the closeness centrality, in Figure 6 and Figure 7, respectively. The Elliptic data set more closely presents a scale-free distribution, while the IBM-AML data set deviates from this with fewer nodes with low degree and more nodes with higher degree (hubs). The hubs resulted in a much lower average path length, and higher closeness centrality values. 



<!-- Start of picture text -->
Degree distribution<br>Elliptic<br>105 IBM-AML<br>104<br>103<br>102<br>101<br>100<br>100 101 102 103<br>Degree (log)<br>Frequency (log)<br><!-- End of picture text -->

Figure 6: Comparison of the degree distributions of the Elliptic and IBM-AML data sets. 



<!-- Start of picture text -->
Closeness distribution<br>25000 EllipticIBM-AML<br>20000<br>15000<br>10000<br>5000<br>0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>Closeness Centrality<br>Frequency<br><!-- End of picture text -->

Figure 7: Comparison of the distributions of the closeness centrality of the Elliptic and IBM-AML data sets. 

### **4.3 Hyperparameter Tuning** 

The hyperparameters are tuned to maximises the AUC-PR on the validation set. However, it is not feasible to tune these using a grid search. As the methods often have many hyperparameters, the number of combinations increases 

21 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 13: The overview of the hyperparameter values for the supervised learning models. The tuned hyperparameter value is added between parentheses next to the name of the model. 

|**Hyperparameter**|**Tuning Range**|**Model (hyperparameter value Elliptic**_|_**IBM )**|
|---|---|---|
|_α_: random jump parameter|[0_._1_,_0_._9]<br>|PageRank<sup>S </sup>(0_._593_|_0_._646)<br>|
|Number of walks per node|[1_,_3]_∩_N|DeepWalk<sup>S </sup>(2_|_2), node2vec<sup>S </sup>(1_|_3)<br>|
|Walk length|[3_,_10]_∩_N|DeepWalk<sup>S </sup>(3_|_7), node2vec<sup>S </sup>(9_|_9)|
|Word2vec context window|[2_,_10]_∩_N|DeepWalk<sup>S </sup>(2_|_2), node2vec<sup>S </sup>(5_|_9)|
|size|||
|Latent/Embedding<br>dimension|[2_,_64]_∩_N|DeepWalk<sup>S </sup>(5_|_52), node2vec<sup>S </sup>(47_|_9)|
|Latent/Embedding|[32_,_128]_∩_N|GCN (108_|_97), GraphSAGE (48_|_53), GAT (99_|_51),|
|dimension||GIN (74_|_47)<br>|
|_p_: return parameter|[0_._5_,_2]|node2vec<sup>S </sup>(1_._17_|_0_._537)<br>|
|_q_: in-out parameter|[0_._5_,_2]|node2vec<sup>S </sup>(1_._60_|_1_._394)|
|Number of negative samples|[1_,_5]_∩_N|DeepWalk<sup>S </sup>(1_|_1), node2vec<sup>S </sup>(1_|_4)|
|GNN hidden dimensions|[64_,_256]_∩_N|GCN (NA_|_88), GraphSAGE (NA_|_175), GAT (NA_|_NA),<br>GIN (NA_|_NA)|
|GNN layers|[1_,_3]_∩_N|GCN (1_|_2), GraphSAGE (1_|_2), GAT (1_|_1), GIN (1_|_1)|
|Learning rate|[0_._01_,_0_._1]|IF<sup>S </sup>(0_._0163_|_0_._0770), Manual<sup>S </sup>(0_._0166_|_0_._0401),<br>DeepWalk<sup>S </sup>(0_._0554_|_0_._0526), node2vec<sup>S </sup>(0_._0159_|_0_._0708),<br>GCN (0_._0225_|_0_._0514), GraphSAGE (0_._0248_|_0_._0660),<br>GAT (0_._0421_|_0_._0313), GIN (0_._0345_|_0_._0166)|
|Aggregator|{min, mean, max}|GraphSAGE (mean_|_max)|
|Number of attention heads|[1_,_5]_∩_N_|_[1_,_2]_∩_N|GAT (4_|_2)|
|Dropout rate|[0_,_0_._5]|GCN (0_._288_|_0_._163), GraphSAGE (0_._350_|_0_._244),<br>GAT (0_._186_|_0_._188), GIN (0_._271_|_0_._273)|
|Number of layers decoder|[1_,_3]_∩_N|IF<sup>S </sup>(1_|_2), Manual<sup>S </sup>(1_|_2)|
|Hidden dimension decoder|[5_,_20]_∩_N|IF<sup>S </sup>(5_|_14), Manual<sup>S </sup>(6_|_17)|
|Number of epochs decoder|<br>[5_,_500]_∩_N|<br>IF<sup>S </sup>(497_|_450)|
|Number of epochs decoder|<br>[5_,_100]_∩_N|<br>Manual<sup>S </sup>(64_|_59), DeepWalk<sup>S </sup>(80_|_75), node2vec<sup>S </sup>(93_|_75)|
|Number of epochs|[5_,_500]_∩_N_|_<br>[5_,_100]_∩_N|DeepWalk<sup>S </sup>(176_|_75), node2vec<sup>S </sup>(222_|_84), GCN (483_|_325),<br>GraphSAGE (498_|_399), GAT (280_|_341), GIN (218_|_89)|



exponentially. Additionally, transaction networks are very large, which results in a relative long training time for each individual combination of hyperparameters. Therefore, we tune the hyperparameters efficiently via Optuna [2], whereby we give a range of possible values to select from. We use optuna’s default `TPESampler` sampler. 

The ranges are summarised in Table 13 and 14, for supervised and unsupervised learning, respectively. We add superscript S and U to indicate that tuning is done for supervised and unsupervised learning, respectively. A model only based on the intrinsic features (IF) is included. The number of trials is set to 50 for DeepWalk and node2vec, and 100 for the GNNs. 

Since the IBM data set is much larger, scalability in both time and memory are very important. Especially DeepWalk and node2vec take much time to run. The higher average degree also makes transition probability calculations more expensive. That is why the number of epochs for calculating the embedding are capped at 100. The attention mechanism of GAT is conditional on the query node, meaning that attention is learned for each node pair. To be able to train the model in memory, we limit the attention head to a maximum of two. 

All decoders used for classification have a fixed architecture (Section 4.1). For the GNNs, the training of the decoder weights happens simultaneously with the other weights during training. For the manual and shallow representation features, the decoder/classifier needs to be trained afterwards. The only hyperparameter tuned is the number of epochs. 

The tuned hyperparameters are presented in parentheses in Table 13 and Table 14. 

22 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 14: The overview of the hyperparameter values for the unsupervised learning models. The tuned hyperparameter value is added between parentheses next to the name of the model. 

|**Hyperparameter**|**Tuning Range**|**Model (hyperparameter value Elliptic**_|_**IBM )**|
|---|---|---|
|_α_: random jump parameter|[0_._1_,_0_._9]|PageRank<sup>U </sup>(0_._187_|_0_._117)<br>|
|Number of walks per node|[1_,_3]_∩_N|DeepWalk<sup>U </sup>(3_|_2), node2vec<sup>U </sup>(3_|_2)<br>|
|Walk length|[3_,_10]_∩_N|DeepWalk<sup>U </sup>(6_|_7), node2vec<sup>U </sup>(4_|_6)<br>|
|Word2vec context window<br>|[2_,_10]_∩_N|DeepWalk<sup>U </sup>(4_|_6), node2vec<sup>U </sup>(2_|_4)|
|size|||
|Latent/embedding<br>dimension|[2_,_64]_∩_N|DeepWalk<sup>U </sup>(64_|_6), node2vec<sup>U </sup>(63_|_47)|
|_p_: return parameter|[0_._5_,_2]|node2vec<sup>U </sup>(0_._78_|_1_._52)<br>|
|_q_: in-out parameter|[0_._5_,_2]|node2vec<sup>U </sup>(1_._58_|_1_._60)<br>|
|Number of negative samples|[1_,_5]_∩_N|DeepWalk<sup>U </sup>(4_|_5), node2vec<sup>U </sup>(4_|_1)<br>|
|Learning rate|[0_._01_,_0_._1]|DeepWalk<sup>U </sup>(0_._0948_|_0_._0541), node2vec<sup>U </sup>(0_._0414_|_0_._062)|
|Number of epochs|[5_,_500]_∩_N_|_<br>[5_,_100]_∩_N|DeepWalk<sup>U </sup>(17_|_46), node2vec<sup>U </sup>(59_|_50)|
|Number of estimators|[50_,_200]_∩_N|IF<sup>U </sup>(60_|_119), Manual<sup>U </sup>(70_|_66), DeepWalk<sup>U </sup>(69_|_169),<br>node2vec<sup>U </sup>(60_|_100)<br>|
|Max. number of samples|[0_._1_,_1]|IF<sup>U </sup>(0_._484_|_0_._111), Manual<sup>U </sup>(0_._333_|_0_._559),<br>DeepWalk<sup>U </sup>(0_._533_|_0_._386), node2vec<sup>U </sup>(0_._824_|_0_._354)|
|Max. number of features<br>(%)|_{_10_∗n | n ∈_N10_}_|IF<sup>U </sup>(0_._1_|_0_._9), Manual<sup>U </sup>(0_._8_|_0_._3), DeepWalk<sup>U </sup>(0_._8_|_0_._2),<br>node2vec<sup>U </sup>(0_._8_|_0_._4)|
|Bootstrap|_{_True, False_}_|IF<sup>U </sup>(True_|_False), Manual<sup>U </sup>(True_|_True),<br>DeepWalk<sup>U </sup>(True_|_False), node2vec<sup>U </sup>(True_|_True)|



### **4.4 Performance Metrics** 

We tackle money laundering as a binary classification problem. We adopt the most popular performance metrics from the literature, namely the precision, recall and F1-score (Section 3.2). These require a classification threshold. Since investigation resources are limited [129], the thresholds will be set relatively high, e.g., by classifying the top 0 _._ 1%, 1% or 10% of scores as money laundering. This is supplemented by a threshold equal to the relative prevalence of money laundering in the data sets. 

The selection of these thresholds is ad-hoc, and can influence the conclusions. Therefore, we also include thresholdindependent metrics. The most popular one for binary classification is the area under the ROC curve (AUC-ROC). The ROC curve plots the true positive rate against the false positive rate. However, in highly imbalanced data sets with many negative samples, the false positive rate can appear low even when the absolute number of false positives is high, making it less sensitive to performance issues in detecting the minority class [26, 111]. The area under the precision-recall curve (AUC-PR) is better suited to compare these models [26, 102, 111]. This also adds to the literature, since we showed in Section 3.2 that few papers base their conclusions on the AUC-PR. 

To test model stability, the models are trained 10 times on a randomised version of the training set. The manual network features and shallow representation embeddings result in classic tabular features (cf. Figure 5). For these, we apply a bootstrap method, where a new train set is constructed of the same size as the original, where the observations are sampled with replacement. The deep representation methods apply train masks. We initialise new train masks by randomly selecting half the train masks from the original ones. 

## **5 Results and Discussion** 

### **5.1 Supervised Learning** 

The results for the threshold-independent metrics, i.e., AUC-ROC and AUC-PR, on the test set are shown in Table 15, including the standard deviation calculated as described in Section 4.4. Additionally, the threshold-dependent results, i.e., precision, recall and F1-score, are given in Table 16 and Table 17 for the top-1% results. To analyse the sensitivity to the threshold, we report the results on the other three thresholds in Appendix C. 

23 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 15: Threshold-independent metrics: AUC-ROC and AUC-PR values over the different methods for the two data sets, based on the test set. The standard deviation is also reported. 

|**Supervised**|**Elli**|**ptic**|**IBM-**|**AML**|
|---|---|---|---|---|
|**Methods**|**AUC-ROC**|**AUC-PR**|**AUC-ROC**|**AUC-PR**|
|Intrinsic features|0_._8607_±_0_._0079|0_._5730_±_0_._0958|0_._7560_±_0_._0075|0_._0186_±_0_._0008|
|Egonet features|0_._8648_±_0_._0090|0_._5956_±_0_._0328|**0****_._7579****_±_ 0****_._0033**|**0****_._0187****_±_ 0****_._0004**|
|DeepWalk|0_._8483_±_0_._0083|0_._5823_±_0_._0265|0_._7558_±_0_._0033|0_._0185_±_0_._0004|
|Node2vec|0_._8497_±_0_._0044|0_._5938_±_0_._0251|0_._7551_±_0_._0031|0_._0184_±_0_._0004|
|GCN|0_._8465_±_0_._0110|0_._5948_±_0_._0164|0_._5657_±_0_._0904|0_._0113_±_0_._0029|
|GraphSAGE|**0****_._8712****_±_ 0****_._0122**|**0****_._6392****_±_ 0****_._0336**|0_._6068_±_0_._0793|0_._0120_±_0_._0028|
|GAT|0_._8579_±_0_._0161|0_._6376_±_0_._0270|0_._4203_±_0_._1889|0_._0103_±_0_._0041|
|GIN|0_._8213_±_0_._0203|0_._5079_±_0_._0593|0_._5028_±_0_._2533|0_._0119_±_0_._0065|



Table 16: Threshold-dependent metrics: Precision, recall and F1-score values for the top 1% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Elliptic Supervised**<br>**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._8570_±_0_._2706|0_._1840_±_0_._1173|0_._2932_±_0_._1451|
|Egonet features|0_._9339_±_0_._0463|0_._1645_±_0_._0082|0_._2797_±_0_._0139|
|DeepWalk|0_._9116_±_0_._0464|**0****_._1881****_±_ 0****_._0895**|**0****_._3049****_±_ 0****_._1054**|
|Node2vec|0_._9321_±_0_._0465|0_._1642_±_0_._0082|0_._2791_±_0_._0139|
|GCN|0_._9911_±_0_._0094|0_._1728_±_0_._0047|0_._2942_±_0_._0070|
|GraphSAGE|0_._9822_±_0_._0188|0_._1721_±_0_._0077|0_._2929_±_0_._0113|
|GAT|**0****_._9964****_±_ 0****_._0075**|0_._1753_±_0_._0067|0_._2981_±_0_._0099|
|GIN|0_._9286_±_0_._0709|0_._1633_±_0_._0129|0_._2777_±_0_._0214|



Table 17: Threshold-dependent metrics: Precision, recall and F1-score values for the top 1% scores over the different methods for the IBM-AML data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**I**<br>**Precision**|**BM-AML Supervise**<br>**Recall**|**d**<br>**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._0190_±_0_._0040|0_._0210_±_0_._0044|0_._0200_±_0_._0042|
|Egonet features|**0****_._0192****_±_ 0****_._0034**|0_._0213_±_0_._0038|0_._0202_±_0_._0036|
|DeepWalk|0_._0174_±_0_._0053|0_._0193_±_0_._0059|0_._0183_±_0_._0056|
|Node2vec|0_._0181_±_0_._0047|0_._0200_±_0_._0052|0_._0190_±_0_._0049|
|GCN|0_._0100_±_0_._0045|0_._8807_±_0_._3119|0_._0197_±_0_._0089|
|GraphSAGE|0_._0119_±_0_._0027|**0****_._9742****_±_ 0****_._0298**|**0****_._0236****_±_ 0****_._0052**|
|GAT|0_._0065_±_0_._0070|0_._1805_±_0_._3695|0_._0104_±_0_._0139|
|GIN|0_._0085_±_0_._0087|0_._0166_±_0_._0087|0_._0092_±_0_._0089|



24 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

For both data sets, the models with intrinsic features give already good result. Adding egonet features is beneficial for both data sets. We do, however, see a drop in performance for DeepWalk and node2vec. It is possible that some of the feature values, i.e., the coordinates in the latent space, are correlated or noisy. Neural networks, used here as the decoder for classification, are known to suffer under correlated and noisy features [51]. GIN also performs poorly on both data sets. One explanation is that the neural network in the aggregation step is not optimal, and should be tuned further. Another explanation is that the inclusion of neural network aggregation makes GIN more sensitive to the high class imbalance. Hence, GIN seems less suited for anti-money laundering. It should be noted that good performance is observed for the precision indicating that GIN reduces false positives in the top-percentiles of predictions. 

When looking at the Elliptic data set, we see that also the GNNs appear to underperform, except for GraphSAGE. Additionally, GAT has good AUC-PR performance. We see in Tables 16, C1, C3 and C5 that for the top percentages the GNNs have remarkably higher precision, but lower recall. Although fewer cases are detected, those with highest predicted money laundering propensity contain far fewer false positives. This is highly important when adopting these models in practice. 

The results on the IBM data set in Table 15 show a different picture. Only the egonet features improve performance. The threshold-dependent metrics show a similar picture. 

One explanation for the poor performance of GNNs might be related to the network structure (Section 4.2), with very short average paths and many high-degree nodes. This results in strong over-smoothing [77], resulting in very similar embeddings for the different nodes with insufficient discriminatory power. 

Moreover, a deeper analysis reveals that the training loss, when training on the train and validation set, is highly unstable. Depending on the epoch and specific weight initialization, the results vary widely, with the models sometimes just predicting the majority class. For this specific data set, GNNs seem to face instability due to the extreme class imbalance. 

The supervised learning experiments on both data sets demonstrate that the network structure provides additional insights for anti-money laundering. The observed performance improvements are limited, but even a small relative increase in performance has material implications for the business. The efficiency of AML practices increases significantly since investigators have more high quality predictions, and spend less time on the investigation of false positives. 

However, depending on the network topology and the extent of the class imbalance, GNNs might be too unstable. Additional experiments are needed to quantify the sensitivity of GNNs to the imbalance, but this is outside the scope of this research. 

### **5.2 Unsupervised Learning** 

The results of the unsupervised methods using isolation forests are given in Tables 18. None of the results on the Elliptic data set are particularly good, with AUC-ROC values below 0.5. For completeness, we also include the threshold-dependent metric in Table 19, where most models fail to detect any money laundering cases. 

These results are in line with a previous study performed by Lorenz et al. [83] who tested anomaly detection methods on the Elliptic data set. The authors did not include any additional network analytics methods and only reported the F1 score. However, they found that across the methods, the anomaly detection results were much worse than for supervised learning, with reported F1-scores for the isolation forest equal to 0. 

We perform additional analyses on these results. Figure 8 shows the anomaly scores for the intrinsic features for the Elliptic data set. Here, a very negative score signifies anomalies. All money laundering cases have scores near zero, within the bulk of the distribution, indicating that the isolation forest does not recognize these cases as anomalies. 

Conversely, the results for IBM-AML are more in line with expectations. Network information captured by DeepWalk, results in the best performing model here. Comparing these results to those in Table 15, we see that DeepWalk even outperforms the supervised learning methods in terms of AUC-PR. 

The most plausible explanation in our opinion lies in the assumptions underlying isolation forest and the data generating process. Two specific assumptions by Liu et al. [80]: 

- Anomalies yield fewer partitions since anomalies occupy regions with lower density; 

- Instances with distinct attributes are separated earlier in the tree, contributing to shorter paths. 

Therefore, the nature of the data sets explains the difference in performance. The Elliptic data set is a real-life data set, containing criminals that actively attempt to cover their tracks [10]. They try to mimic the ‘average behaviour’, but no real transaction/person is average on all attributes. This is why it seems that classifying the least suspicious 

25 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Histogram of Anomaly Scores by True Label<br>4000 Legit<br>Money Laundering<br>3500<br>3000<br>2500<br>2000<br>1500<br>1000<br>500<br>0<br>0.7 0.6 0.5 0.4<br>Anomaly Score<br>Count (log scale)<br><!-- End of picture text -->

Figure 8: The anomaly scores for the legitimate and money laundering cases. More negative is more anomalous. 

Table 18: Threshold-independent metrics: AUC-ROC and AUC-PR values over the different methods for the two data set, based on the test set. The standard deviation is also reported. 

|**Unsupervised**|**Ell**|**iptic**|**IBM-**|**AML**|
|---|---|---|---|---|
|**Methods**|**AUC-ROC**|**AUC-PR**|**AUC-ROC**|**AUC-PR**|
|Intrinsic features|0_._134_±_0_._007|0_._058_±_0_._001|0_._710_±_0_._015|0_._017_±_0_._003|
|Egonet features|0_._217_±_0_._010|0_._013_±_0_._000|0_._700_±_0_._033|0_._011_±_0_._005|
|DeepWalk|0_._182_±_0_._008|0_._056_±_0_._001|**0****_._744****_±_ 0****_._023**|**0****_._024****_±_ 0****_._008**|
|Node2vec|0_._190_±_0_._008|0_._056_±_0_._0005|0_._632_±_0_._031|0_._005_±_0_._001|



transactions as money launderers, we would achieve an AUC-ROC around 80%. Hence, transactions that appear ‘too normal’ are actually most suspicious. In contrast, the IBM-AML data set is synthetic, with fraudulent patterns and behaviours deliberately introduced. It is likely that camouflage behaviour is not present in this data set. 

These insight highlights the importance of knowing the exact assumptions underlying the data sets used. Synthetic data sets might give overly optimistic results, since complex behaviour inherent to the problem at hand are not captured. 

### **5.3 Ablation Study** 

To better understand the effect of the network structure on the predictive power of the models, we analyse the performance of the different models without intrinsic features. For the models, we take the hyperparameters as determines in Table 13 and Table 14. The GNNs require feature values, so we give each node a dummy feature with value 1 [128]. Additionally, for the Elliptic data set, we have no control on the summary features, so we discard these as well, although they represent some network structure. 

Table 19: Threshold-dependent metrics: Precision, recall and F1-score values for the top 1% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

||**Ell**|**iptic Unsupervis**|**ed**|
|---|---|---|---|
|**Methods**|**Precision**|**Recall**|**F1-score**|
|Intrinsic features|0_._000_±_0_._000|0_._000_±_0_._000|0_._000_±_0_._000|
|Egonet features|0_._000_±_0_._000|0_._000_±_0_._000|0_._000_±_0_._000|
|DeepWalk|0_._005_±_0_._005|0_._001_±_0_._001|0_._001_±_0_._001|
|Node2vec|0_._003_±_0_._004|0_._000_±_0_._000|0_._001_±_0_._001|



26 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 20: Threshold-dependent metrics: Precision, recall and F1-score values for the top 1% scores over the different methods for the IBM-AML data set, based on the test set. The standard deviation is also reported. 

||**IB**|**M-AML Unupervis**|**ed**|
|---|---|---|---|
|**Methods**|**Precision**|**Recall**|**F1-score**|
|Intrinsic features|0_._042_±_0_._005|0_._156_±_0_._020|0_._067_±_0_._008|
|Egonet features|0_._027_±_0_._013|0_._098_±_0_._048|0_._042_±_0_._021|
|DeepWalk|**0****_._055****_±_ 0****_._009**|**0****_._204****_±_ 0****_._034**|**0****_._087****_±_ 0****_._015**|
|Node2vec|0_._007_±_0_._002|0_._025_±_0_._008|0_._010_±_0_._003|



Table 21: Threshold-independent metrics: AUC-ROC and AUC-PR values over the different methods for the two data sets, based on the test set. The standard deviation is also reported. 

|**Supervised**<br>**Methods**|**Elli**<br>**AUC-ROC**|**ptic**<br>**AUC-PR**|**IBM-**<br>**AUC-ROC**|**AML**<br>**AUC-PR**|
|---|---|---|---|---|
|Egonet features|0_._5172_±_0_._0607|0_._0630_±_0_._0122|0_._4758_±_0_._0187|0_._0088_±_0_._0003|
|DeepWalk|0_._5125_±_0_._0184|0_._0597_±_0_._0021|0_._5019_±_0_._0143|0_._0092_±_0_._0004|
|Node2vec|0_._4454_±_0_._0118|0_._0511_±_0_._0021|0_._4931_±_0_._0095|0_._0091_±_0_._0003|
|GCN|0_._5655_±_0_._0139|0_._0696_±_0_._0039|0_._4985_±_0_._0183|0_._0092_±_0_._0007|
|GraphSAGE|0_._5000_±_0_._0000|0_._0565_±_0_._0025|0_._5050_±_0_._0081|**0****_._0093****_±_ 0****_._0003**|
|GAT|0_._5002_±_0_._0032|0_._0566_±_0_._0018|0_._5014_±_0_._0035|0_._0091_±_0_._0002|
|GIN|**0****_._6177****_±_ 0****_._0110**|**0****_._0739****_±_ 0****_._0049**|**0****_._5061****_±_ 0****_._0035**|0_._0092_±_0_._0003|



The results for supervised learning are given in Table 21. The AUC values indicate that most methods based purely on the network structure are close to random. 

One notable exception is GIN, which is specifically designed to better distinguish certain graph structures [144]. Graph structure is the only information available in this part of the experiments. The results for GIN indicate that there are some structural difference in the network for fraudulent and non-fraudulent nodes in the Elliptic data set. These structural difference are complex, since they are not picked up by the egonet features or the random-walk-based ones. 

The other GNN architectures struggle, since they rely heavily on the node features. Especially GAT—which performed well with features—leverages node features for the attention calculations. 

For the IBM data set, the differences are less pronounced that for the Elliptic data set. The network structure alone might not be enough to uncover money laundering. The data generation process of this synthetic data set creates a direct correlation between the money laundering pattern and feature values. Therefore, these feature values are more important than the network structure. 

The results for the isolation forest in Table 22 show an improvement for the Elliptic data set, compared to the isolation forest with intrinsic features. Here, however, the models mostly make random predictions. When including the features, a better distinction can be made between money and non-money laundering. As mentioned before, the features of the money laundering transactions are so average (not anomalous according to the isolation forest) that they become suspicious again. The inverse predictions were in that case very good predictors. When only considering the network structure, it is harder to find meaningful anomalies. Hence, a higher AUC in this case points to a less useful model, as the model is as good as random. 

The results on the IBM data set are in line with what we have seen before. The model makes almost random predictions, with deepwalk giving the best AUC-ROC and AUC-PR values. 

This ablation study showed that it seems that network structure alone is not enough to come to a performant AML model. There is an important interplay between the network and the node features, pointing to a need to combine the two. Additionally, it highlighted the strong dependence of GNNs on these node features—except for the GIN in some cases—and on the data generation process. The explicit correlation between the money laundering cases and feature values for the IBM data set resulted in sub-optimal results when only considering the network structure. 

### **5.4 Model Stability** 

As a final analysis, we look at the range of the results to assess model stability. Figure 9 present the box plots of the results for the models on the ten different training sets. 

27 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table 22: Threshold-independent metrics: AUC-ROC and AUC-PR values over the different methods for the two data sets, based on the test set. The standard deviation is also reported. 



<!-- Start of picture text -->
Supervised Elliptic IBM-AML<br>Methods AUC-ROC AUC-PR AUC-ROC AUC-PR<br>Egonet features 0 . 5254  ±  0 . 0120 0 . 0226  ±  0 . 0008 0 . 4737  ±  0 . 0057 0 . 0027  ±  0 . 0000<br>DeepWalk 0 . 4337  ±  0 . 0064 0 . 0834  ±  0 . 0013 0 . 5380  ±  0 . 0081 0 . 0031  ±  0 . 0001<br>Node2vec 0 . 4231  ±  0 . 0050 0 . 0815  ±  0 . 0010 0 . 4910  ±  0 . 0092 0 . 0026  ±  0 . 0001<br>AUC-ROC for elliptic dataset AUC-ROC for ibm dataset<br>0.875 0.7<br>0.850 0.6<br>0.5<br>0.825<br>0.4<br>0.800<br>0.3<br>0.775<br>Method Method<br>AUC-PR for elliptic dataset AUC-PR for ibm dataset<br>0.7<br>0.020<br>0.6<br>0.015<br>0.5<br>0.010<br>0.4<br>0.005<br>Method Method<br>intrinsic positionaldeepwalknode2vec gcn sage gat gin intrinsic positionaldeepwalknode2vec gcn sage gat gin<br>intrinsic positionaldeepwalknode2vec gcn sage gat gin intrinsic positionaldeepwalknode2vec gcn sage gat gin<br>AUC AUC<br>AP AP<br><!-- End of picture text -->

Figure 9: Box plots of the AUC-ROC and AUC-PR for the ten different training sets. 

The results are quite stable on the Elliptic data set, although the AUC-ROC shows a bit more variability for the GNNs. The values on the IBM data set show strong instability for all GNNs, while the other models are relatively stable. The GNNs are probably less stable on the IBM data set due to the larger class imbalance. 

## **6 Conclusion** 

### **6.1 Literature Review** 

This paper present a literature review on network analytics for anti-money laundering. The 97 selected papers were processed in two steps to achieve (1) a comprehensive understanding of the broader literature and (2) detailed insight into research that has been well received by the community. The comprehensive analysis classified all papers according to seven categories, each having different sub-characteristics. The detailed analysis was performed on the 10% most-cited papers in scope, which were further categorized using the framework recently proposed by Bockel-Rickermann et al. [14]. Information was given on the different processing steps taken, the challenges faced or addressed in the papers, the experiments performed, and the definition of nodes and edges in the networks. 

A large part of the literature concerns methods that rely heavily on expert-knowledge by applying manual feature engineering and rule-based methods. Most of these are based on basic network centrality metrics. 

The analysis and comparison of the models is mostly done using threshold-dependent metric, most notably accuracy, precision and recall. Given the class imbalance, we strongly advise against the use of accuracy as an evaluation metric 

28 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

for AML models. Instead, the AML research should adopt the threshold-independent AUC-PR as a standard evaluation metric, as this is well suited for model comparison in the presence of high class imbalance. 

Our analysis of the literature identified six critical gaps that future research must address: 

1. **Benchmark** : There is a lack of comparison of novel methods with other models in the literature. This shows that there is a need for a unified approach to perform test including standard open-source data sets on which to test as well as a consensus on which methods can be seen as baseline models and which as state-of-the-art. 

2. **Interpretability** : Few studies have explored tools to explain the outputs of graph neural networks (GNNs), which is essential for regulatory and operational transparency. 

3. **Unsupervised Methods** : Limited exploration of classic unsupervised methods restricts their potential to detect anomalies, a key requirement as money laundering tactics evolve. 

4. **Dynamic Network Analysis** : Most approaches rely on static networks, with insufficient research into dynamic methods that could better capture the temporal nature of transactions. 

5. **Fraud Specific Methods** : The inherent incompleteness of AML labels, due to undetected cases and resource constraints, highlights the need for specialized PU-learning techniques, while limited resources necessitate actionable, prioritized predictions, which could be achieved through learning-to-rank methods. 

6. **Adoption State-of-the-Art:** AML is often done using basic network metrics, while new developments in the field are adopted slowly. There is a need for standardising the use of heterogeneous and dynamic networks in AML. Future work should study how this can be achieved by comparing the latest state-of-the-art in continual graph learning, graph contrastive learning, heterogeneous and temporal GNNs, and graph transformers for AML, to name a few. 

### **6.2 Experimental Evaluation** 

To extend upon the existing body of knowledge and to partially address the gaps in the literature, we implemented an extensive experimental evaluation covering manual feature engineering, shallow and deep representation learning, both in a supervised and unsupervised setting. We can that network features bring additional predictive power to AML models, when combined with the node features. However, the GNNs struggled when faced with extreme class imbalance and with a network with many hubs. 

The observed performance improvements were limited, but even a small relative increase in performance has material implications for the business. The efficiency of AML practices increases significantly since investigators have more high quality predictions, and spend less time on the investigation of false positives. 

In the unsupervised learning experiments, we employed isolation forests and observed notable differences between the real-world Elliptic data set and the synthetic IBM-AML data set. Although on synthetic data, the methods performed well, the observations from the Elliptic data seemed to exhibit strong camouflaging tactics. The anomaly scores of the latter were so low that investigating the least suspicious transactions for money laundering became the better strategy. We conclude from this that care should be taken when testing methods on synthetic data, since it can give overly optimistic results. 

Our experiments on unsupervised learning are, however, limited. We only applied an isolation forest, which is a global anomaly detection method. Future work should focus on extending this to a larger benchmark, in line with the work of Lorenz et al. [83], to analyse if the same conclusions hold for other global and local anomaly detection methods. 

The additional ablation study highlighted the need to incorporate both the network and feature information. When excluding the node features, the model predictions are as good as random. While this approach is common in the literature, we recommend against relying solely on the network structure. 

### **6.3 Limitations** 

There are limitations to our work that should be addressed in future research. 

First, conclusions are drawn based on only two data sets. To generalise our conclusions, additional tests need to be performed on a broader range of data sets. This can be done by including more open-source data sets when these become available, and rely on practitioners to use our code to test the methods on their proprietary data and publish the results. 

29 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Second, the decoders were always based on neural networks. The application and impact analysis of applying different machine learning methods as down-stream classifiers is another interesting area of study to extend the experiments presented here. 

Third, the hyperparameter tuning process was limited. While Optuna provided an efficient method for hyperparameter search, the 50 to 100 tuning rounds used in this study may be insufficient. The choices in this paper were made to keep everything tractable and feasible given the limited resources available. Future work could expand the parameter ranges and number of tuning rounds. 

## **Acknowledgments** 

This work was supported by the Research Foundation – Flanders (FWO research project 1SHEN24N) and by the BNP Paribas Fortis Chair in Fraud Analytics. The resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government. 

## **References** 

- [1] Florian Adriaens, Cigdem Aslay, Tijl De Bie, Aristides Gionis, and Jefrey Lijffijt. Discovering interesting cycles in directed graphs. In _Proceedings of the 28th ACM International Conference on Information and Knowledge Management_ , pages 1191–1200, 2019. 

- [2] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , 2019. 

- [3] Leman Akoglu, Hanghang Tong, and Danai Koutra. Graph based anomaly detection and description: a survey. _Data Mining and Knowledge Discovery_ , 29(3):626–688, 2015. doi:10.1007/s10618-014-0365-y. URL `https://doi.org/10.1007/s10618-014-0365-y` . 

- [4] Ismail Alarab, Simant Prakoonwit, and Mohamed Ikbal Nacer. Comparative analysis using supervised learning methods for anti-money laundering in bitcoin. In _Proceedings of the 2020 5th International Conference on Machine Learning Technologies_ , ICMLT ’20, page 11–17, Beijing, China, 2020. Association for Computing Machinery. ISBN 9781450377645. doi:10.1145/3409073.3409078. URL `https://doi.org/10.1145/ 3409073.3409078` . 

- [5] Ismail Alarab, Simant Prakoonwit, and Mohamed Ikbal Nacer. Competence of graph convolutional networks for anti-money laundering in bitcoin blockchain. In _Proceedings of the 2020 5th International Conference on Machine Learning Technologies_ , ICMLT ’20, page 23–27, Beijing, China, 2020. Association for Computing Machinery. ISBN 9781450377645. doi:10.1145/3409073.3409080. URL `https://doi.org/10.1145/ 3409073.3409080` . 

- [6] Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Beni Egressy, Andreea Anghel, and Kubilay Atasu. Realistic synthetic financial transactions for anti-money laundering models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems_ , volume 36, pages 29851–29874. Curran Associates, Inc., 2023. URL `https://proceedings.neurips.cc/paper_files/paper/2023/file/ 5f38404edff6f3f642d6fa5892479c42-Paper-Datasets_and_Benchmarks.pdf` . 

- [7] Eugenio Angriman, Alexander van der Grinten, Michael Hamann, Henning Meyerhenke, and Manuel Penschuck. _Algorithms for Large-Scale Network Analysis and the NetworKit Toolkit_ , pages 3–20. Springer Nature Switzerland, Cham, 2022. ISBN 978-3-031-21534-6. doi:10.1007/978-3-031-21534-6_1. URL `https://doi.org/10.1007/978-3-031-21534-6_1` . 

- [8] Henrique S Assumpção, Fabrício Souza, Leandro Lacerda Campos, Vinícius T de Castro Pires, Paulo M Laurentys de Almeida, and Fabricio Murai. Delator: Money laundering detection via multi-task learning on large transaction graphs. In _2022 IEEE International Conference on Big Data (Big Data)_ , pages 709–714. IEEE, 2022. doi:10.1109/BigData55660.2022.10021010. 

- [9] LAa Badalov, SDa Belov, and ISa Kadochnikov. Checking foreign counterparty companies using big data. In _CEUR Workshop Proceedings_ , volume 2267, pages 523–527, 2018. 

- [10] Bart Baesens, Veronique Van Vlasselaer, and Wouter Verbeke. _Fraud analytics using descriptive, predictive, and social network techniques: a guide to data science for fraud detection_ . John Wiley & Sons, Inc, 2015. ISBN 9781119133124. doi:10.1002/9781119146841. 

30 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [11] Ashwin Bahulkar, N Orkun Baycik, Thomas Sharkey, Yeming Shen, Boleslaw Szymanski, and William Wallace. Integrative analytics for detecting and disrupting transnational interdependent criminal smuggling, money, and money-laundering networks. In _2018 IEEE International Symposium on Technologies for Homeland Security (HST)_ , pages 1–6. IEEE, 2018. doi:10.1109/THS.2018.8574121. 

- [12] Luigi Bellomarini, Davide Magnanimi, Markus Nissl, and Emanuel Sallinger. Neither in the programs nor in the data: Mining the hidden financial knowledge with knowledge graphs and reasoning. In _Mining Data for Financial Applications: 5th ECML PKDD Workshop, MIDAS 2020, Ghent, Belgium, September 18, 2020, Revised Selected Papers 5_ , pages 119–134. Springer, 2021. 

- [13] Rasika Bhalerao, Maxwell Aliapoulios, Ilia Shumailov, Sadia Afroz, and Damon McCoy. Mapping the underground: Supervised discovery of cybercrime supply chains. In _2019 APWG Symposium on Electronic Crime Research (eCrime)_ , pages 1–16. IEEE, 2019. doi:10.1109/eCrime47957.2019.9037582. 

- [14] Christopher Bockel-Rickermann, Tim Verdonck, and Wouter Verbeke. Fraud analytics: A decade of research: Organizing challenges and solutions in the field. _Expert Systems with Applications_ , 232:120605, 2023. ISSN 09574174. doi:https://doi.org/10.1016/j.eswa.2023.120605. URL `https://www.sciencedirect.com/science/ article/pii/S0957417423011077` . 

- [15] Richard J. Bolton and David J. Hand. Statistical Fraud Detection: A Review. _Statistical Science_ , 17(3):235 – 255, 2002. doi:10.1214/ss/1042727940. URL `https://doi.org/10.1214/ss/1042727940` . 

- [16] Shaked Brody, Uri Alon, and Eran Yahav. How attentive are graph attention networks? _CoRR_ , abs/2105.14491, 2021. URL `https://arxiv.org/abs/2105.14491` . 

- [17] Mehmet Ça˘glayan and ¸Serif Bahtiyar. Money laundering detection with node2vec. _Gazi University Journal of Science_ , 35(3):854–873, 2022. 

- [18] Hongyun Cai, Vincent W Zheng, and Kevin Chen-Chuan Chang. A comprehensive survey of graph embedding: Problems, techniques, and applications. _IEEE transactions on knowledge and data engineering_ , 30(9):1616–1637, 2018. doi:10.1109/TKDE.2018.2807452. 

- [19] Ramiro Daniel Camino, Radu State, Leandro Montero, and Petko Valtchev. Finding suspicious activities in financial transactions and distributed ledgers. In _2017 IEEE International Conference on Data Mining Workshops (ICDMW)_ , pages 787–796. IEEE, 2017. doi:10.1109/ICDMW.2017.109. 

- [20] Mário Cardoso, Pedro Saleiro, and Pedro Bizarro. Laundrograph: Self-supervised graph representation learning for anti-money laundering. In _Proceedings of the Third ACM International Conference on AI in Finance_ , ICAIF ’22, pages 130–138, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393768. doi:10.1145/3533271.3561727. URL `https://doi.org/10.1145/3533271.3561727` . 

- [21] Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. _ACM Comput. Surv._ , 41(3), July 2009. ISSN 0360-0300. doi:10.1145/1541880.1541882. URL `https://doi.org/10.1145/1541880. 1541882` . 

- [22] Remco Chang, Mohammad Ghoniem, Robert Kosara, William Ribarsky, Jing Yang, Evan Suma, Caroline Ziemkiewicz, Daniel Kern, and Agus Sudjianto. Wirevis: Visualization of categorical, time-varying data from financial transactions. In _2007 IEEE symposium on visual analytics science and technology_ , pages 155–162. IEEE, 2007. doi:10.1109/VAST.2007.4389009. 

- [23] Tat-Man Cheong and Yain-Whar Si. Event-based approach to money laundering data analysis and visualization. In _Proceedings of the 3rd International Symposium on Visual Information Communication_ , VINCI ’10, Beijing, China, 2010. Association for Computing Machinery. ISBN 9781450304368. doi:10.1145/1865841.1865869. URL `https://doi.org/10.1145/1865841.1865869` . 

- [24] Mariusz Chmielewski and Piotr Stkapor. Money laundering analytics based on contextual analysis. application of problem solving ontologies in financial fraud identification and recognition. In _Information Systems Architecture and Technology: Proceedings of 37th International Conference on Information Systems Architecture and Technology–ISAT 2016–Part I_ , pages 29–39. Springer, 2017. 

- [25] Elliptic Dataset. Source code for torch_geometric.datasets.elliptic. `https://pytorch-geometric. readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html` . Accessed: 2024-01-31. 

- [26] Jesse Davis and Mark Goadrich. The relationship between precision-recall and roc curves. In _Proceedings of the 23rd International Conference on Machine Learning_ , ICML ’06, page 233–240, Pittsburgh, Pennsylvania, USA, 2006. ISBN 1595933832. doi:10.1145/1143844.1143874. URL `https://doi.org/10.1145/1143844. 1143874` . 

31 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [27] Min-Yuh Day. Artificial intelligence for knowledge graphs of cryptocurrency anti-money laundering in fintech. In _Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining_ , ASONAM ’21, page 439–446, Virtual Event, Netherlands, 2022. Association for Computing Machinery. ISBN 9781450391283. doi:10.1145/3487351.3488415. URL `https://doi.org/10.1145/ 3487351.3488415` . 

- [28] Floris Devriendt, Jente Van Belle, Tias Guns, and Wouter Verbeke. Learning to rank for uplift modeling. _IEEE Transactions on Knowledge and Data Engineering_ , 34(10):4888–4904, 2022. doi:10.1109/TKDE.2020.3048510. 

- [29] Walter Didimo, Giuseppe Liotta, Fabrizio Montecchiani, and Pietro Palladino. An advanced network visualization system for financial crime detection. In _2011 IEEE Pacific visualization symposium_ , pages 203–210. IEEE, 2011. doi:10.1109/PACIFICVIS.2011.5742391. 

- [30] Walter Didimo, Giuseppe Liotta, and Fabrizio Montecchiani. Vis4aui: Visual analysis of banking activity networks. In _Proceedings of the International Conference on Computer Graphics Theory and Applications and International Conference on Information Visualization Theory and Applications - Volume 1: IVAPP, (VISIGRAPP 2012)_ , pages 799–802. INSTICC, SciTePress, 2012. ISBN 978-989-8565-02-0. doi:10.5220/0003933407990802. 

- [31] Walter Didimo, Luca Grilli, Giuseppe Liotta, Fabrizio Montecchiani, and Daniele Pagliuca. Visual querying and analysis of temporal fiscal networks. _Information Sciences_ , 505:406–421, 2019. ISSN 0020-0255. doi:https://doi.org/10.1016/j.ins.2019.07.097. URL `https://www.sciencedirect.com/science/article/ pii/S0020025519307182` . 

- [32] Peng Dong, Marie Loh, and Adrian Mondry. The "impact factor" revisited. _Biomedical Digital Libraries_ , 2(1):7, 2005. doi:10.1186/1742-5581-2-7. URL `https://doi.org/10.1186/1742-5581-2-7` . 

- [33] Rafał Dre˙zewski, Jan Sepielak, and Wojciech Filipkowski. The application of social network analysis algorithms in a system supporting money laundering detection. _Information Sciences_ , 295:18–32, 2015. ISSN 0020-0255. doi:https://doi.org/10.1016/j.ins.2014.10.015. URL `https://www.sciencedirect.com/science/article/ pii/S0020025514009979` . 

- [34] Bogdan Dumitrescu, Andra B˘altoiu, and ¸Stefania Budulan. Anomaly detection in graphs of bank transactions for anti money laundering applications. _IEEE Access_ , 10:47699–47714, 2022. doi:10.1109/ACCESS.2022.3170467. 

- [35] The Economist. Losing the war, April 2021. URL `https://www.economist.com/ finance-and-economics/2021/04/12/the-war-against-money-laundering-is-being-lost` . 

- [36] Béni Egressy, Luc von Niederhäusern, Jovan Blanusa, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. Provably powerful graph neural networks for directed multigraphs, 2024. 

- [37] Elliptic. Elliptic. `www.elliptic.co` . Accessed: 2024-01-31. 

- [38] Europol. Does crime still pay? criminal asset recovery in the eu, 2016. 

- [39] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In _ICLR Workshop on Representation Learning on Graphs and Manifolds_ , 2019. 

- [40] Emily Fletcher, Charles Larkin, and Shaen Corbet. Countering money laundering and terrorist financing: A case for bitcoin regulation. _Research in International Business and Finance_ , 56:101387, 2021. ISSN 02755319. doi:https://doi.org/10.1016/j.ribaf.2021.101387. URL `https://www.sciencedirect.com/science/ article/pii/S0275531921000088` . 

- [41] Andrea Fronzetti Colladon and Elisa Remondi. Using social network analysis to prevent money laundering. _Expert Systems with Applications_ , 67:49–58, 2017. ISSN 0957-4174. doi:https://doi.org/10.1016/j.eswa.2016.09.029. URL `https://www.sciencedirect.com/science/ article/pii/S0957417416305139` . 

- [42] Zengan Gao and Mao Ye. A framework for data mining-based anti-money laundering research. _Journal of Money Laundering Control_ , 10(2):170–179, 2007. doi:10.1108/13685200710746875. URL `https://doi.org/10. 1108/13685200710746875` . 

- [43] Ignacio González García and Alfonso Mateos. Use of social network analysis for tax control in spain. _Hacienda Publica Espanola_ , (239):159–197, 2021. 

- [44] Olmer Garcia-Bedoya, Oscar Granados, and José Cardozo Burgos. Ai against money laundering networks: the colombian case. _Journal of Money Laundering Control_ , 24(1):49–62, 2021. 

- [45] Javier Garcia-Bernardo, Joost Witteman, and Marilou Vlaanderen. Uncovering the size of the illegal corporate service provider industry in the netherlands: a network approach. _EPJ Data Science_ , 11(1):23, 2022. 

32 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [46] Ayshan Gasanova, Alexander N Medvedev, Evgeny I Komotskiy, Kamen B Spasov, and Igor N Sachkov. On the use of data mining methods for money laundering detection based on financial transactions information. In _AIP Conference Proceedings_ , volume 2040, page 050021. AIP Publishing LLC, 2018. 

- [47] Peter Gerbrands, Brigitte Unger, Michael Getzner, and Joras Ferwerda. The effect of anti-money laundering policies: an empirical network analysis. _EPJ Data Science_ , 11(1):15, 2022. 

- [48] Yanan Gong, Kam-Pui Chow, Hing-Fung Ting, and Siu-Ming Yiu. Analyzing the error rates of bitcoin clustering heuristics. In _Advances in Digital Forensics XVIII: 18th IFIP WG 11.9 International Conference, Virtual Event, January 3–4, 2022, Revised Selected Papers_ , pages 187–205. Springer, 2022. 

- [49] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. _Deep learning_ . MIT press, 2016. 

- [50] Palash Goyal and Emilio Ferrara. Graph embedding techniques, applications, and performance: A survey. _Knowledge-Based Systems_ , 151:78–94, 2018. ISSN 0950-7051. doi:https://doi.org/10.1016/j.knosys.2018.03.022. URL `https://www.sciencedirect.com/science/article/pii/S0950705118301540` . 

- [51] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? _Advances in neural information processing systems_ , 35:507–520, 2022. 

- [52] Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , KDD ’16, page 855–864, San Francisco, California, USA, 2016. Association for Computing Machinery. ISBN 9781450342322. doi:10.1145/2939672.2939754. URL `https://doi.org/10.1145/2939672.2939754` . 

- [53] Andrea Gulino, Stefano Ceri, Georg Gottlob, Emanuel Sallinger, and Luigi Bellomarini. Distributed company control in company shareholding graphs. In _2021 IEEE 37th International Conference on Data Engineering (ICDE)_ , pages 2637–2648. IEEE, 2021. doi:10.1109/ICDE51399.2021.00294. 

- [54] Aric A. Hagberg, Daniel A. Schult, and Pieter J. Swart. Exploring network structure, dynamics, and function using networkx. In Gaël Varoquaux, Travis Vaught, and Jarrod Millman, editors, _Proceedings of the 7th Python in Science Conference_ , pages 11 – 15, Pasadena, CA USA, 2008. 

- [55] Oussama H Hamid. Breaking through opacity: A context-aware data-driven conceptual design for a predictive anti money laundering system. In _2017 9th IEEE-GCC conference and exhibition (GCCCE)_ , pages 1–9. IEEE, 2017. doi:10.1109/IEEEGCC.2017.8448084. 

- [56] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. 30, 2017. URL `https://proceedings.neurips.cc/paper_files/paper/2017/file/ 5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf` . 

- [57] HuaLong Han, YuPeng Chen, ChenYing Guo, and Yin Zhang. Blockchain abnormal transaction behavior analysis: a survey. In _Blockchain and Trustworthy Systems: Third International Conference, BlockSys 2021, Guangzhou, China, August 5–6, 2021, Revised Selected Papers 3_ , pages 57–69. Springer, 2021. 

- [58] Jing He, Jiao Tian, Yuanyuan Wu, Xinyi Cia, Kai Zhang, Mengjiao Guo, Hui Zheng, Junfeng Wu, and Yimu Ji. An efficient solution to detect common topologies in money launderings based on coupling and connection. _IEEE Intelligent Systems_ , 36(1):64–74, 2021. doi:10.1109/MIS.2021.3057590. 

- [59] Tamer Hossam Helmy, Mohamed Zaki, Tarek Salah, and Khaled Badran. Design of a monitor for detecting money laundering and terrorist financing. _Journal of Theoretical & Applied Information Technology_ , 85(3), 2016. 

- [60] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. _Neural Networks_ , 4(2):251– 257, 1991. ISSN 0893-6080. doi:https://doi.org/10.1016/0893-6080(91)90009-T. URL `https://www. sciencedirect.com/science/article/pii/089360809190009T` . 

- [61] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. _Neural Networks_ , 2(5):359–366, 1989. ISSN 0893-6080. doi:https://doi.org/10.1016/0893-6080(89)90020-8. URL `https://www.sciencedirect.com/science/ article/pii/0893608089900208` . 

- [62] Maryam Imanpour, Stephanie Rosenkranz, Bastian Westbrock, Brigitte Unger, and Joras Ferwerda. A microeconomic foundation for optimal money laundering policies. _International Review of Law and Economics_ , 60:105856, 2019. ISSN 0144-8188. doi:https://doi.org/10.1016/j.irle.2019.105856. URL `https: //www.sciencedirect.com/science/article/pii/S0144818818302643` . 

- [63] Angela SM Irwin and Adam B Turner. Illicit bitcoin transactions: challenges in getting to the who, what, when and where. _Journal of money laundering control_ , 21(3):297–313, 2018. URL `https://doi.org/10.1108/ JMLC-07-2017-0031` . 

33 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [64] Qing-Xuan Jia, Chun-Xu Chen, Xin Gao, Xin-Peng Li, Bo Yan, Guan-Qun Ai, Jun-Liang Li, and Jian-Hang Xu. Anomaly detection method using center offset measurement based on leverage principle. _KnowledgeBased Systems_ , 190:105191, 2020. ISSN 0950-7051. doi:https://doi.org/10.1016/j.knosys.2019.105191. URL `https://www.sciencedirect.com/science/article/pii/S0950705119305301` . 

- [65] Chengxiang Jin, Jie Jin, Jiajun Zhou, Jiajing Wu, and Qi Xuan. Heterogeneous feature augmentation for ponzi detection in ethereum. _IEEE Transactions on Circuits and Systems II: Express Briefs_ , 69(9):3919–3923, 2022. doi:10.1109/TCSII.2022.3177898. 

- [66] Hai Jin, Chenchen Li, Jiang Xiao, Teng Zhang, Xiaohai Dai, and Bo Li. Detecting arbitrage on ethereum through feature fusion and positive-unlabeled learning. _IEEE Journal on Selected Areas in Communications_ , 40(12): 3660–3671, 2022. doi:10.1109/JSAC.2022.3213335. 

- [67] Yaqin Jin and Zhenxin Qu. Research on anti-money laundering hierarchical model. In _2018 IEEE 9th International Conference on Software Engineering and Service Science (ICSESS)_ , pages 406–411. IEEE, 2018. doi:10.1109/ICSESS.2018.8663895. 

- [68] Mikel Joaristi, Edoardo Serra, and Francesca Spezzano. Inferring bad entities through the panama papers network. In _2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)_ , pages 767–773. IEEE, 2018. doi:10.1109/ASONAM.2018.8508497. 

- [69] Mikel Joaristi, Edoardo Serra, and Francesca Spezzano. Detecting suspicious entities in offshore leaks networks. _Social Network Analysis and Mining_ , 9:1–15, 2019. 

- [70] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks, 2017. 

- [71] Jerzy Korczak and Walter Łuszczyk. Visual exploration of cash flow chains. In _2011 Federated Conference on Computer Science and Information Systems (FedCSIS)_ , pages 41–46. IEEE, 2011. 

- [72] Aditya Kurniawan, Muhammad Yusuf Handeu, Bayu Alvian, and Ade Andriani Renouw. Information system for depicting relations between banking customer with risk: Study case in indonesia. In _2015 3rd International Conference on Information and Communication Technology (ICoICT)_ , pages 125–128. IEEE, 2015. doi:10.1109/ICoICT.2015.7231409. 

- [73] Eren Kurshan and Hongda Shen. Graph computing for financial crime and fraud detection: Trends, challenges and outlook. _International Journal of Semantic Computing_ , 14(04):565–589, 2020. 

- [74] Dattatray Vishnu Kute, Biswajeet Pradhan, Nagesh Shukla, and Abdullah Alamri. Deep learning and explainable artificial intelligence techniques applied for detecting money laundering–a critical review. _IEEE Access_ , 9: 82300–82317, 2021. doi:10.1109/ACCESS.2021.3086230. 

- [75] Michael Levi and Peter Reuter. Money laundering. _Crime and justice_ , 34(1):289–375, 2006. doi:10.1086/501508. 

- [76] An Li, Zhongshuai Wang, Minghao Yu, and Di Chen. Blockchain abnormal transaction detection method based on weighted sampling neighborhood nodes. In _2022 3rd International Conference on Big Data, Artificial Intelligence and Internet of Things Engineering (ICBAIE)_ , pages 746–752. IEEE, 2022. doi:10.1109/ICBAIE56435.2022.9985815. 

- [77] Qimai Li, Zhichao Han, and Xiao-ming Wu. Deeper insights into graph convolutional networks for semisupervised learning. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 32(1), Apr. 2018. doi:10.1609/aaai.v32i1.11604. URL `https://ojs.aaai.org/index.php/AAAI/article/view/11604` . 

- [78] Xiangfeng Li, Shenghua Liu, Zifeng Li, Xiaotian Han, Chuan Shi, Bryan Hooi, He Huang, and Xueqi Cheng. Flowscope: Spotting money laundering based on graphs. _Proceedings of the AAAI Conference on Artificial Intelligence_ , 34(04):4731–4738, 2020. doi:10.1609/aaai.v34i04.5906. URL `https://ojs.aaai.org/index. php/AAAI/article/view/5906` . 

- [79] Ziyu Li, Yanmei Zhang, Qian Wang, and Shiping Chen. Transactional network analysis and money laundering behavior identification of central bank digital currency of china. _Journal of Social Computing_ , 3(3):219–230, 2022. doi:10.23919/JSC.2022.0011. 

- [80] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation-based anomaly detection. _ACM Trans. Knowl. Discov. Data_ , 6(1), mar 2012. ISSN 1556-4681. doi:10.1145/2133360.2133363. URL `https://doi.org/10.1145/ 2133360.2133363` . 

- [81] Xuan Liu, Jia Li, Zhigao Chen, and Pengzhu Zhang. Research on financial super-network model based on variational inequalities. In Michael J. Shaw, Dongsong Zhang, and Wei T. Yue, editors, _E-Life: Web-Enabled Convergence of Commerce, Work, and Social Life_ , pages 66–76, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg. ISBN 978-3-642-29873-8. 

34 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [82] Mark E. Lokanan. Financial fraud detection: the use of visualization techniques in credit card fraud and money laundering domains. _Journal of Money Laundering Control_ , 26(3):436–444, 2023. doi:10.1108/JMLC-04-20220058. URL `https://doi.org/10.1108/JMLC-04-2022-0058` . 

- [83] Joana Lorenz, Maria Inês Silva, David Aparício, João Tiago Ascensão, and Pedro Bizarro. Machine learning methods to detect money laundering in the bitcoin blockchain in the presence of label scarcity. In _Proceedings of the First ACM International Conference on AI in Finance_ , ICAIF ’20, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450375849. doi:10.1145/3383455.3422549. URL `https://doi.org/ 10.1145/3383455.3422549` . 

- [84] Pritheega Magalingam, Stephen Davis, and Asha Rao. Using shortest path to discover criminal community. _Digital Investigation_ , 15:1–17, 2015. ISSN 1742-2876. doi:https://doi.org/10.1016/j.diin.2015.08.002. URL `https://www.sciencedirect.com/science/article/pii/S1742287615000894` . Special Issue: Big Data and Intelligent Data Analysis. 

- [85] Shamil Magomedov, Sergei Pavelyev, Irina Ivanova, Alexey Dobrotvorsky, Marina Khrestina, and Timur Yusubaliev. Anomaly detection with machine learning and graph databases in fraud management. _International Journal of Advanced Computer Science and Applications_ , 9(11), 2018. 

- [86] Maryam Mahootiha, Alireza Hashemi Golpayegani, and Babak Sadeghian. Designing a new method for detecting money laundering based on social network analysis. In _2021 26th International Computer Conference, Computer Society of Iran (CSICC)_ , pages 1–7. IEEE, 2021. doi:10.1109/CSICC52343.2021.9420621. 

- [87] Aili Malm and Gisela Bichler. Using friends for money: the positional importance of money-launderers in organized crime. _Trends in Organized Crime_ , 16:365–381, 2013. doi:10.1007/s12117-013-9205-5. URL `https://doi.org/10.1007/s12117-013-9205-5` . 

- [88] Dan McGinn, David Birch, David Akroyd, Miguel Molina-Solana, Yike Guo, and William J Knottenbelt. Visualizing dynamic bitcoin transaction patterns. _Big data_ , 4(2):109–119, 2016. doi:10.1089/big.2015.0056. 

- [89] Murad Mehmet and Duminda Wijesekera. Data analytics to detect evolving money laundering. In _STIDS_ , pages 71–78, 2013. 

- [90] Murad Mehmet and Duminda Wijesekera. Using dynamic risk estimation & social network analysis to detect money laundering evolution. In _2013 IEEE International Conference on Technologies for Homeland Security (HST)_ , pages 310–315. IEEE, 2013. doi:10.1109/THS.2013.6699020. 

- [91] Murad Mehmet, Murat Güne¸sta¸s, and Duminda Wijesekera. Dynamic risk model of money laundering. In _Risk Assessment and Risk-Driven Testing: First International Workshop, RISK 2013, Held in Conjunction with ICTSS 2013, Istanbul, Turkey, November 12, 2013. Revised Selected Papers 1_ , pages 3–20. Springer, 2014. 

- [92] Giovanni Micale, Alfredo Pulvirenti, Alfredo Ferro, Rosalba Giugno, and Dennis Shasha. Fast methods for finding significant motifs on labelled multi-relational networks. _Journal of Complex Networks_ , 7(6):817–837, 03 2019. ISSN 2051-1329. doi:10.1093/comnet/cnz008. URL `https://doi.org/10.1093/comnet/cnz008` . 

- [93] Krzysztof Michalak and Jerzy Korczak. Graph mining approach to suspicious transaction detection. In _2011 Federated conference on computer science and information systems (FedCSIS)_ , pages 69–75. IEEE, 2011. 

- [94] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space, 2013. 

- [95] Anuraj Mohan, Karthika P. V., Parvathi Sankar, K. Maya Manohar, and Amala Peter. Improving anti-money laundering in bitcoin using evolving graph convolutions and deep neural decision forest. _Data Technologies and Applications_ , 57(3):313–329, 2023. doi:10.1108/DTA-06-2021-0167. URL `https://doi.org/10.1108/ DTA-06-2021-0167` . 

- [96] Philippe Mongeon and Adèle Paul-Hus. The journal coverage of web of science and scopus: a comparative analysis. _Scientometrics_ , 106(1):213–228, 2016. doi:10.1007/s11192-015-1765-5. URL `https://doi.org/ 10.1007/s11192-015-1765-5` . 

- [97] M Nandhini and Bikram Bikash Das. An assessment and methodology for fraud detection in online social network. In _2016 Second International Conference on Science Technology Engineering and Management (ICONSTEM)_ , pages 104–108. IEEE, 2016. doi:10.1109/ICONSTEM.2016.7560932. 

- [98] Mark Newman. _Networks: An Introduction_ . Oxford University Press, 03 2010. ISBN 9780199206650. doi:10.1093/acprof:oso/9780199206650.001.0001. URL `https://doi.org/10.1093/ acprof:oso/9780199206650.001.0001` . 

- [99] E.W.T. Ngai, Yong Hu, Y.H. Wong, Yijun Chen, and Xin Sun. The application of data mining techniques in financial fraud detection: A classification framework and an academic review of literature. _Decision Support_ 

35 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

_Systems_ , 50(3):559–569, 2011. ISSN 0167-9236. doi:https://doi.org/10.1016/j.dss.2010.08.006. URL `https: //www.sciencedirect.com/science/article/pii/S0167923610001302` . On quantitative methods for detection of financial fraud. 

- [100] Normah Omar, Ismail bin Mohamed, Zuraidah Mohd Sanusi, and Hendi Yogi Prabowo. Understanding social network analysis (sna) in fraud detection. In _Proceedings of the International Congress on Interdisciplinary Behaviour and Social Sciences_ , pages 543–548, 2014. 

- [101] Michael Ovelgönne, Chanhyun Kang, Anshul Sawant, and VS Subrahmanian. Covertness centrality in networks. In _2012 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining_ , pages 863–870. IEEE, 2012. doi:10.1109/ASONAM.2012.156. 

- [102] Brice Ozenne, Fabien Subtil, and Delphine Maucort-Boulch. The precision–recall curve overcame the optimism of the receiver operating characteristic curve in rare diseases. _Journal of clinical epidemiology_ , 68(8):855–859, 2015. 

- [103] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford InfoLab, 1999. 

- [104] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: online learning of social representations. In _Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , KDD ’14, page 701–710, New York, New York, USA, 2014. Association for Computing Machinery. ISBN 9781450329569. doi:10.1145/2623330.2623732. URL `https://doi.org/10.1145/2623330.2623732` . 

- [105] Silivanxay Phetsouvanh, Frédérique Oggier, and Anwitaman Datta. Egret: Extortion graph exploration techniques in the bitcoin network. In _2018 IEEE International conference on data mining workshops (ICDMW)_ , pages 244–251. IEEE, 2018. doi:10.1109/ICDMW.2018.00043. 

- [106] Clifton Phua, Vincent C. S. Lee, Kate Smith-Miles, and Ross W. Gayler. A comprehensive survey of data miningbased fraud detection research. _CoRR_ , abs/1009.6119, 2010. URL `http://arxiv.org/abs/1009.6119` . 

- [107] Kirill Plaksiy, Andrey Nikiforov, and Natalia Miloslavskaya. Applying big data technologies to detect cases of money laundering and counter financing of terrorism. In _2018 6th International Conference on Future Internet of Things and Cloud Workshops (FiCloudW)_ , pages 70–77. IEEE, 2018. 

- [108] Mario Alfonso Prado-Romero and Andrés Gago-Alonso. Community feature selection for anomaly detection in attributed graphs. In _Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications: 21st Iberoamerican Congress, CIARP 2016, Lima, Peru, November 8–11, 2016, Proceedings 21_ , pages 109–116. Springer, 2017. 

- [109] M Mazhar Rathore, Sushil Chaurasia, and Dhirendra Shukla. Mixers detection in bitcoin network: a step towards detecting money laundering in crypto-currencies. In _2022 IEEE International Conference on Big Data (Big Data)_ , pages 5775–5782. IEEE, 2022. doi:10.1109/BigData55660.2022.10020982. 

- [110] Piergiorgio Ricci. How economic freedom reflects on the bitcoin transaction network. _Journal of Industrial and Business Economics_ , 47(1):133–161, 2020. 

- [111] Takaya Saito and Marc Rehmsmeier. The precision-recall plot is more informative than the roc plot when evaluating binary classifiers on imbalanced datasets. _PloS one_ , 10(3):e0118432, 2015. 

- [112] A Semenov, D Doropheev, A Mazeev, and T Yusubaliev. Survey of common design approaches in aml software development. In _CEUR Workshop Proceedings_ , pages 1–9, 2017. 

- [113] Ted E Senator, Henry G Goldberg, Jerry Wooton, Matthew A Cottini, AF Umar Khan, Christina D Klinger, Winston M Llamas, Michael P Marrone, and Raphael WH Wong. Financial crimes enforcement network ai system (fais) identifying potential money laundering from reports of large cash transactions. _AI magazine_ , 16(4): 21–21, 1995. doi:10.1609/aimag.v16i4.1169. 

- [114] Abdul K Shaikh and Amril Nazir. A model for identifying relationships of suspicious customers in money laundering using social network functions. In _Proceedings of the world congress on engineering_ , volume 1, pages 4–7, 2018. 

- [115] Abdul Khalique Shaikh, Malik Al-Shamli, and Amril Nazir. Designing a relational model to identify relationships between suspicious customers in anti-money laundering (aml) using social network analysis (sna). _Journal of Big Data_ , 8:1–22, 2021. 

- [116] Yeming Shen, Thomas C. Sharkey, Boleslaw K. Szymanski, and William (Al) Wallace. Interdicting interdependent contraband smuggling, money and money laundering networks. _Socio-Economic Planning Sciences_ , 78:101068, 2021. ISSN 0038-0121. doi:https://doi.org/10.1016/j.seps.2021.101068. URL `https: //www.sciencedirect.com/science/article/pii/S0038012121000604` . 

36 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [117] Rafael Sousa Lima, André Luiz Marques Serrano, Joshua Onome Imoniana, and César Medeiros Cupertino. Identifying financial patterns of money laundering with social network analysis: A brazilian case study. _Journal of Money Laundering Control_ , 25(1):118–134, 2022. 

- [118] Michele Starnini, Charalampos E Tsourakakis, Maryam Zamanipour, André Panisson, Walter Allasia, Marco Fornasiero, Laura Li Puma, Valeria Ricci, Silvia Ronchiadin, Angela Ugrinoska, et al. Smurf-based anti-money laundering in time-evolving transaction networks. In _Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part IV 21_ , pages 171–186. Springer, 2021. 

- [119] Kai Sun, Kun Meng, and Ziqiang Zheng. Game-bc: A graph attention model for exploring bitcoin crime. In _2022 6th International Symposium on Computer Science and Intelligent Control (ISCSIC)_ , pages 342–346. IEEE, 2022. doi:10.1109/ISCSIC57216.2022.00077. 

- [120] Haseeb Tariq and Marwan Hassani. Topology-agnostic detection of temporal money laundering flows in billion-scale transactions, 2023. URL `https://arxiv.org/abs/2309.13662` . 

- [121] Jeyakumar Samantha Tharani, Eugene Yougarajah Andrew Charles, Zhé Hóu, Marimuthu Palaniswami, and Vallipuram Muthukkumarasamy. Graph based visualisation techniques for analysis of blockchain transactions. In _2021 IEEE 46th Conference on Local Computer Networks (LCN)_ , pages 427–430. IEEE, 2021. doi:10.1109/LCN52139.2021.9524878. 

- [122] Elena Tiukhova, Manon Reusens, Bart Baesens, and Monique Snoeck. Benchmarking conventional outlier detection methods. In Renata Guizzardi, Jolita Ralyté, and Xavier Franch, editors, _Research Challenges in Information Science_ , pages 597–613, Cham, 2022. Springer International Publishing. ISBN 978-3-031-05760-1. 

- [123] Elena Tiukhova, Emiliano Penaloza, María Óskarsdóttir, Bart Baesens, Monique Snoeck, and Cristián Bravo. Inflect-dgnn: Influencer prediction with dynamic graph neural networks. _IEEE Access_ , 2024. doi:10.1109/ACCESS.2024.3443533. 

- [124] Adam B Turner, Stephen McCombie, and Allon J Uhlmann. A target-centric intelligence approach to wannacry 2.0. _Journal of Money Laundering Control_ , 22(4):646–665, 2019. 

- [125] Adam B Turner, Stephen McCombie, and Allon J Uhlmann. Discerning payment patterns in bitcoin from ransomware attacks. _Journal of Money Laundering Control_ , 23(3):545–589, 2020. 

- [126] United Nations Office on Drugs and Crime. Money laundering. `https://www.unodc.org/unodc/en/ money-laundering/overview.html` . Accessed: 2023-04-07. 

- [127] Rafaël Van Belle, Sandra Mitrovi´c, and Jochen De Weerdt. Representation learning in graphs for credit card fraud detection. In _Mining Data for Financial Applications: 4th ECML PKDD Workshop, MIDAS 2019, Würzburg, Germany, September 16, 2019, Revised Selected Papers 4_ , pages 32–46. Springer, 2020. 

- [128] Rafaël Van Belle, Charles Van Damme, Hendrik Tytgat, and Jochen De Weerdt. Inductive graph representation learning for fraud detection. _Expert Systems with Applications_ , 193:116463, 2022. doi:https://doi.org/10.1016/j.eswa.2021.116463. 

- [129] Rafaël Van Belle, Bart Baesens, and Jochen De Weerdt. Catchm: A novel network-based credit card fraud detection method using node representation learning. _Decision Support Systems_ , 164:113866, 2023. ISSN 01679236. doi:https://doi.org/10.1016/j.dss.2022.113866. URL `https://www.sciencedirect.com/science/ article/pii/S0167923622001373` . 

- [130] Toon Vanderschueren, Bart Baesens, Tim Verdonck, and Wouter Verbeke. A new perspective on classification: Optimally allocating limited resources to uncertain tasks. _Decision Support Systems_ , 179:114151, 2024. ISSN 0167-9236. doi:https://doi.org/10.1016/j.dss.2023.114151. URL `https://www.sciencedirect.com/ science/article/pii/S0167923623002269` . 

- [131] Rafael B Velasco, Igor Carpanese, Ruben Interian, Octavio CG Paulo Neto, and Celso C Ribeiro. A decision support system for fraud detection in public procurement. _International Transactions in Operational Research_ , 28(1):27–47, 2021. 

- [132] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph attention networks, 2018. 

- [133] Zhonglin Wang, Niu Guiqian, Zhou Yan, and Yaqiong Mu. Detection mechanism of money laundering based on random walk and skip-grim model. In _2022 IEEE 5th International Conference on Electronic Information and Communication Technology (ICEICT)_ , pages 444–448. IEEE, 2022. doi:10.1109/ICEICT55736.2022.9909113. 

- [134] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I. Weidele, Claudio Bellei, Tom Robinson, and Charles E. Leiserson. Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics, 2019. 

37 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [135] Ollie A Williams. Watchdogs hit money launderers with record fines, Aug 2021. URL `https://www.forbes.com/sites/oliverwilliams1/2021/08/26/ money-laundering-thrived-during-covid-pandemic/` . 

- [136] Donald Winiecki, Katherine Kappelman, Bryant Hay, Mikel Joaristi, Edoardo Serra, and Francesca Spezzano. Validating bad entity ranking in the panama papers via open-source intelligence. In _2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)_ , pages 752–759. IEEE, 2020. doi:10.1109/ASONAM49781.2020.9381389. 

- [137] Jiajing Wu, Jieli Liu, Weili Chen, Huawei Huang, Zibin Zheng, and Yan Zhang. Detecting mixing services via mining bitcoin transaction network with hybrid motifs. _IEEE Transactions on Systems, Man, and Cybernetics: Systems_ , 52(4):2237–2249, 2021. 

- [138] Lei Wu, Yufeng Hu, Yajin Zhou, Haoyu Wang, Xiapu Luo, Zhi Wang, Fan Zhang, and Kui Ren. Towards understanding and demystifying bitcoin mixing services. In _Proceedings of the Web Conference 2021_ , WWW ’21, page 33–44, Ljubljana, Slovenia, 2021. ISBN 9781450383127. doi:10.1145/3442381.3449880. URL `https://doi.org/10.1145/3442381.3449880` . 

- [139] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A comprehensive survey on graph neural networks. _IEEE transactions on neural networks and learning systems_ , 32(1):4–24, 2020. doi:10.1109/TNNLS.2020.2978386. 

- [140] Pingfan Xia, Zhiwei Ni, Hongwang Xiao, Xuhui Zhu, and Peng Peng. A novel spatiotemporal prediction approach based on graph convolution neural networks and long short-term memory for money laundering fraud. _Arabian Journal for Science and Engineering_ , 47(2):1921–1937, 2022. doi:10.1007/s13369-021-06116-2. URL `https://doi.org/10.1007/s13369-021-06116-2` . 

- [141] Zhao Xiao, Yuelei Li, and Kang Zhang. Visual analysis of risks in peer-to-peer lending market. _Personal and Ubiquitous Computing_ , 22:825–838, 2018. 

- [142] Jianying Xiong and Wen Xiao. Identification of key nodes in abnormal fund trading network based on improved pagerank algorithm. _Journal of Physics: Conference Series_ , 1774(1):012001, jan 2021. doi:10.1088/17426596/1774/1/012001. URL `https://dx.doi.org/10.1088/1742-6596/1774/1/012001` . 

- [143] Jianying Xiong and Haifeng Zhong. Identification of money laundering accounts based on weighted capital flow network. _Journal of Physics: Conference Series_ , 1629(1):012023, sep 2020. doi:10.1088/17426596/1629/1/012023. URL `https://dx.doi.org/10.1088/1742-6596/1629/1/012023` . 

- [144] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks?, 2019. 

- [145] Xie Xueshuo, Wang Jiming, Ye Junyi, Fang Yaozheng, Lu Ye, Li Tao, and Wang Guiling. Awap: Adaptive weighted attribute propagation enhanced community detection model for bitcoin de-anonymization. _Applied Soft Computing_ , 109:107507, 2021. ISSN 1568-4946. doi:https://doi.org/10.1016/j.asoc.2021.107507. URL `https://www.sciencedirect.com/science/article/pii/S1568494621004300` . 

- [146] Qingqing Yang, Yuexin Xiang, Wenmao Liu, and Wei Ren. An illicit bitcoin address analysis scheme based on subgraph evolution. In _2022 IEEE 24th Int Conf on High Performance Computing & Communications; 8th Int Conf on Data Science & Systems; 20th Int Conf on Smart City; 8th Int Conf on Dependability in Sensor, Cloud & Big Data Systems & Application (HPCC/DSS/SmartCity/DependSys)_ , pages 679–686. IEEE, 2022. doi:10.1109/HPCC-DSS-SmartCity-DependSys57074.2022.00116. 

- [147] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph embeddings. In Maria Florina Balcan and Kilian Q. Weinberger, editors, _Proceedings of The 33rd International Conference on Machine Learning_ , volume 48 of _Proceedings of Machine Learning Research_ , pages 40–48, New York, New York, USA, 20–22 Jun 2016. PMLR. URL `https://proceedings.mlr.press/v48/yanga16. html` . 

- [148] Yahan Yu, Yixuan Xu, Jian Wang, Zhenxing Li, and Bin Cao. Anti-money laundering risk identification of financial institutions based on aspect-level graph neural networks. In _2022 IEEE 22nd International Conference on Software Quality, Reliability, and Security Companion (QRS-C)_ , pages 542–546. IEEE, 2022. doi:10.1109/QRSC57518.2022.00086. 

- [149] Shilei Zhang, Toyotaro Suzumura, and Li Zhang. Dyngraphtrans: Dynamic graph embedding via modified universal transformer networks for financial transaction data. In _2021 IEEE International Conference on Smart Data Services (SMDS)_ , pages 184–191. IEEE, 2021. doi:10.1109/SMDS53860.2021.00032. 

- [150] Maria Zhdanova, Jürgen Repp, Roland Rieke, Chrystel Gaber, and Baptiste Hemery. No smurfs: Revealing fraud chains in mobile money transfers. In _2014 Ninth International Conference on Availability, Reliability and Security_ , pages 11–20. IEEE, 2014. doi:10.1109/ARES.2014.10. 

38 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

- [151] Dawei Zhou, Si Zhang, Mehmet Yigit Yildirim, Scott Alcorn, Hanghang Tong, Hasan Davulcu, and Jingrui He. A local algorithm for structure-preserving graph cut. In _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , KDD ’17, page 655–664, Halifax, NS, Canada, 2017. ISBN 9781450348874. doi:10.1145/3097983.3098015. URL `https://doi.org/10.1145/3097983.3098015` . 

- [152] Jiajun Zhou, Chenkai Hu, Jianlei Chi, Jiajing Wu, Meng Shen, and Qi Xuan. Behavior-aware account deanonymization on ethereum interaction graph. _IEEE Transactions on Information Forensics and Security_ , 17: 3433–3448, 2022. doi:10.1109/ASONAM.2012.156. 

- [153] Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. Graph neural networks: A review of methods and applications. _AI Open_ , 1:57–81, 2020. ISSN 2666-6510. doi:https://doi.org/10.1016/j.aiopen.2021.01.001. URL `https://www.sciencedirect. com/science/article/pii/S2666651021000012` . 

- [154] Yadong Zhou, Ximi Wang, Junjie Zhang, Peng Zhang, Lili Liu, Huan Jin, and Hongbo Jin. Analyzing and detecting money-laundering accounts in online social networks. _IEEE Network_ , 32(3):115–121, 2017. doi:10.1109/MNET.2017.1700213. 

39 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Data - Crypto<br>Proprietary<br>Open-source<br>1.0<br>Synthetic<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0 1<br>Bitcoin/Crypto<br><!-- End of picture text -->

Figure A1: The data for papers dealing with crypto currencies (1) and those that do not (0). 

## **A Analysis of Categories between Crypto and Non-Crypto Papers** 

This part of the appendix gives the plots in which we compare the difference in nature of the crypto-literature compared non-crypto-related research. 



<!-- Start of picture text -->
Data without Crypto Data Crypto<br>10 Proprietary 10 Proprietary<br>Open-source Open-source<br>Synthetic Synthetic<br>8 8<br>6 6<br>4 4<br>2 2<br>0 0<br>Year Year<br>1995 2007 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2016 2017 2018 2019 2020 2021 2022<br><!-- End of picture text -->

(a) The evolution of the data over the years for papers not (b) The evolution of the data over the years for papers covcovering crypto currencies. ering crypto currencies. 

Figure A2: Evolution of the data. 

40 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Data<br>Non-Crypto<br>Crypto<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>Community detection Flow/Chain Detection Transaction classification Client classification<br><!-- End of picture text -->

Figure A3: The distribution of the objective of the papers. 



<!-- Start of picture text -->
Data<br>0.5 Non-Crypto<br>Crypto<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>AUPRC TPR FPR Accuracy Precision F1 Recall AUROC Time<br><!-- End of picture text -->

Figure A4: The distribution of the evaluation metrics used. 

41 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

## **B The Top-Cited Papers** 

In this part of the appendix, we give the figures that summarise the top-cited papers. 



<!-- Start of picture text -->
Method Type<br>0.7<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>Semi-Supervised Mixed Supervised Visualisation Unsupervised<br><!-- End of picture text -->

Figure B1: The methods for the top-cited papers. 



<!-- Start of picture text -->
Modelling Method<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>Shallow repr. Neural Networks Deep repr. Correlation-based Tree-based SVM-based Walk-based Anomaly detection ClusteringLogistic regression Rule-based Manual features<br><!-- End of picture text -->

Figure B2: The modelling methods for the top-cited papers. 



<!-- Start of picture text -->
Metric<br>0.200<br>0.175<br>0.150<br>0.125<br>0.100<br>0.075<br>0.050<br>0.025<br>0.000<br>Accuracy AUPRC AUROC TPR FPR Precision Recall F1 Time<br><!-- End of picture text -->

Figure B3: The performance metrics for the top-cited papers. 

42 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 



<!-- Start of picture text -->
Objective<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>Community detection Transaction classification Flow/Chain Detection Client classification<br><!-- End of picture text -->

Figure B4: The objective of the top-cited papers. 



<!-- Start of picture text -->
Data<br>0.6<br>0.5<br>0.4<br>0.3<br>0.2<br>0.1<br>0.0<br>Synthetic Proprietary Open-source<br><!-- End of picture text -->

Figure B5: The nature of the data of the top-cited papers. 

43 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table C1: Threshold-dependent metrics: Precision, recall and F1-score values for the top 0 _._ 1% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._5396_±_0_._3093|0_._0204_±_0_._0302|0_._0383_±_0_._0542|
|Egonet features|0_._6833_±_0_._3187|0_._0129_±_0_._0060|0_._0253_±_0_._0118|
|DeepWalk|0_._5333_±_0_._3148|0_._0101_±_0_._0059|0_._0198_±_0_._0117|
|Node2vec|0_._5167_±_0_._3352|0_._0097_±_0_._0063|0_._0191_±_0_._0124|
|GCN|**1****_._0000****_±_ 0****_._0000**|0_._0189_±_0_._0007|0_._0371_±_0_._0014|
|GraphSAGE|0_._9624_±_0_._0615|**0****_._0300****_±_ 0****_._0071**|**0****_._0580****_±_ 0****_._0133**|
|GAT|**1****_._0000****_±_ 0****_._0000**|0_._0187_±_0_._0005|0_._0367_±_0_._0010|
|GIN|0_._8500_±_0_._2540|0_._0191_±_0_._0093|0_._0372_±_0_._0179|



Table C2: Threshold-dependent metrics: Precision, recall and F1-score values for the top 0 _._ 1% scores over the different methods for the IBM-AML data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._0050_±_0_._0087|0_._0007_±_0_._0011|0_._0012_±_0_._0019|
|Egonet features|0_._0079_±_0_._0085|0_._0011_±_0_._0012|0_._0019_±_0_._0020|
|DeepWalk|0_._0076_±_0_._0085|0_._0011_±_0_._0013|0_._0019_±_0_._0022|
|Node2vec|0_._0035_±_0_._0058|0_._0006_±_0_._0009|0_._0010_±_0_._0016|
|GCN|**0****_._0103****_±_ 0****_._0047**|**0****_._8815****_±_ 0****_._3117**|**0****_._0203****_±_ 0****_._0092**|
|GraphSAGE|0_._0057_±_0_._0062|0_._4919_±_0_._5188|0_._0112_±_0_._0123|
|GAT|0_._0054_±_0_._0087|0_._1798_±_0_._3688|0_._0095_±_0_._0156|
|GIN|0_._0044_±_0_._0080|0_._0102_±_0_._0131|0_._0016_±_0_._0015|



## **C Results Threshold-Dependent Metrics** 

In this part of the appendix, we give the precision, recall and F1-scores for the models using different thresholds. The thresholds are set such that the observations with the top 0 _._ 1%, 10% and _p_ % of scores are classified as money laundering, where _p_ % is equal to the relative occurrence of the labels. 

Table C3: Threshold-dependent metrics: Precision, recall and F1-score values for the top 10% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._3526_±_0_._0163|0_._6204_±_0_._0287|0_._4497_±_0_._0208|
|Egonet features|0_._3559_±_0_._0101|0_._6263_±_0_._0179|0_._4539_±_0_._0129|
|DeepWalk|0_._3517_±_0_._0050|0_._6189_±_0_._0089|0_._4485_±_0_._0064|
|Node2vec|0_._3536_±_0_._0040|0_._6222_±_0_._0071|0_._4509_±_0_._0051|
|GCN|0_._3471_±_0_._0272|0_._5988_±_0_._0358|0_._4393_±_0_._0306|
|GraphSAGE|**0****_._3675****_±_ 0****_._0118**|**0****_._6395****_±_ 0****_._0207**|**0****_._4666****_±_ 0****_._0124**|
|GAT|0_._3661_±_0_._0247|0_._6388_±_0_._0288|0_._4653_±_0_._0270|
|GIN|0_._3307_±_0_._0228|0_._5889_±_0_._0407|0_._4234_±_0_._0274|



44 

Network Analytics for Anti-Money Laundering – A Systematic Literature Review and Experimental Evaluation 

Table C4: Threshold-dependent metrics: Precision, recall and F1-score values for the top 10% scores over the different methods for the IBM-AML data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|**0****_._0148****_±_ 0****_._0006**|0_._1635_±_0_._0063|**0****_._0272****_±_ 0****_._0011**|
|Egonet features|**0****_._0148****_±_ 0****_._0002**|0_._1626_±_0_._0026|0_._0271_±_0_._0004|
|DeepWalk|0_._0145_±_0_._0003|0_._1602_±_0_._0033|0_._0267_±_0_._0005|
|Node2vec|**0****_._0148****_±_ 0****_._0006**|0_._1630_±_0_._0065|0_._0271_±_0_._0011|
|GCN|0_._0109_±_0_._0030|0_._9814_±_0_._0374|0_._0214_±_0_._0058|
|GraphSAGE|0_._0121_±_0_._0026|0_._9761_±_0_._0257|0_._0238_±_0_._0051|
|GAT|0_._0054_±_0_._0074|**0****_._1993****_±_ 0****_._3722**|0_._0104_±_0_._0144|
|GIN|0_._0081_±_0_._0077|0_._0947_±_0_._0793|0_._0149_±_0_._0141|



Table C5: Threshold-dependent metrics: Precision, recall and F1-score values for the top _p_ = 2% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._3244_±_0_._0143|0_._6303_±_0_._0277|0_._4283_±_0_._0188|
|Egonet features|0_._3271_±_0_._0098|0_._6357_±_0_._0190|0_._4319_±_0_._0129|
|DeepWalk|0_._3220_±_0_._0052|0_._6258_±_0_._0102|0_._4252_±_0_._0069|
|Node2vec|0_._3241_±_0_._0046|0_._6299_±_0_._0089|0_._4280_±_0_._0060|
|GCN|0_._3209_±_0_._0284|0_._6230_±_0_._0371|0_._4234_±_0_._0325|
|GraphSAGE|0_._3343_±_0_._0130|0_._6558_±_0_._0166|0_._4427_±_0_._0121|
|GAT|**0****_._3419****_±_ 0****_._0084**|**0****_._6613****_±_ 0****_._0212**|**0****_._4507****_±_ 0****_._0109**|
|GIN|0_._3034_±_0_._0169|0_._5949_±_0_._0285|0_._4018_±_0_._0205|



Table C6: Threshold-dependent metrics: Precision, recall and F1-score values for the top _p_ = 0 _._ 11% scores over the different methods for the Elliptic data set, based on the test set. The standard deviation is also reported. 

|**Methods**|**Precision**|**Recall**|**F1-score**|
|---|---|---|---|
|Intrinsic features|0_._0056_±_0_._0090|0_._0008_±_0_._0012|0_._0014_±_0_._0021|
|Egonet features|0_._0085_±_0_._0077|0_._0012_±_0_._0011|0_._0021_±_0_._0019|
|DeepWalk|0_._0080_±_0_._0076|0_._0013_±_0_._0013|0_._0023_±_0_._0021|
|Node2vec|0_._0047_±_0_._0063|0_._0008_±_0_._0010|0_._0013_±_0_._0018|
|GCN|**0****_._0103****_±_ 0****_._0046**|**0****_._8811****_±_ 0****_._3119**|**0****_._0203****_±_ 0****_._0089**|
|GraphSAGE|0_._0055_±_0_._0061|0_._4907_±_0_._5175|0_._0108_±_0_._0121|
|GAT|0_._0047_±_0_._0078|0_._1763_±_0_._3656|0_._0085_±_0_._0146|
|GIN|0_._0058_±_0_._0084|0_._0091_±_0_._0116|0_._0019_±_0_._0016|



45 

