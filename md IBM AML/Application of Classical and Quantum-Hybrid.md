# **FD4QC: Application of Classical and Quantum-Hybrid Machine Learning for Financial Fraud Detection** A Technical Report 

Matteo Cardaioli<sup>∗</sup> Luca Marangoni<sup>∗</sup> Giada Martini<sup>†</sup> Francesco Mazzolin<sup>†</sup> Luca Pajola<sup>†</sup> Andrea Ferretto Parodi<sup>∗</sup> Alessandra Saitta<sup>∗</sup> Maria Chiara Vernillo<sup>∗</sup> 

July 28, 2025 

##### **Abstract** 

The increasing complexity and volume of financial transactions pose significant challenges to traditional fraud detection systems. This technical report investigates and compares the efficacy of classical, quantum, and quantum-hybrid machine learning models for the binary classification of fraudulent financial activities. 

As of our methodology, first, we develop a comprehensive behavioural feature engineering framework to transform raw transactional data into a rich, descriptive feature set. Second, we implement and evaluate a range of models on the IBM Anti-Money Laundering (AML) dataset. The classical baseline models include _Logistic Regression_ , _Decision Tree_ , _Random Forest_ , and _XGBoost_ . These are compared against three hybrid classic quantum algorithms architectures: a _Quantum Support Vector Machine_ ( **QSVM** ), a _Variational Quantum Classifier_ ( **VQC** ), and a _Hybrid Quantum Neural Network_ ( **HQNN** ). 

Furthermore, we propose Fraud Detection for Quantum Computing ( **FD4QC** ), a practical, API-driven system architecture designed for real-world deployment, featuring a “classical-first, quantum-enhanced” philosophy with robust fallback mechanisms. 

Our results demonstrate that classical tree-based models, particularly _Random Forest_ , significantly outperform the quantum counterparts in the current setup, achieving high accuracy (97 _._ 34%) and F-measure (86 _._ 95%). Among the quantum models, **QSVM** shows the most promise, delivering high precision (77 _._ 15%) and a low false-positive rate (1 _._ 36%), albeit with lower recall and significant computational overhead. 

This report provides a benchmark for a real-world financial application, highlights the current limitations of quantum machine learning in this domain, and outlines promising directions for future research. 

> ∗GFT Technologies 

> †Spritzmatter 

1 

Technical Report 

## **1 Introduction** 

Financial institutions face significant financial and reputational risks from fraudulent activities, making them prime targets for advanced detection systems. In the European Economic Area (EEA), fraud losses across major payment instruments totaled e4.3 billion in 2022, with an additional e2.0 billion reported in the first half of 2023 alone [5]. For card payments, the fraud rate was 0.031% of the total transaction value during the first half of 2023, equivalent to 3.1 cents for every e100 transacted. The threat is amplified in cross-border transactions; fraud rates for card payments were ten times higher when the counterpart was located outside the EEA, where the application of Strong Customer Authentication (SCA) is not legally required [5]. Beyond direct financial costs, fraud inflicts substantial reputational damage and erodes customer trust. A recent industry analysis highlights that over 30% of fraud victims leave their financial institution, underscoring the critical importance of robust security [7]. These fraudulent activities are often enabled by sophisticated techniques such as social engineering and phishing, which lead to stolen card details and manipulated credit transfers. 

The inefficiencies of current detection systems, often marked by high false-positive rates, further highlight the need for technological innovation. Financial fraud is a pervasive and evolving threat, requiring the continuous development of sophisticated detection methodologies. While classical Machine Learning (ML) has been the cornerstone of fraud detection systems for years, the escalating complexity of fraudulent schemes and the sheer volume of data are pushing the boundaries of these approaches. Concurrently, the nascent field of Quantum Machine Learning (QML) offers intriguing possibilities, leveraging quantum phenomena like superposition and entanglement to unlock potentially new computational paradigms for complex pattern recognition. 

**Purpose and Scope.** This research provides a comparative analysis of classical and quantumhybrid ML models for the binary classification of financial transactions. We develop, implement, and evaluate three distinct quantum-inspired architectures: the _Quantum Support Vector Machine (QSVM)_ , the _Variational Quantum Classifier (VQC)_ , and a _Hybrid Quantum Neural Network (HQNN)_ . These are benchmarked against robust classical models. A core contribution of this work is an exploration of behavioural feature engineering, which is crucial for transforming raw data into meaningful inputs. Beyond model development, we address the practical deployment of these technologies through the design of the **FD4QC** service, an API conceptualised to integrate advanced fraud detection models into operational environments. This technical report synthesizes our findings, reflects on current limitations, and proposes future directions for for future research and experimental improvements. 

## **2 Methodology** 

Our experimental methodology includes data selection and behavioural feature engineering, the implementation of both classical and quantum models, and the design of a deployable system architecture. 

### **2.1 Dataset and Behavioural Feature Engineering** 

Our analysis is based on the synthetic IBM Transactions for Anti-Money Laundering (AML) dataset [1, 4], which simulates realistic transactional behaviour and suspicious activities realistic transactional behavior and suspicious activities within a network of customers and financial institu- 

2 

Technical Report 

tions.<sup>1</sup> . This synthetic dataset was generated by IBM to support research and development in the field of financial crime detection, particularly for Anti-Money Laundering tasks. 

To enhance model performance, we engineered a comprehensive set of behavioural features designed to capture the historical patterns of sender and receiver accounts.<sup>2</sup> These features, calculated based on past activity, can be grouped into several categories: 

- **Statistical Features by Account Role:** Mean, standard deviation, max, and min of transaction amounts for sender and receiver accounts, considering past sending and receiving activities separately. 

- **Temporal Dynamics:** Time elapsed since the last transaction for each account, including standard and _Exponentially Weighted Moving Averages (EWMA)_ to smooth patterns. 

- **Categorical and Contextual Features:** One-hot encoded features for payment currency and format, alongside contextual flags like `Same_Bank` or `is_self_loop` . 

- **Behavioural Change Features:** Indicators that detect deviations from an account’s typical transaction patterns in terms of currency or payment format. 

- **Pairwise Features:** Statistics capturing the specific transactional history between a senderreceiver pair, including a novel `Pair_Equilibrium` metric to measure the balance of the relationship. 

### **2.2 Classical Baseline Models** 

To provide a robust reference, we evaluated four widely used supervised learning algorithms: 

- **Logistic Regression (LR):** a linear model for binary classification. 

- **Decision Tree (DT):** a non-linear model based on hierarchical, axis-aligned splits. 

- **Random Forest (RF):** an ensemble method of decision trees to reduce variance. 

- **XGBoost (XGB):** an efficient, scalable implementation of gradient-boosted decision trees. 

### **2.3 Quantum and Hybrid Models** 

We explored three quantum-classical hybrid architectures that integrate quantum computing with classical optimization techniques. These algorithms have been implemented using _PennyLane_ and its integrations with _Scikit-learn_ [10] and _PyTorch_ [9]. 

1. **Quantum Support Vector Machine (QSVM).** It extends the classical SVM by using a quantum circuit to compute the kernel matrix. Data points are encoded into quantum states, and their inner product in the Hilbert space (an abstract vector space like Cartesian space but possibly infinite-dimensional, fundamental to quantum mechanics) defines the kernel, potentially revealing complex correlations intractable for classical kernels. 

2. **Variational Quantum Classifier (VQC).** A hybrid Quantum Machine Learning (QML) model consisting of three main components: 

> 1 `https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml` 

> 2We defined behavioral feature engineering following the idea proposed in [8]. 

3 

Technical Report 

   - _Classical Data to Quantum States Encoder._ A quantum feature map is used to encode classical input data into quantum states using a quantum circuit. This step is crucial to enable the quantum model to process classical data. 

   - _Parametrised Quantum Circuit (Ansatz)._ A quantum circuit that defines the model architecture and contains trainable quantum gate parameters, which determine how the quantum state evolves [6]. 

   - _Classical Optimization Algorithm._ Classical machine learning techniques, such as gradient descent, are used to iteratively update the parameters of the Ansatz. 

3. **Hybrid Quantum Neural Network (HQNN).** It integrates a variational quantum circuit as a hidden layer within a classical deep neural network. A classical encoder compresses the input features, which are then processed by the quantum layer. The results are fed back into a classical classifier for the final prediction. This architecture follows the paradigm outlined in [2, 3]. 

### **2.4 System Architecture: The FD4QC Service** 

To address practical deployment challenges, we designed **FD4QC** , a stateless RESTful API. Its “classical-first, quantum-enhanced” philosophy ensures operational robustness by using proven classical models as the backbone, while allowing for the controlled, gradual integration and A/B testing of experimental quantum models. 

The system includes a lightweight router for model selection and a transparent fallback mechanism: if a quantum backend is unavailable, the request is automatically rerouted to a classical surrogate, ensuring service continuity. The API response explicitly flags whether the prediction was generated by a “classical” or “quantum” engine, ensuring auditability. 

## **3 Experimental Setup and Results** 

All models were trained and evaluated on a reduced and undersampled version of the IBM dataset to facilitate experimentation, maintaining a class ratio of 9: 1 (non-suspicious to suspicious). The performance of the models was assessed on a holdout test set using **Accuracy** , **F-measure** , **Precision** , **Recall** , and the **False Positive Rate (FPR)** . 

Although real quantum hardware was not used in this study, the code implemented is compatible with quantum devices and can be executed on actual hardware via the _Pennylane_ [2] interface for quantum computing platforms. 

The comparative results are summarized in Table 1. 

### **3.1 Analysis of Results** 

#### • **Classical Baselines** 

Tree-based ensemble models ( _Random Forest_ and _XGBoost_ ) demonstrate clear superiority, achieving excellent balance across all metrics. _Random Forest_ shows the best overall performance, with high F-measure and a low FPR, which is critical for minimizing false alarms in an operational setting. 

#### • **Quantum-Inspired Models** 

- _VQC_ and _HQNN_ models performed poorly. Despite relatively high accuracy scores (attributable to the class imbalance), their F-measure and Recall were near-zero, indicating 

4 

Technical Report 

|**Algorithm**|**Accuracy**|**F-measure**|**Precision**|**Recall**|**FPR**|
|---|---|---|---|---|---|
||**Classic**<br>|**al Models**<br>||||
|Logistic Regression|0.8588|0.1241|0.1634|0.1000|0.0569|
|Decision Tree|0.9652|0.8374|0.7860|0.8960|0.0271|
|Random Forest|0.9734|0.8695|0.8536|0.8860|0.0169|
|XGBoost|0.9698|0.8558|0.8190|0.8960|0.0220|
||**Quantu**|**m Models**||||
|VQC – 1 layer – 4 qubits|0.5990|0.2128|0.1324|0.5420|0.3947|
|VQC – 2 layers – 4 qubits|0.5024|0.1566|0.0943|0.4620|0.4931|
|HQNN – 1L* – 4 qubits|0.9000|0.0000|0.0000|0.0000|0.0000|
|HQNN – 2L* – 4 qubits|0.8448|0.0827|0.1012|0.0700|0.0691|
|QSVC – 2 qubits|0.9272|0.5297|0.7482|0.4100|0.0153|
|QSVC – 4 qubits|0.9290|0.5372|0.7715|0.4120|0.0136|



Table 1: Performance metrics of classical and quantum-inspired models on the test set. L* indicates the number of quantum layers. 

a failure to identify the positive (fraudulent) class. Their behaviour suggests a bias towards the majority class. 

- _QSVC_ emerged as the most viable quantum model. The 4-qubit configuration achieved a respectable accuracy (92.9%) and F-measure (53.72%). Notably, it delivered high precision (77.15%) and a very low FPR (1.36%), comparable to the best classical models. However, its Recall (41.20%) is substantially lower than classical baselines, and its training/inference times were significantly longer, posing a practical challenge. 

## **4 Discussion** 

The experimental results clearly highlight that for this fraud detection task, well-established classical ensemble methods, powered by domain-specific feature engineering, remain the preferred choice for reliable and effective detection. The superior performance of _Random Forest_ and _XGBoost_ suggests that the rich, engineered features provide a sufficiently expressive data representation that these models can effectively exploit. 

The underperformance of the quantum models can be attributed to several factors, including algorithmic immaturity, challenges in training variational circuits, and the possibility that the dataset, even with complex features, does not possess the specific structure that would unlock a quantum advantage. 

Despite this, the performance of the _QSVC_ is noteworthy. Its ability to achieve high precision and a low FPR suggests a potential niche in settings where the cost of a false positive is extremely high, and a lower detection rate (recall) is, in turn, an acceptable trade-off. This finding warrants further investigation into quantum kernel methods. 

Several promising research directions could improve the viability of QML for this application: 

- **Advanced Circuit Design:** Exploring tailored ansätze and adaptive feature maps to enhance model expressiveness without incurring training instabilities. 

- **Hybrid Architectures:** Combining temporal feature extraction (e.g., with LSTMs) with variational quantum circuits to better leverage the strengths of both paradigms [12]. 

5 

Technical Report 

- **Data Encoding & Feature Selection** : Utilizing techniques like quantum autoencoders [11] for feature compression or metaheuristic methods for feature selection to prepare highdimensional data for quantum processing. 

## **5 Conclusion** 

This study provides a benchmark of classical, quantum-hybrid, and quantum machine learning models for financial fraud detection. Our findings indicate that, at present, quantum AI is not mature enough to outperform traditional algorithms in this practical, high-stakes domain. Classical ensemble methods demonstrate robust, superior performance, making them the preferred choice for operational deployment. 

However, our exploration has yielded valuable insights. The _QSVC_ model showed notable potential in achieving high precision, and the conceptual **FD4QC** architecture offers a pragmatic roadmap for integrating future quantum capabilities into financial security systems. Continued investigation into quantum kernel methods, advanced hybrid architectures, and sophisticated data encoding strategies is essential for unlocking the future potential of quantum computing in the financial sector. 

## **6 Acknowledgments** 

This work was supported by the project FD4CQ – Fraud Detection con Computer Quantistici, funded under the Italian PNRR initiative, Mission 4 “Education and Research” – Component 2 “From Research to Business”, Investment Line 1.4, and financed by the European Union – NextGenerationEU. The project is part of ICSC – Spoke 10 "Quantum Computing". 

## **References** 

- [1] Erik Altman, Jovan Blanuša, Luc Von Niederhäusern, Béni Egressy, Andreea Anghel, and Kubilay Atasu. Realistic synthetic financial transactions for anti-money laundering models. _Advances in Neural Information Processing Systems_ , 36:29851–29874, 2023. 

- [2] Ville Bergholm, Maria Schuld, Christian Gogolin, Josh Izaac, and Nathan Killoran. Pennylane: Automatic differentiation of hybrid quantum-classical computations. _arXiv preprint arXiv:1811.04968_ , 2022. 

- [3] Lukas Bischof, Stefan Teodoropol, Rudolf M Füchslin, and Kurt Stockinger. Hybrid quantum neural networks show strongly reduced need for free parameters in entity matching. _Scientific Reports_ , 15(1):4318, 2025. 

- [4] Béni Egressy, Luc Von Niederhäusern, Jovan Blanuša, Erik Altman, Roger Wattenhofer, and Kubilay Atasu. Provably powerful graph neural networks for directed multigraphs. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 38, pages 11838–11846, 2024. 

- [5] European Central Bank and European Banking Authority. Ecb and eba publish joint report on payment fraud, August 2024. 

- [6] Tobias Haug, Kishor Bharti, and MS Kim. Capacity and quantum geometry of parametrized quantum circuits. _PRX Quantum_ , 2(4):040309, 2021. 

6 

Technical Report 

- [7] Martins, Cleber. What percentage of customers leave their bank after an incident of app fraud? _Fintech Finance News_ , April 2025. 

- [8] Luca Pajola, Dongkai Chen, Mauro Conti, and VS Subrahmanian. A novel review helpfulness measure based on the user-review-item paradigm. _ACM Transactions on the Web_ , 17(4):1–31, 2023. 

- [9] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc, E. Fox, and R. Garnett, editors, _Advances in Neural Information Processing Systems 32 (NeurIPS 2019)_ , pages 8026–8037. Curran Associates, Inc., 2019. 

- [10] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikitlearn: Machine learning in python. _Journal of machine learning research_ , 12(Oct):2825–2830, 2011. 

- [11] Jonathan Romero, Jonathan P Olson, and Alan Aspuru-Guzik. Quantum autoencoders for efficient compression of quantum data. _Quantum Science and Technology_ , 2(4):045001, 2017. 

- [12] Yuto Takaki, Kosuke Mitarai, Makoto Negoro, Keisuke Fujii, and Masahiro Kitagawa. Learning temporal data with a variational quantum recurrent neural network. _Physical Review A_ , 103(5):052414, 2021. 

7 

