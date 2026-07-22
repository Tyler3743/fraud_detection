# **Graph Neural Networks Applied to Money Laundering Detection in Intelligent Information Systems** 

Ítalo Della Garza Silva Universidade Federal de Lavras Lavras, Minas Gerais, Brazil italo.silva7@estudante.ufla.br 

Luiz Henrique Andrade Correia Erick Galani Maziero Universidade Federal de Lavras Universidade Federal de Lavras Lavras, Minas Gerais, Brazil Lavras, Minas Gerais, Brazil lcorreia@ufla.br egmaziero@gmail.com 

## **ABSTRACT** 

### **ACM Reference Format:** 

Ítalo Della Garza Silva, Luiz Henrique Andrade Correia, and Erick Galani Maziero. 2023. Graph Neural Networks Applied to Money Laundering Detection in Intelligent Information Systems. In _XIX Brazilian Symposium on Information Systems (SBSI ’23), May 29–June 01, 2023, Maceió, Brazil._ ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3592813.3592912 

**Context** : Financial crimes exist in all world countries, and one of the most recurrent ones is Money Laundering (MoL). This crime can harm the country’s economy, increase criminality, and compromise social investments. Moreover, it can increase the investment risk factor, raising exchange and interest rates and causing high inflation. In recent years, financial institutions and government agencies have searched for solutions to detect MoL in financial transactions. **Problem** : Several institutions have employed naive IS for detect MoL. Most systems are based on rules and label a large transactions number as suspicious, which makes the decision process inaccurate and inefficient. **Solution** : The recent literature presents Graph Neural Networks (GNN) as a promising solution to illegal transaction detection. We applied the Node and Edge Neural Network (NENN) architecture to classification, using the attributes of both bank accounts (vertices) and transactions (edges). **IS Theory** : In the Intelligent Information Systems context, Machine Learning is a way to improve the decision-making ability of programs in IS. **Method** : The GCN, Skip-GCN, and NENN architectures were evaluated for the MoL detection problem, comparing two ways of representing transactions as graphs (transactions as vertices or edges). Also, was considered the performance of XGBoost and Softmax classifiers in the solution. **Summary of Results** : Results showed better performance when transactions represented the nodes. In addition, NENN+XGBoost was superior for higher class imbalance values, with an F1-score of 74,51±4,21% to "illicit" transactions. **Contributions and impacts in the IS area** : This paper improves the decision-making ability of Anti-Money Laundering systems, helping the organization and efficiency of public and private institutions, and contributing to the fight against corruption. This theme is aligned with the GrandDSI-BR2016-2026. 

## **1 INTRODUÇÃO** 

Os crimes financeiros são um problema para todos os países, e cabe a determinadas agências federais e instituições financeiras identificar tais crimes, sendo que esse trabalho é feito através de uma série de processos investigativos bem-estabelecidos. Define-se "crime financeiro" por aquele praticado por uma entidade através do mercado financeiro, buscando ganho próprio em detrimento de outras entidades [7]. Dentre as principais atividades que se enquadram na categoria, vale destacar a Fraude Financeira, Sonegação de Impostos, Falsificação de Moedas, Manipulação de Mercado, e Lavagem de Dinheiro [12]. A Lavagem de Dinheiro pode ser definida como a ação de esconder dados que comprovem a ilegalidade do patrimônio obtido através de crimes, de forma a prevenir a aplicação dos devidos processos legais e a possibilidade de uso dos dados como prova do crime. A 11ª edição do Relatório Global de Fraude e Risco da Kroll mostra que a Lavagem de Dinheiro é um dos incidentes que mais afetaram as organizações em 2019 [17]. 

Dentre os Grandes Desafios de Pesquisa em Sistemas de Informação para o período 2016–2026, se encontra o uso de Sistemas de Informação na promoção da transparência, que é motivado também pela luta contra a corrupção [3], alinhado ao tema deste trabalho. Além disso, outro grande desafio é a integração e o aprimoramento de ecossistemas digitais – tanto de entidades privadas quanto governamentais – incluindo tecnologias emergentes e melhorando a eficiência e a organização no seu gerenciamento de processos de negócios [21], que também é objetivo deste trabalho. 

## **CCS CONCEPTS** 

Atualmente, tanto instituições financeiras quanto agências governamentais usam Inteligência Artificial em sistemas de detecção de Lavagem de Dinheiro. Considerando o contexto de Sistemas de Informação, pode-se afirmar que tais sistemas de detecção compõem os Sistemas de Informação Inteligentes (SSI) dessas organizações, uma vez que integram conjuntos de dados e Inteligência Artificial para auxiliar no cumprimento de uma tarefa de decisão. No entanto, a maior parte desses sistemas tende a ser simplista e baseado em regras [15], geralmente gerando um grande número de transações suspeitas e tornando a tarefa de detecção ineficiente. 

• **Social and professional topics** → **Financial crime** ; • **Computing methodologies** → **Neural networks** . 

## **KEYWORDS** 

anti-money laundering, graph neural networks, deep learning. 

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. _SBSI ’23, May 29–June 01, 2023, Maceió, Brazil_ 

A literatura tem apresentado várias alternativas aos sistemas baseados em regras, como sistemas baseados em Redes Neurais de Grafos ( _Graph Neural Networks_ – GNN). A principal motivação para aplicar GNNs nesse contexto é a possibilidade de se associar atributos de transações ou contas financeiras vizinhas, i.e., que estejam 

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0759-9/23/05...$15.00 https://doi.org/10.1145/3592813.3592912 

252 

Ítalo Della Garza Silva, Luiz Henrique Andrade Correia, and Erick Galani Maziero 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

ligadas direta ou indiretamente por meio de uma transação ou conta em comum, e com isso minerar informações sobre o contexto das transações a fim de tornar mais acurada a detecção das transações ilegais. 

Existem duas maneiras de se representar dados de transações financeiras através de grafos. Uma opção é fazer com que os vértices do grafo sejam as transações financeiras. Essa forma é bastante utilizada quando o conjunto de dados vêm de um sistema de criptomoedas, como o Bitcoin Blockchain [25], e, nesse caso, as arestas do grafo representam o fluxo monetário. Para representar dados de transações bancárias comuns, é possível ligar as transações (vértices) entre si quando ambos tiverem a mesma origem ou destino [22]. A outra opção para representar o conjunto de dados é fazer com que os vértices sejam as contas bancárias e as arestas sejam as transações. Dessa maneira, é possível utilizar tanto os atributos das próprias contas financeiras quanto os atributos das transações e definir como a arquitetura de GNN irá incorporar os dados das arestas e vértices vizinhos separadamente. No entanto, nesse cenário, é necessário utilizar uma arquitetura de GNN que seja capaz de gerar _embeddings_ para as arestas. 

Um dos grandes desafios envolvendo GNNs aplicadas à detecção de Lavagem de Dinheiro é o desbalanceamento de classe [22], ou seja, o número de instâncias de uma classe é muito superior ao da outra classe (quando se trata de classificação binária). Isso ocorre porque, em ambientes de transação financeira, existem muito mais transações lícitas do que ilícitas. Isso afeta negativamente o treinamento do modelo e consequentemente diminui sua qualidade em uma aplicação no mundo real. A taxa de desbalanceamento de classe em um conjunto de dados para classificação binária pode ser definida como a razão entre o número total de instâncias e o número de instâncias da classe minoritária. 

Este trabalho avalia a performance de modelos de GNN para o problema de detecção automática de Lavagem de Dinheiro em um conjunto de dados de transação financeira, classificando cada transação como "lícita" ou "ilícita", e comparando os resultados em ambas as maneiras de se representar o conjunto de dados em grafos (os vértices como transações financeiras e os vértices como contas financeiras), em diferentes níveis de desbalanceamento de classe. 

Este artigo está organizado como descrito a seguir. A Seção 2 apresenta a base teórica necessária para a compreensão dos experimentos executados. Os trabalhos relacionados ao tema do artigo são apresentados na Seção 3. A Seção 4 descreve a metodologia adotada neste estudo. As Seções 5 e 6 listam e discutem, respectivamente, os resultados obtidos e seu impacto no contexto de Sistemas de Informação. Por fim, a Seção 7 conclui este trabalho e lista possíveis trabalhos futuros. 

## **2 REFERENCIAL TEÓRICO** 

Esta seção apresenta o conceito de "Anti-Lavagem de Dinheiro" e a evolução da Inteligência Artificial aplicada a esta área. Também explana sobre as Redes Neurais de Grafos e cada variação desse subtipo de rede neural utilizada neste trabalho. 

## **2.1 Anti-Lavagem de Dinheiro (** **_Anti-Money Laundering_ )** 

Os sistemas Anti-Lavagem de Dinheiro ( _Anti-Money Laudering_ – AML) são implementados por vários tipos de instituições financeiras 

e agências governamentais para prevenir atividades de Lavagem de Dinheiro em seu sistema financeiro [15]. Um sistema AML de uma instituição precisa ser suficientemente eficaz para prevenir que a existência de transações aprovadas em seu sistema sejam rotuladas como "ilegais" futuramente pelo sistema de outra instituição. Quando isso ocorre, o sistema de segurança da instituição pode estar comprometido, consequentemente afetando sua reputação e valor de mercado. Para lidar com isso, a maioria dos sistemas de segurança no mundo seguem as recomendações da _Financial Action Task Force_ (FATF), chamadas de "Quarenta Recomendações da FATF" ( _FATF 40 Recommendations_ ) [10], cujo desenvolvimento foi iniciado em 1990. Os sistemas de segurança que seguem as regras da FATF geram uma série de relatórios de atividade suspeita ( _Suspicious Activity Reports_ – SAR), os quais contêm informações sobre as transações, contas envolvidas e o tipo de atividade suspeita. 

Esses sistemas baseados em regras normalmente geram um grande número de atividades suspeitas, sendo que a análise humana posterior é frequentemente necessária para filtrar as transações com um potencial real de serem suspeitas. Logo, a detecção leva um tempo considerável para ser executada, sendo necessário um gasto financeiro com os especialistas que realizam a análise posterior. Para solucionar esses problemas, a literatura apresenta várias soluções propondo técnicas alternativas de Inteligência Artificial para realizar a detecção de AML. Uma das abordagens é a Análise de Redes [8, 11], que pode ser aplicada a dados financeiros transacionais de forma a se obter links ocultos ou diretos de um nó no qual já foi detectada a Lavagem de Dinheiro. 

Outra alternativa é a Análise de Links [20], que trata os dados financeiros como um grafo conexo e avalia as relações entre os nós desse grafo. A Detecção de _Outliers_ [18, 19] busca detectar transações com um comportamento (i.e, seu conjunto de atributos) consideravelmente diferente do restante das transações (transações "anormais"). Muitos trabalhos também fazem uso de métodos de _Machine Learning_ para classificação/ _scoring_ de risco, usando o resultado como base para detectar as transações suspeitas [5]. Por fim, há também o uso de Aprendizado de Grafos aplicado à detecção de Lavagem de Dinheiro [1], que compreende majoritariamente Redes Neurais de Grafos e é o foco deste trabalho. 

A principal motivação da aplicação de Redes Neurais de Grafos na detecção de atividades suspeitas, é a possibilidade de se representar um conjunto de transações através de diferentes tipos de grafos [24], fazendo com que a detecção, ao avaliar uma transação, considere o contexto no qual ela está inserida, i.e, sua vizinhaça e suas conexões na rede de transações. 

## **2.2 Redes Neurais de Grafos** 

De acordo com [14], uma Rede Neural de Grafo ( _Graph Neural Network_ - GNN) é uma rede neural usada para lidar com dados estruturados em grafo. Nessa maneira de se representar os dados, cada vértice ou aresta (ou ambos) contêm um vetor de atributos. Uma GNN recebe e processa esse grafo de dados, i.e, além do conjunto de atributos como entrada, que é comum a todas as redes neurais, ela receberá informações sobre como cada dado se liga a outros dados do conjunto. Essa informação pode ser representada por uma matriz de adjacência. A cada passo, a rede combina a informação de cada instância com a informação de seu respectivo vizinho no grafo, gerando um vetor _embedding_ . Essa operação é chamada de 

253 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

Graph Neural Networks Applied to Money Laundering Detection in Intelligent Information Systems 

" _message passing_ ". É possível representar um passo do " _message passing_ " de uma GNN pela Equação 1. 



Dessa forma, os _embeddings_ de todos os vértices _𝑣_ , na iteração _𝑘_ ( _ℎ𝑣_<sup>(</sup><sup>_𝑘_)</sup> ), sendo que _𝑣_ é vizinho do vértice _𝑢_ , são agregados por uma função AGREGA qualquer. O resultado é combinado com o _embedding_ de _𝑢_ na mesma iteração ( _ℎ𝑢_<sup>(</sup><sup>_𝑘_)</sup> ), e então atualizados por uma função ATUALIZA qualquer, gerando o novo vetor _embedding_ de _𝑢_ para a iteração _𝑘_ + 1. No primeiro passo, os vetores _embeddings_ são os próprios vetores do conjunto de dados original. O grafo computacional da Figura 1 ilustra um exemplo do processo de agregação. 



<!-- Start of picture text -->
A<br>C<br>B B<br>A A<br>C A AGREGA C B<br>F<br>D D E<br>E<br>F<br>A<br><!-- End of picture text -->

**Figure 1: Grafo computacional para dois passos do** **_message passing_ . Adaptado de [14].** 

O processo definido acima é a base de uma GNN. As arquiteturas específicas de GNN utilizadas neste trabalho são descritas a seguir. 

_2.2.1 Rede Convolucional de Grafo._ Proposta por [16], essa variação é uma das implementações mais comuns e simples de GNN, que realiza operações de convolução baseadas nas conexões entre os diferentes vértices do grafo. A Rede Convolucional de Grafo ( _Graph Convolutional Network_ - GCN) agrega os vetores _embedding_ dos vértices _𝑢_ e _𝑣_ normalizando cada _embedding_ através do produto entre os graus de _𝑢_ e _𝑣_ (|N ( _𝑢_ )| e |N ( _𝑣_ )|), respectivamente. O resultado é multiplicado pelo conjunto de pesos _𝑊_<sup>(</sup><sup>_𝑘_)</sup> inserido da função de ativação _𝜎_ (pode ser usada uma _Rectified Linear Unit_ – ReLU – ou uma função sigmóide, por exemplo). Esse processo pode ser representado pela Equação 2. Para realizar a classificação, os _embeddings_ da última camada são inseridos em uma camada linear seguida de uma camada _Softmax_ . 



_2.2.2 Skip-GCN._ Uma variação " _skip_ " da GCN comum foi apresentada por [25], e consiste na adição de uma conexão " _skip_ " para amenizar o problema da degradação progressiva causada pela profundidade da rede neural. Essa conexão alimenta as camadas profundas da rede neural com os valores de entrada. Esse processo é mostrado na Equação 3 



_2.2.3 Rede Neural de Nós e Arestas._ Introduzida por [26], a Rede Neural de Grafos com Nós e Arestas ( _Node and Edge Neural Networks_ - NENN) utiliza em suas operações tanto atributos de nós quanto de arestas, gerando _hidden embeddings_ para ambos. Para compreender a NENN, é necessário definir os seguintes conceitos sobre vizinhança em grafos: 

- **Vizinhos Baseados em Nó** : Para um grafo _𝐺_ = ( _𝑉, 𝐸_ ), os vizinhos baseados em nó N _𝑖_ de um nó _𝑖_ são os nós ligados diretamente a este por uma aresta, e o próprio nó _𝑖_ . Para uma aresta _𝑗_ desse mesmo grafo, seus vizinhos baseados em nó N _𝑗_ são os nós conectados por essa aresta. 

- **Vizinhos Baseados em Aresta** : Para um grafo _𝐺_ = ( _𝑉, 𝐸_ ), os vizinhos baseados em aresta _𝜖𝑖_ de um nó _𝑖_ são as arestas que se conectam a esse nó. Para uma aresta _𝑗_ desse mesmo grafo, os seus vizinhos baseados em aresta são as arestas que se conectam aos nós conectados por esta aresta, e a própria aresta _𝑗_ . 

A NENN possui dois tipos de camadas. As Camadas de Atenção no Nível de Nó ( _Node-Level Attention Layer_ ) geram _embeddings_ para os nós através de seus vizinhos baseados em nó e em aresta. Já as Camadas de Atenção no Nível de Aresta ( _Edge-Level Attention Layer_ ) geram os _embeddings_ para as arestas. 

A Camada de Atenção no Nível de Nó calcula um coeficiente de importância _𝛼𝑖𝑗_<sup>_𝑛_de cada nó</sup><sup>_𝑗_para cada nó</sup><sup>_𝑖_. Esse coeficiente,</sup> para um dado nó _𝑖_ é obtido multiplicando-se os conjuntos de atributos _𝑥𝑖_ do nó _𝑖_ e _𝑥 𝑗_ do nó _𝑗_ pela matriz de pesos _𝑊𝑛_ ∈ R<sup>_𝑑_</sup> _𝑣_<sup>_𝑙_+</sup><sup>_𝑑_</sup> _𝑣_<sup>_𝑙_+1</sup> , sendo _𝑑_<sup>_𝑙_</sup> _𝑣_ o tamanho do _embedding_ de um nó na camada _𝑙_ . Os resultados dessas multiplicações sofrem uma concatenação (nesta seção representados pelo símbolo ||) e multiplicado pelo vetor de parâmetros _𝑎_<sup>_𝑇_</sup> _𝑛_ ∈ R<sup>2</sup><sup>_𝑑_</sup> _𝑣_<sup>_𝑙_+1</sup> de uma Rede Alimentada Adiante de camada única. Em seguida, o resultado passa pela função de ativação _𝜎_ (p. ex. LeakyReLU), e é normalizado via _Softmax_ , conforme mostra a Equação 4. 



Outro coeficiente de importância calculado por esta camada é o _𝛼_<sup>_𝑒_</sup> _𝑖𝑘_<sup>, que quantifica a importância de cada aresta</sup><sup>_𝑘_para cada nó</sup> _𝑖_ . Multiplica-se o conjunto de atributos _𝑒𝑘_ da aresta _𝑘_ pela matriz de pesos _𝑊𝑒_ ∈ R<sup>_𝑑_</sup> _𝑒_<sup>_𝑙_+</sup><sup>_𝑑_</sup> _𝑒_<sup>_𝑙_+1</sup> (sendo _𝑑𝑒_<sup>_𝑙_</sup> o tamanho do _𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔_ da aresta _𝑒_ na camada _𝑙_ ). As multiplicações são então concatenadas e multiplicadas pelo vetor de parâmetros _𝑎_<sup>_𝑇_</sup> _𝑒_ ∈ R<sup>_𝑑_</sup> _𝑣_<sup>_𝑙_+1</sup> + _𝑑𝑒_<sup>_𝑙_+1</sup> de uma Rede Alimentada Adiante de camada única. Por fim, o resultado é ativado pela função _𝜎_ e normalizado via _Softmax_ , como ilustrado na Equação 5. 



É calculada então uma média dos valores de entrada (já multiplicados pelas matrizes de peso da camada) ponderadas pelas importâncias obtidas, gerando os conjunto de valores _𝑥_ N _𝑖_ e _𝑥𝜖𝑖_ , conforme mostram as Equações 6 e 7. 

254 

Ítalo Della Garza Silva, Luiz Henrique Andrade Correia, and Erick Galani Maziero 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 





Esses conjuntos de valores são então concatenados para formar o _embedding_ de saída da camada, conforme mostra a Equação 8. 



A Camada de Atenção no Nível de Aresta trabalha de maneira similar à Camada de Atenção no Nível de Nó, porém calculando as atenções e _embeddings_ dos vizinhos de cada aresta. Dada uma aresta _𝑖_ (com um conjunto de atributos _𝑒𝑖_ ) qualquer do grafo e uma aresta _𝑘_ vizinha de _𝑖_ (com um conjunto de atributos _𝑒𝑘_ ), multiplicam-se seus conjuntos de atributos pela matriz de pesos _𝑊𝑒_ e concatenamse os resultados. Esse valor é então multiplicado pelo vetor de parâmetros _𝑞_<sup>_𝑇_</sup> _𝑒_ ∈ R<sup>2</sup><sup>_𝑑_</sup> _𝑒_<sup>_𝑙_+1</sup> , ativado pela função _𝜎_ e normalizado pela função _Softmax_ para se obter a importância _𝛽𝑖𝑘_<sup>_𝑒_,comomostraa</sup> Equação 9 



Da mesma forma, calcula-se a importância _𝛽𝑖𝑗_<sup>_𝑛_de cada nó</sup><sup>_𝑗_(com</sup> um conjunto de atributos _𝑥 𝑗_ ) para cada aresta _𝑖_ (com um conjunto de atributos _𝑒𝑖_ ). O conjunto de atributos _𝑥 𝑗_ é multiplicado pela matriz de pesos _𝑊𝑛_ e o conjunto de atributos _𝑒𝑖_ é multiplicado pela matriz de pesos _𝑊𝑒_ . Os dois resultados são concatenados e multiplicados pelo vetor de parâmetros _𝑞_<sup>_𝑇_</sup> _𝑛_ ∈ R<sup>_𝑑_</sup> _𝑣_<sup>_𝑙_+1</sup> + _𝑑𝑒_<sup>_𝑙_+1</sup> . Ativa-se o resultado com a função _𝜎_ e normaliza-se via _Softmax_ , como ilustra a Equação 10. 



Ocorre então o cálculo das médias dos valores de entrada, já multiplicados pelas matrizes de peso, ponderadas pelas importâncias obtidas. Em seguida, ocorre a concatenação desses valores para gerar o _embedding_ , como mostram as Equações 11 e 12. 



Os _embeddings_ calculados em uma camada são usados como atributos pela camada seguinte, sejam eles associados aos nós ou às arestas. A Figura 2 ilustra um modelo simples baseado na arquitetura NENN. 

## **3 TRABALHOS RELACIONADOS** 

Uma base de dados de grafo foi desenvolvida por [25] para detectar crimes financeiros baseados em dados do mundo real pela _Bitcoin Blockchain_ . Para realizar a classificação nesses dados, foi utilizada uma GCN simples de duas camadas. Os vetores _embedding_ gerados pela rede foram também posteriormente incluídos no conjunto de atributos, visando testar a influência da inclusão de contexto na informação em outros métodos de classificação, nomeadamente Regressão Logística, _Random Forest_ e _Multi-Layer Perceptron_ . A influência da temporalidade nos dados foi também testada usando uma abordagem recorrente de GCN. O melhor resultado foi obtido 

pela _Random Forest_ via vetores _embedding_ . Nenhum dos métodos propostos foi capaz de lidar com o fechamento do Mercado Negro que ocorreu em um dos _timesteps_ do banco de dados. 

Diversas técnicas de Mineração de Dados foram testadas no contexto de Lavagem de Dinheiro por [6]. Algoritmos de busca em grafos (como Dijkstra e _Breadth-First Search_ ), mineração de padrões frequentes (como _Eclat_ e _FP-Close_ ) e correspondência em grafos (VF2) foram aplicadas em três conjuntos de dados reais de transações bancárias, cada conjunto relacionado a uma investigação diferente realizada por uma agência governamental. O estudo apresentou resultados satisfatórios usando-se técnicas baseadas em grafo para o conjunto de dados utilizado. 

Um conjunto de dados de contas ilegais foi gerado por [9] para transações financeiras da _Ethereum Blockchain_ e uma lista de contas bancárias que realmente praticaram atividades ilícitas naquele cenário. Foi utilizado XGBoost para detectar tais contas ilegais, obtendo altas acurácia e área sob curva ROC (AUC). O método proposto, no entanto, é incapaz de detectar a lógica computacional dos _smart contracts_ , realizados para ocultar as atividades ilícitas. Além disso, [9] abordaram a detecção de contas financeiras, enquanto esse trabalho aborda a detecção das transações em si. 

Uma nova abordagem de GCN foi usada por [2] para lidar com a base de dados desenvolvida por [25]. A GCN padrão foi modificada para lidar com grafos direcionados, através do Laplaciano normalizado simétrico. Foi obtida alta acurácia, porém baixa revocação para o conjunto de transações "ilícitas". A abordagem também não leva em consideração a natureza temporal dos dados no conjunto de dados. Os autores também sugeriram que a inclusão da data e hora real de cada transação pode ser muito mais significativa para o aprendizado que os _timesteps_ colocados no conjunto de dados. 

Uma nova abordagem que utiliza _multi-task learning_ para a geração de _embeddings_ foi proposta por [4]. O sistema, conhecido como DELATOR, utiliza o modelo GraphSAGE para a obtenção dos _embeddings_ através de ambos aprendizado não supervisionado e regressão, que são combinados para se gerar os _embeddings_ de saída. Para a classificação, foram testados tanto um classificador único quanto dois classificadores combinados (o segundo classificador "filtra" os casos considerados suspeitos pelo primeiro, buscando uma maior precisão). O delator foi testado em uma base de dados real, com transações do banco Inter, sendo que os resultados obtidos foram maiores do que os _baselines_ comparados. 



<!-- Start of picture text -->
Grafo de Camada de Atenção Camada de Atenção Grafo de<br>Entrada no Nível de Nó no Nível de Aresta Saída<br>𝑒 3 𝑥 4 𝑒 4 𝑒 3 𝑥 4 𝑒 4 𝑒 3 𝑥 4 𝑒 4 𝑒 3 𝑥 4 𝑒 4<br>𝑥𝑥 32 𝑒𝑒 12 𝑥𝑒 17 𝑒𝑒 56 𝑥𝑥 65 𝑥𝑥 32 𝑒𝑒 12 𝑥𝑒 17 𝑒𝑒 56 𝑥𝑥 65 𝑥𝑥 32 𝑒𝑒 12 𝑥𝑒 17 𝑒𝑒 56 𝑥𝑥 65 camadasmais 𝑥𝑥 32 𝑒𝑒 12 𝑥𝑒 17 𝑒𝑒 56 𝑥𝑥 65<br>𝑥 7 𝑒𝑒 12 𝑥 7 𝑥 7 Ativação 𝑥 7<br>[ 𝑦𝑖 ]<br> 𝑥𝑥𝑥𝑑𝑥... 12   𝑒𝑑𝑒...  𝑥𝑥 32 𝛼𝑥𝛼 12 𝑛 𝑥 5 𝛼 13 𝑛 717 𝑛 𝛼𝛼 15 𝑛 𝑥𝑒 17 𝑒 12 𝛼𝑒 𝛼 12 𝑒 716 𝑒 𝛼𝛼 11 𝑒 15 𝑒 𝑒𝑒 61 𝑒 5 𝑒 1 𝑒𝛽 2 𝑒𝛽 61 𝑒 767 𝑒 𝛽 62 𝑒 𝑒 6 𝛽𝛽 66 𝑒 𝑥 61 𝑒 𝑥 16<br><!-- End of picture text -->

**Figure 2: Arquitetura NENN. Adaptado de [26].** 

255 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

Graph Neural Networks Applied to Money Laundering Detection in Intelligent Information Systems 

A maioria dos trabalhos citados utilizam bases de dados de transações feitas em criptomoedas, enquanto o foco deste trabalho são as transações bancárias. Os trabalhos que utilizam transações bancárias não divulgam o seu conjunto de dados, uma vez que a natureza de tais dados é sigilosa. Este trabalho utiliza dados gerados pelo simulador AMLSim, utilizado por [22]. No entanto, diferentemente dos trabalhos citados, este trabalho avalia a arquitetura NENN, que incorpora atributos de nós e arestas, no contexto da detecção de Lavagem de Dinheiro. 

## **4 EXPERIMENTOS REALIZADOS** 

Para os experimentos realizados neste trabalho, foram utilizadas as arquiteturas GCN, Skip-GCN e NENN com uma camada totalmente conectada e um _Softmax_ para fazer a classificação. Além disso, o algoritmo XGBoost foi treinado e testado com os _embeddings_ gerados por essas três arquiteturas. Tanto a GCN quanto a SkipGCN utilizadas possuem duas camadas e 16 neurônios nas camadas escondidas. a NENN foi construída com duas camadas de atenção de arestas (no início e no fim do conjunto de camadas escondidas) e uma camada de atenção de nós (no centro do conjunto de camadas escondidas). 

Para as três arquiteturas o otimizador utilizado foi o _Adam_ , sendo que a taxa de aprendizagem para as arquiteturas baseadas em GCN foi fixada em 1 _,_ 0 × 10<sup>−3</sup> , enquanto para a arquitetura NENN foi estabelecida em 1 _,_ 0 × 10<sup>−4</sup> . Foi utilizada a Entropia Cruzada com Pesos como função de erro. Para as arquiteturas GCN e Skip-GCN, os pesos foram fixados em 0,7 para a classe "ilícito" e 0,3 para a classe "lícito". A arquitetura NENN se mostrou consideravelmente sensível à variação da taxa de desbalanceamento e, por isso, foram fixados pesos diferentes para cada conjunto de dados testado. Para o AMLSim 1/3 não foram estabelecidos pesos. Para o AMLSim 1/5 foi fixado 0,6 para a classe "ilícito" e 0,4 para a classe "lícito". Para o AMLSim 1/10 foi fixado 0,85 para a classe "ilícito" e 0,15 para a classe "lícito". Por fim, para o AMLSim 1/20 foi fixado 0,96 para a classe "ilícito" e 0,04 para a classe "lícito". Foram testados vários pesos diferentes para cada conjunto de dados até que fosse encontrado um ponto de equilíbrio entre a precisão e a revocação do modelo. Cada modelo foi testado em cada conjunto de dados 100 vezes e os intervalos das métricas foram coletadas com 95% de confiança. 

Os testes foram implementados em Python 3.8, sendo que as bibliotecas utilizadas para implementação dos modelos foram Pytorch 1.10 e Pytorch-Geometric 2.0.2, o XGBoost foi implementado com a biblioteca XGBoost 1.6.1, e as métricas foram coletadas via Scipy 1.8 e Scikit-learn 1.0.2. Os códigos utilizados encontram-se disponíveis _on-line_ para fins de estudo<sup>1</sup> 

## **4.1 Conjuntos de Dados** 

Os dados utilizados neste trabalho foram gerados por um simulador de transações financeiras baseado em agentes, denominado AMLSim [23]. O AMLSim foi baseado no PaySim, porém contendo instruções específicas para a geração de transações com um comportamento típico de um esquema de Lavagem de Dinheiro. Através desse simulador, é possível configurar a quantidade total de _timesteps_ (dias) da simulação, a data de início, o valor mínimo e máximo das transações, entre outros parâmetros estatísticos. Para este estudo, foram geradas transações diárias para o ano de 2020 (365 

1Disponível em: https://github.com/italodellagarza/SBSITests Acesso: 09/12/2022 

_timesteps_ ). Durante a simulação, são geradas informações sobre o identificador da conta, se é uma conta suspeita a priori, suas datas de abertura e fechamento, o depósito inicial, o identificador do padrão de comportamento, o identificador do banco, entre outros. As informações referentes à transação são o seu identificador, as contas de origem e destino, o tipo de transação, o dia, e se a transação é ou não suspeita. Para este trabalho, as informações utilizadas foram somente o valor transferido ( _base_amt_ ), _timestep_ ( _tran_timestamp_ ) e tipo de transação ( _tx_type_ ), para cada transação (além da informação sobre sua ilicitude ( _is_sar_ ) e a conta de origem e destino, _orig_acct_ e _bene_acct_ , usadas na construção dos grafos), e informações de depósito inicial ( _initial_deposit_ ) e categoria de comportamento ( _tx_behavior_id_ ), para cada conta bancária. Uma amostra dos dados de transações e contas financeiras gerados pelo AMLSim, somente com os atributos utilizados neste trabalho, pode ser vista nas tabelas 1 e 2. 

**Table 1: Amostra de dados de transações bancárias geradas pelo AMLSim.** 

|**tran_id**|**orig_acct**|**bene_acct**|**tx_type**|**base_amt**|**tran_timestamp**|**is_sar**|
|---|---|---|---|---|---|---|
|1|743|409|TRANSFER|727.06|2020-01-01|False|
|2|883|751|TRANSFER|397.13|2020-01-01|False|
|3|993|678|CASH_IN|933.56|2020-01-01|False|
|18|774|217|TRANSFER|199.22|2020-01-01|True|



**Table 2: Amostra de dados de contas bancárias geradas pelo AMLSim.** 

|**acct_id**|**initial_deposit**|**tx_behavior_id**|
|---|---|---|
|7|66450.68|1|
|8|71285.46|1|
|9|93684.83|1|
|10|52558.76|1|



De maneira similar ao que foi realizado por [22], os dados gerados foram posteriormente selecionados de forma a variar em cada conjunto a taxa de desbalanceamento de classe, para que fosse possível testar a performance de cada modelo em diferentes taxas de desbalanceamento. A Tabela 3 mostra cada conjunto de dados, sua respectiva taxa de desbalanceamento e a quantidade de exemplos que cada classe possui. 

**Table 3: Informações sobre os conjuntos de dados utilizados.** 

|**Conjunto**|**# de transações**|**# de transações**|**Total de**|**Taxa de**|
|---|---|---|---|---|
|**de dados**|**lícitas**|**ilícitas**|**transações**|**desbalanceamento**|
|AMLSim 1/3|2142|1071|3213|3|
|AMLSim 1/5|4284|1071|5355|5|
|AMLSim 1/10|9639|1071|10710|10|
|AMLSim 1/20|20349|1071|21420|20|



Os dados gerados foram então normalizados via _MinMax_ , sendo que os dados categóricos foram convertidos em múltiplas tabelas de dados binários via _one-hot-encoding_ , ou seja, cada um correspondendo a uma categoria, e, para determinado atributo pertencente a uma determinada categoria, atribui-se 1 para a coluna referente àquela categoria e 0 para as demais. Os dados foram então convertidos para grafos. Para a testagem nas arquiteturas GCN e _Skip_ -GCN, os vértices do grafo são as transações e dois vértices (ou transações) 

256 

Ítalo Della Garza Silva, Luiz Henrique Andrade Correia, and Erick Galani Maziero 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

estão ligadas por uma aresta se tiverem alguma conta em comum entre elas, seja como depositante ou como beneficiário do depósito. 

Para a testagem na arquitetura NENN, cada vértice do grafo é uma conta financeira e as arestas são as transações entre as diferentes contas. Dessa forma, os dados relativos as contas financeiras são utilizados somente nos testes feitos com a NENN. Todas as transações em todos os conjuntos de dados estão classificadas como "lícita" (0) ou "ilícita" (1). A Figura 3 ilustra a diferença entre os tipos de representações geradas para cada conjunto de dados. 



<!-- Start of picture text -->
[0.789, 0.0, 0.0, 0.0, 0.0, 1.0]<br>[0.364, 0.082, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]<br>[0.987, 0.0, 1.0, 0.0, 0.0, 0.0]<br>[0.234, 0.0, 0.0, 0.0, 1.0, 0.0] [0.489, 0.064, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]<br>(a) (b)<br>[0.364, 0.082, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]<br>[0.489, 0.064, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]<br><!-- End of picture text -->

**Figure 3: Diferença entre as representações via grafo do conjunto de dados utilizado. (a) conta bancária como um vértice do grafo e (b) transação como vértice do grafo.** 

## **4.2 Métricas** 

As métricas utilizadas neste trabalho foram Precisão (Equação 14), Revocação (Equação 15) e F1 (Equação 16). Todas as métricas foram calculadas considerando ambas as classes (macro), e também somente para a classe "ilícito". Para comparar os resultados, este trabalho foca na F1 da classe "ilícito", uma vez que ela consegue avaliar tanto a precisão quanto a revocação das classificações para esta classe. Ambas as métricas seriam importantes numa aplicação real, uma vez que, ao passo que se deseja obter o maior número possível de transações ilícitas, também é desejável que o número de falsos positivos seja o menor possível. 





## **5 RESULTADOS** 

A partir dos experimentos descritos na Seção 4, foi coletada a precisão, revocação e F1, tanto no nível macro como para a classe “ilícito”, bem como seus respectivos intervalos de confiança. As métricas resultantes para cada modelo do experimento e separadas pelo conjunto de dados estão dispostas na Tabela 4. 

Nota-se que, dentre todas métricas obtidas nos experimentos realizados nas bases de dados com taxa de desbalanceamento mais baixas, i.e., AMLSim 1/3 e AMLSim 1/5, os modelos baseados em GCN se mostraram superiores, com destaque para a combinação GCN+XGBoost, que obteve maior precisão e F1 para esses dois 

conjuntos de dados. Também vale destacar as revocações obtidas pela arquitetura Skip-GCN, que foram mais altas que as demais, indicando que essa arquitetura é que obteve melhor cobertura para os conjuntos de dados supracitados, ainda que não tenha sido a mais precisa ao classifica-los. A arquitetura NENN obteve métricas mais baixas para esses conjuntos de dados (ainda que tenham sido satisfatórias), sobretudo nos testes em que a classificação foi realizada com o Softmax. Dessa forma, nota-se que a representação das transações como nós do grafo pode ter resultados positivos, ao menos para as taxas de desbalanceamento mais baixas. 

A adição do XGBoost em todos os modelos provocou um aumento expressivo nos resultados. O XGBoost é um modelo que geralmente tem resultados positivos sobre dados tabulares [13] e sua inclusão nos modelos incrementou a capacidade dos mesmos ao lidar com a complexidade dos conjuntos de dados e sua resistência ao desequilíbrio de classe. Considerando-se os resultados sobre os conjuntos de dados AMLSim 1/10 e AMLSim 1/20, nota-se que a adição do XGBoost à arquitetura NENN fez com que sua F1, que de maneira geral era a pior em relação aos outros modelos, se tornasse a melhor. 

## **5.1 Impacto do desbalanceamento de classe** 

O efeito do desbalanceamento começa a ser perceptível quando a taxa de desbalanceamento é igual a 10, sobretudo nos valores de precisão para a classe “ilícito”. No cenário com as taxas de desbalanceamento mais altas, i.e, AMLSim 1/10 e AMLSim 1/20, a combinação NENN+XGBoost obteve precisão e F1 superiores às demais, exceto para o conjunto AMLSim 1/20, para o qual a maior precisão foi obtida pela combinação Skip-GCN+XGBoost. A arquitetura NENN combinada com a classificação por Softmax atingiu a melhor macro-revocação para o conjunto de dados AMLSim 1/20 e a melhor revocação para a classe “ilícita” no conjunto AMLSim 1/10. No entanto, a arquitetura apresenta baixa precisão em ambos os conjuntos de dados, o que indica que ela está atribuindo muitas instâncias para a classe “ilícita”, sem necessariamente filtrar por aquelas que realmente são ilícitas. A baixa precisão dessa arquitetura afetou também a F1 do modelo, piorando-o em relação aos demais. Por fim, a GCN combinada com Softmax atingiu a melhor macro-revocação no conjunto AMLSim 1/10. É notável que a representação das transações como arestas do grafo se saiu melhor para taxas de desbalanceamento maiores. 

A Figura 4 mostra o impacto do desbalanceamento de classe na F1 e na precisão para a classe "ilícito" para o modelo GCN, com e sem XGBoost. Nota-se uma queda significativa em ambas as métricas quando a taxa de desbalanceamento sobe de 5 para 10. Além disso, destaca-se o efeito positivo do uso do XGBoost como classificador para valores mais altos de desbalanceamento de classe, sobretudo na precisão. 

Os efeitos do desbalanceamento de classe na F1 e na precisão para a classe "ilícito" para o modelo Skip-GCN são evidenciados pela Figura 5. Nota-se que as métricas foram muito similares às obtidas pelo modelo GCN, de forma que percebe-se o mesmo efeito positivo da adição do XGBoost ao modelo. 

A Figura 6 mostra o efeito do desbalanceamento de classe na precisão e F1 para a classe "ilícito" do modelo NENN, com e sem a adição do XGBoost. Nesse modelo, quando o XGBoost não é usado, percebe-se uma queda mais acentuada nas métricas obtidas em 

257 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

Graph Neural Networks Applied to Money Laundering Detection in Intelligent Information Systems 



<!-- Start of picture text -->
100<br>80<br>60<br>40<br>20 F1 para a classe "ilícito" F1 para a classe "ilícito"<br>Precisão para a classe "ilícito" Precisão para a classe "ilícito"<br>0<br>3 5 10 20 3 5 10 20<br>Taxa de Desbalanceamento Taxa de Desbalanceamento<br>(a) (b)<br>Figure 4: F1 e precisão para a classe "ilícito" do modelo GCN,<br>com os classificadores Softmax (a) e XGBoost (b).<br>100<br>80<br>60<br>40<br>20 F1 para a classe "ilícito" F1 para a classe "ilícito"<br>Precisão para a classe "ilícito" Precisão para a classe "ilícito"<br>0<br>3 5 10 20 3 5 10 20<br>Taxa de Desbalanceamento Taxa de Desbalanceamento<br>(a) (b)<br>Percentual<br>Percentual<br><!-- End of picture text -->

**Figure 4: F1 e precisão para a classe "ilícito" do modelo GCN, com os classificadores Softmax (a) e XGBoost (b).** 

**Figure 5: F1 e precisão para a classe "ilícito" do modelo SkipGCN, com os classificadores Softmax (a) e XGBoost (b).** 

relação aos demais modelos. Logo, o modelo NENN, quando utilizando juntamente com o Softmax para classificação, se mostrou, no cenário abordado por este estudo, mais sensível aos efeitos do desbalanceamento de classe. Porém, com a adição do XGBoost, o modelo se mostra resistente a esses efeitos, inclusive superando os demais em termos de F1 e precisão. Pode-se dizer que a combinação NENN+XGBoost é a mais indicada para a tarefa de detecção apresentada neste estudo. 



<!-- Start of picture text -->
100<br>80<br>60<br>40<br>20 F1 para a classe "ilícito" F1 para a classe "ilícito"<br>Precisão para a classe "ilícito" Precisão para a classe "ilícito"<br>0<br>3 5 10 20 3 5 10 20<br>Taxa de Desbalanceamento Taxa de Desbalanceamento<br>(a) (b)<br>Percentual<br><!-- End of picture text -->

**Figure 6: F1 e precisão para a classe "ilícito" do modelo NENN, com os classificadores Softmax (a) e XGBoost (b).** 

## **6 DISCUSSÃO** 

O combate à Lavagem de Dinheiro é um tema de extrema relevância para diversas organizações, tanto financeiras – para evitar prejuízos e perda de credibilidade perante a sociedade – quanto governamentais – para contribuir com a diminuição da corrupção e promover uma sociedade mais justa, transparente, e com menores índices de criminalidade. A substituição dos processos de detecção de Lavagem de Dinheiro, utilizados atualmente por essas instituições, é de suma importância para a constante evolução e inovação do seu ambiente organizacional, no que tange ao seu meio digital. Nesse sentido, pode-se afirmar que este trabalho contribui ao trazer uma potencial solução que, com as devidas adaptações, pode inovar os SSI utilizados para o combate à Lavagem de Dinheiro de diversas organizações. 

Utilizando como exemplo os resultados do modelo NENN+XGBoost, têm-se aproximadamente 71% das transações ilícitas do conjunto de teste corretamente classificadas, sendo que aproximadamente 77% do total de instâncias do conjunto foram corretamente classificadas como ilícitas. Esses resultados equilibrados evidenciam uma solução que conseguiria, em um ambiente real, detectar uma imensa maioria de transações ilícitas e gerar apenas 30% de transações ilícitas, fazendo a discriminação manual de que são 

**Table 4: Métricas coletadas nos testes sobre os conjuntos de dados.** 

||||**AML**<br>**1/**|**Sim**<br>**3**|||||**AML**<br>**1/**|**Sim**<br>**5**|||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Prec|isão|Revoc|ação|F|1|Prec|isão|Revo|cação|F|1|
|Arquitetura|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|
|GCN|95,31±0,12|91,30±0,23|97,27±0,06|98,43±0,00|96,21±0,09|94,73±0,13|91,67±1,05|83,97±2,14|96,60±0,28|97,32±0,43|93,87±0,73|90,15±1,15|
|Skip-GCN|95,34±0,17|91,35±0,34|**97,28**±**0,08**|**98,44**±**0,00**|96,23±0,14|94,76±0,18|91,70±1,07|84,01±2,16|**96,62**±**0,30**|**97,35**±**0,28**|93,90±0,76|90,19±1,19|
|NENN|94,15±1,49|89,36±2,34|96,34±1,70|97,52±3,00|95,13±1,54|93,26±2,14|90,34±1,61|82,34±3,10|94,14±1,60|92,70±3,18|92,07±1,42|87,20±2,28|
|GCN+XGBoost|**95,79**±**0,65**|**92,80**±**0,99**|96,98±1,00|97,10±2,03|**96,35**±**0,77**|**94,89**±**1,10**|**93,13**±**1,26**|**87,76**±**2,39**|95,23±1,48|93,35±2,94|**94,12**±**1,18**|**90,46**±**1,93**|
|Skip-GCN+XGBoost|95,74±0,63|92,72±1,14|96,94±0,89|97,04±1,86|96,30±0,69|94,83±0,98|93,08±1,24|87,64±2,49|95,23±1,38|93,40±2,89|94,10±1,06|90,42±1,73|
|NENN+XGBoost|95,39±1,36|91,79±2,44|97,00±1,24|97,65±2,08|96,14±1,26|94,61±1,74|92,75±2,16|87,32±4,33|94,47±1,41|91,91±2,79|93,57±1,57|89,53±2,52|
||||**AML**|**Sim**|||||**AML**|**Sim**|||
||||**1/**|**10**|||||**1/**|**20**|||
||Prec|isão|Revoc|ação|F|1|Prec|isão|Revo|cação|F|1|
|Arquitetura|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|macro|classe<br>"ilícito"|
|GCN|85,78±2,91|72,38±5,85|**94,23**±**0,50**|91,99±0,94|89,39±2,00|80,98±3,49|83,15±3,08|67,32±6,17|88,20±1,01|78,19±2,04|85,45±1,90|72,31±3,56|
|Skip-GCN|85,54±2,80|71,90±5,64|94,20±0,49|92,02±1,04|89,23±1,92|80,69±3,34|83,09±2,87|67,20±5,73|88,22±0,82|**78,24**±**1,62**|85,42±1,80|72,27±3,36|
|NENN|79,26±2,59|59,19±5,12|93,57±1,71|**93,60**±**3,17**|84,41±2,40|72,49±4,07|65,36±2,81|31,10±5,63|**91,42**±**3,14**|92,57±6,71|70,59±3,87|46,46±6,30|
|GCN+XGBoost|88,53±2,57|78,62±5,16|89,76±1,77|84,56±4,03|89,76±1,77|81,44±3,17|89,92±4,27|81,31±8,35|83,86±3,15|68,45±6,10|86,60±3,33|74,29±6,39|
|Skip-GCN+XGBoost|88,20±2,52|77,95±5,03|91,16±1,55|84,73±3,16|89,60±1,68|81,17±2,99|**89,98**±**3,46**|**81,45**±**6,86**|83,67±2,06|68,06±4,05|86,51±2,19|74,12±4,20|
|NENN+XGBoost|**90,63**±**1,76**|**82,79**±**3,53**|91,44±2,19|84,65±4,49|**91,02**±**1,49**|**83,68**±**2,73**|88,21±3,17|77,74±6,34|85,35±2,90|71,66±5,89|**86,69**±**2,20**|**74,51**±**4,21**|



258 

Ítalo Della Garza Silva, Luiz Henrique Andrade Correia, and Erick Galani Maziero 

SBSI ’23, May 29–June 01, 2023, Maceió, Brazil 

transações realmente ilícitas feita posteriormente diminua consideravelmente o consumo de tempo e recursos. Vale lembrar que este trabalho usou dados gerados por um simulador, e os resultados podem sofrer variações se os mesmos modelos forem aplicados a conjuntos de transações reais. 

A principal limitação relacionada à solução proposta é o fato de a mesma ter sido treinada com dados fictícios, gerados por um simulador, os quais não trazem consigo todos os ruídos e particularidades de dados reais. Além disso, esta solução não aborda o desvio de conceito, outro fator presente em dados reais de transações bancárias. O desvio de conceito é a mudança nas características estatísticas dos dados ao longo do tempo. 

## **7 CONCLUSÕES E TRABALHOS FUTUROS** 

Este trabalho comparou o desempenho de um conjunto de modelos de Redes Neurais de Grafos e também duas maneiras diferentes de representação das transações via grafos. De uma maneira geral, a representação das transações como nós do grafo pode ter resultados positivos. No entanto, para um maior desbalanceamento de classe, representar os nós como arestas do grafo pode ser bastante eficaz, sobretudo porque os _embeddings_ são gerados tanto a partir dos dados da transação quanto a partir dos dados das contas financeiras. 

Levando em consideração os resultados, conclui-se que, para taxas de desbalanceamento menores, os modelos baseados em GCN são mais apropriados. Para taxas de desbalanceamento maiores, a combinação NENN+XGBoost se mostra mais eficaz. Além disso, como a arquitetura NENN gera _embeddings_ também para os nós, é possível facilmente adaptar o modelo para classificar as contas financeiras, sendo este um importante diferencial em uma situação real. 

Como trabalhos futuros, serão incluídas novas arquiteturas neste experimento, inclusive arquiteturas propositalmente resistentes ao desequilíbrio de classe, além da inclusão de outros métodos de _Machine Learning_ (como o XGBoost, por exemplo), sem o uso dos _embeddings_ . Além disso, pretende-se acrescentar aos experimentos dados de transações financeiras reais, além dos dados gerados automaticamente, o que poderá afetar significativamente os resultados. 

## **REFERENCES** 

- [1] Ismail Alarab and Simant Prakoonwit. 2022. Graph-Based LSTM for Anti-money Laundering: Experimenting Temporal Graph Convolutional Network with Bitcoin Data. _Neural Processing Letters_ 54 (06 2022), 1–19. https://doi.org/10.1007/s11063022-10904-8 

- [2] Ismail Alarab, Simant Prakoonwit, and Mohamed Ikbal Nacer. 2020. Competence of Graph Convolutional Networks for Anti-Money Laundering in Bitcoin Blockchain. In _Proceedings of the 2020 5th International Conference on Machine Learning Technologies_ (Beijing, China) _(ICMLT 2020)_ . Association for Computing Machinery, New York, NY, USA, 23–27. https://doi.org/10.1145/3409073.3409080 

- [3] Renata Araujo. 2017. _Information Systems and the Open World Challenges_ . Brazilian Computer Society (SBC)., 42 – 51. https://doi.org/10.5753/sbc.2884.0.4 

- [4] Henrique Assumpção, Fabrício Souza, Leandro Campos, Vinícius Pires, Paulo Almeida, and Fabrício Murai. 2022. Delator: Detecção Automática de Indícios de Lavagem de Dinheiro por Redes Neurais em Grafos de Transações. In _Anais do XI Brazilian Workshop on Social Network Analysis and Mining_ (Niterói). SBC, Porto Alegre, RS, Brasil, 13–24. https://doi.org/10.5753/brasnam.2022.223137 

   - [7] Rafał Dreżewski, Jan Sepielak, and Wojciech Filipkowski. 2012. System supporting money laundering detection. _Digital Investigation_ 9, 1 (2012), 8–21. https: //doi.org/10.1016/j.diin.2012.04.003 

   - [8] Rafał Dreżewski, Jan Sepielak, and Wojciech Filipkowski. 2015. The application of social network analysis algorithms in a system supporting money laundering detection. _Information Sciences_ 295 (2015), 18–32. https://doi.org/10.1016/j.ins. 2014.10.015 

   - [9] Steven Farrugia, Joshua Ellul, and George Azzopardi. 2020. Detection of illicit accounts over the Ethereum blockchain. _Expert Systems with Applications_ 150 (2020), 113318. https://doi.org/10.1016/j.eswa.2020.113318 

   - [10] FATF. 2003. FATF 40 Recommendations. https://www.fatf-gafi.org/media/fatf/ documents/FATF%20Standards%20-%2040%20Recommendations%20rc.pdf 

   - [11] Andrea Fronzetti Colladon and Elisa Remondi. 2017. Using social network analysis to prevent money laundering. _Expert Systems with Applications_ 67 (2017), 49–58. https://doi.org/10.1016/j.eswa.2016.09.029 

   - [12] Petter Gottschalk. 2010. Categories of financial crime. _Journal of Financial Crime_ 17 (10 2010), 441–458. https://doi.org/10.1108/13590791011082797 

   - [13] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. 2022. Why do tree-based models still outperform deep learning on tabular data? https://doi.org/10.48550/ ARXIV.2207.08815 

   - [14] Willian L. Hamilton. 2020. _Graph Representation Learning_ . Number 46 in Synthesis Lectures on Artifical Intelligence and Machine Learning. Morgan & Claypool, San Rafael – CA, USA. https://doi.org/10.2200/S01045ED1V01Y202009AIM046 

   - [15] Jingguang Han, Yuyun Huang, Sha Liu, and Kieran Towey. 2020. Artificial intelligence for anti-money laundering: a review and extension. _Digital Finance 2020 2:3_ 2, 3 (jun 2020), 211–239. https://doi.org/10.1007/S42521-020-00023-1 

   - [16] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In _International Conference on Learning Representations (ICLR)_ . OpenReview.net, Toulon, France, 14 pages. 

   - [17] Kroll. 2019. _Global Fraud and Risk Report 2019/20_ (11 ed.). Technical Report. Kroll, Boston - MA, US. https://www.kroll.com/en/insights/publications/global-fraudand-risk-report-2019 

   - [18] Asma S. Larik and Sajjad Haider. 2011. Clustering based anomalous transaction reporting. _Procedia Computer Science_ 3 (2011), 606–610. https://doi.org/10.1016/ j.procs.2010.12.101 World Conference on Information Technology. 

   - [19] Nhien-An Le-Khac, Sammer Markos, and Tahar Kechadi. 2009. Towards a New Data Mining-Based Approach for Anti-Money Laundering in an International Investment Bank. In _Lecture Notes of the Institute for Computer Sciences, SocialInformatics and Telecommunications Engineering_ , Vol. 31. Springer, Albany, Ny, USA, 77–84. https://doi.org/10.1007/978-3-642-11534-9_8 

   - [20] Edgar Alonso Lopez-Rojaz and Stefan Axelsson. 2012. Money Laundering Detection using Synthetic Data. In _Linköping Electronic Conference Proceedings, No. 71_ . Linköping University Electronic Press, Linköping, Sweden, 33 – 40. 

   - [21] Vanessa Nunes, Claudia Cappelli, and Célia Ralha. 2017. _Transparency in Information Systems_ . Brazilian Computer Society (SBC)., 73 – 89. https: //doi.org/10.5753/sbc.2884.0.7 

   - [22] Ronald Pereira and Fabrício Murai. 2021. Quão efetivas são Redes Neurais baseadas em Grafos na Detecção de Fraude para Dados em Rede?. In _Anais do X Brazilian Workshop on Social Network Analysis and Mining_ . SBC, Porto Alegre, RS, Brasil, 205–210. https://doi.org/10.5753/brasnam.2021.16141 

   - [23] Toyotaro Suzumura and Hiroki Kanezashi. 2021. Anti-Money Laundering Datasets: InPlusLab Anti-Money Laundering DataDatasets. http://github.com/ IBM/AMLSim 

   - [24] Mark Weber, Jie Chen, Toyotaro Suzumura, Aldo Pareja, Tengfei Ma, Hiroki Kanezashi, Tim Kaler, Charles E. Leiserson, and Tao B. Schardl. 2018. Scalable Graph Learning for Anti-Money Laundering: A First Look. _CoRR_ abs/1812.00076 (2018), 7 pages. 

   - [25] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I. Weidele, Claudio Bellei, Tom Robinson, and Charles E. Leiserson. 2019. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. , 7 pages. https://drive.google.com/drive/folders/1r_iJYFJru-jdDdgpBKZ1N0Zathy2LD2 

   - [26] Yulei Yang and Dongsheng Li. 2020. NENN: Incorporate Node and Edge Features in Graph Neural Networks. In _Proceedings of The 12th Asian Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 129)_ , Sinno Jialin Pan and Masashi Sugiyama (Eds.). PMLR, Bangkok, Thailand, 593–608. https:// proceedings.mlr.press/v129/yang20a.html 

- [5] Zhiyuan Chen, D. Van-Khoa Le, Ee Teoh, Amril Nazir, Ettikan Karuppiah, and Kim Lam. 2018. Machine learning techniques for anti-money laundering (AML) solutions in suspicious transaction detection: a review. _Knowledge and Information Systems_ 57 (11 2018), 245–285. https://doi.org/10.1007/s10115-017-1144-z 

- [6] Luis Fernando Carvalho Dias and Fernando Silva Parreiras. 2019. Comparing Data Mining Techniques for Anti-Money Laundering. In _Proceedings of the XV Brazilian Symposium on Information Systems_ (Aracaju, Brazil) _(SBSI’19)_ . Association for Computing Machinery, New York, NY, USA, Article 73, 8 pages. https://doi.org/ 10.1145/3330204.3330283 

259 

