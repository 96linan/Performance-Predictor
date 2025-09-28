

# Literature Details

Title：[RATs-NAS: Redirection of Adjacent Trails on Graph Convolutional Networks for Predictor-Based Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10685480)  
Publication：IEEE Transactions on Artificial Intelligence-2024  
Source Code：-----  
Architecture Sampling：Random sampling  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RAT-GCN (RAT: adaptive weighting of adjacency matrix)  
Model Training：Supervised  
Interaction with Search Algorithm：Online: partition-based search sampling  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[CIMNet: Joint Search for Neural Network and Computing-in-Memory Architecture](https://ieeexplore.ieee.org/abstract/document/10551739)  
Publication：IEEE Micro-2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Vector-based Fixed length: architecture features + quantization strategy (Quant) + CIM hardware parameters (OU count, ADC resolution, cell precision, etc.) + input image resolution  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[HKT-FNAS: Search Robust Neural Architecture via Heterogeneous Knowledge Transfer for Defect Detection of PV Energy Storage Modules](https://file.sciopen.com/sciopen_public/1941065596264796162.pdf)  
Publication：Tsinghua Science and Technology, 2025  
Source Code：-----  
Architecture Sampling：Latin Hypercube Sampling (LHS)  
Architecture Representation：Single-modal Vector-based Variable length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RBF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: TOP-K  
Transferability on Task/Dataset/Search Space: With Transferability (progressive search space with online updating)  
***  
Title：[Renas: Relativistic evaluation of neural architecture search](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_ReNAS_Relativistic_Evaluation_of_Neural_Architecture_Search_CVPR_2021_paper.pdf)  
Publication：CVPR 2021
Source Code：https://www.mindspore.cn/resources/hub/
Architecture Sampling：Random
Architecture Representation：Single-modal/Graph-based encoding  
Objective Dimension：Single-objective (Accuracy) 
Prediction Type：Relative performance (Pairwise)  
Model Selection：LeNet   
Model Training：Supervised  
Interaction with Search Algorithm：Offline
Transferability on Task/Dataset/Search Space: Without Transferability
***  
Title：[Accuracy Prediction with Non-neural Model for Neural Architecture Search](https://arxiv.org/pdf/2007.04785)  
Publication：2021-Arxiv
Source Code：https://github.com/renqianluo/GBDT-NAS.
Architecture Sampling：Random
Architecture Representation：Single-modal/Sequence-based encoding  
Objective Dimension：Single-objective (Accuracy) 
Prediction Type：Absolute performance
Model Selection：GBDT   
Model Training：Supervised  
Interaction with Search Algorithm：Online/Top-K selection
Transferability on Task/Dataset/Search Space: Without Transferability
***  
Title：[CARL: Causality-guided Architecture Representation Learning for an Interpretable Performance Predictor](https://arxiv.org/abs/2506.04001)   
Publication：ICCV 2025  
Source Code：https://github.com/jihan4431/CARL  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based Key features + redundant features ---> cross-architecture random recombination  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[From Hand-Crafted Metrics to Evolved Training-Free Performance Predictors for Neural Architecture Search via Genetic Programming](https://arxiv.org/abs/2505.15832)  
Publication：2025-arXiv  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Zero-cost indicators: FLOPs, Snip, L2-norm, Zen, ZiCo, MeCo, etc.  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Zero-cost symbolic function expressions  
Model Training：Evolved through Symbolic Regression (SR) + Genetic Programming  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (direct transfer without fine-tuning)  
***  
Title：[Listwise ranking predictor for evolutionary neural architecture search](https://www.sciencedirect.com/science/article/abs/pii/S2210650225001142)  
Publication：Swarm and Evolutionary Computation, 2025  
Source Code：https://github.com/96linan/LRP  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (listwise)  
Model Selection：RBF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A lightweight neural network search algorithm based on in-place distillation and performance prediction for hardware-aware optimization](https://www.sciencedirect.com/science/article/abs/pii/S0952197625007754)  
Publication：Engineering Applications of Artificial Intelligence, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length    
Objective Dimension：Multi-objective ACC + Inference latency + Params  
Prediction Type：Absolute performance  
Model Selection：Gaussian Process + Customized GloKernel  
Model Training：Supervised  
Interaction with Search Algorithm：Online: randomly select t from top-h  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A holistic approach for resource-constrained neural network architecture search](https://www.sciencedirect.com/science/article/pii/S1568494625001437)  
Publication：Applied Soft Computing, 2025  
Source Code：-----  
Architecture Sampling：Manual selection + Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance, ranking-aware (1 if better than ancestor, 0 otherwise)  
Model Selection：Random Forest + k-Nearest Neighbors + Gradient Boosting Classifier  
Model Training：Supervised  
Interaction with Search Algorithm：Online: predictor judges as 1, then taken as new sample  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[PerfSeer: An Efficient and Accurate Deep Learning Models Performance Predictor](https://arxiv.org/abs/2502.01206)  
Publication：2025-arXiv  
Source Code：https://github.com/upuuuuuu/PerfSeer  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Graph-based: global features (node count, edge count, graph density, FLOPs, memory access statistics, batch size, arithmetic intensity (FLOPs/memory access)) + node features (kernel size, FLOPs, memory access info (MAC, total input/weight/output tensor size), arithmetic intensity, proportion in the whole model) + edge features (data flow: tensor size + tensor shape)  
Objective Dimension：Multi-objective ACC + Execution time + Memory usage + Streaming Multiprocessor Utilization (SM Utilization)  
Prediction Type：Absolute performance  
Model Selection：SeerNet (GNN) + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Graph Masked Autoencoder Enhanced Predictor for Neural Architecture Search](https://www.ijcai.org/proceedings/2022/0432.pdf)  
Publication：IJCAI-2022  
Source Code：https://github.com/kunjing96/GMAENAS  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GAT + MLP  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Neural architecture search with interpretable meta-features and fast predictors](https://www.sciencedirect.com/science/article/pii/S0020025523012276?casa_token=mSCJN6wKExUAAAAA:Any8R4hT8DhyyoognrX49JrsNCdw6nhoDmb-UkETpsaSQpM9IVagKE4HUdU3kw2M2guI34e5wNSr)  
Publication：Information Sciences, 2023  
Source Code：-----  
Architecture Sampling：Random 
Architecture Representation：Multi-modal (Architecture encoding, search space meta-features, and #Params)
Objective Dimension：Single-objective (Accuracy)  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Semi-supervised neural architecture search](https://proceedings.neurips.cc/paper/2020/hash/77305c2f862ad1d353f55bf38e5a5183-Abstract.html)  
Publication：NeurIPS, 2020  
Source Code：-----  
Architecture Sampling：Random
Architecture Representation：Single-modal Sequence-based  
Objective Dimension：Single-objective (Accuracy)  
Prediction Type：Absolute performance  
Model Selection：LSTM   
Model Training：Semi-Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Score Predictor-Assisted Evolutionary Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10841460)  
Publication：IEEE Transactions on Emerging Topics in Computational Intelligence, 2025  
Source Code：-----  
Architecture Sampling：All individuals from the top N generations  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective (error rate)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Progressive Neural Architecture Generation with Weaker Predictors](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17)  
Publication：International Conference on Multimedia Modeling, 2024  
Source Code：-----  
Architecture Sampling：Unconditional diffusion model generation  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：XGBoost  
Model Training：Supervised  
Interaction with Search Algorithm：Online, re-training with ranking-weighted samples  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[HGNAS: Hardware-Aware Graph Neural Architecture Search for Edge Devices](https://ieeexplore.ieee.org/abstract/document/10644077)  
Publication：IEEE Transactions on Computers, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Graph-based (architecture graph + device one-hot)  
Objective Dimension：Single-objective with 2 predictors (latency + peak memory usage)  
Prediction Type：Absolute performance  
Model Selection：GNN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Rethinking neural architecture representation for predictors: Topological encoding in pixel space](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036)  
Publication：Information Fusion, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Graph encoding + Image representation + Text representation  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Multi-channel predictor: CLIP-ViT/RN50 (image representation), CLIP-Text Encoder (text representation), GCN (graph encoding)  
Model Training：Mainly supervised, supplemented with self-supervised  
Interaction with Search Algorithm：NB101 / NB201 offline, DARTS online  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Shapley-guided pruning for efficient graph neural architecture prediction in distributed learning environments](https://www.sciencedirect.com/science/article/pii/S0020025524016098)  
Publication：Information Sciences, 2025  
Source Code：https://github.com/BeObm/DGNAP  
Architecture Sampling：Controlled hierarchical random sampling  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: re-encode pruned search space as new samples  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：MLP-GNAS: [Meta-learning-based predictor-assisted Genetic Neural Architecture Search system](https://www.sciencedirect.com/science/article/pii/S1568494624013012)    
Publication：Applied Soft Computing, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：CNN-based performance predictor  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Evolutionary Neural Architecture Search with Performance Predictor Based on Hybrid Encodings](https://ieeexplore.ieee.org/abstract/document/10805362)  
Publication：International Conference on Information Science and Technology, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Transformer-based VAE + Graph flow-based encoding module  
Model Training：Self-supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Single-Domain Generalized Predictor for Neural Architecture Search System](https://ieeexplore.ieee.org/abstract/document/10438213)  
Publication：IEEE Transactions on Computers, 2024  
Source Code：-----  
Architecture Sampling：Sampling from multiple task datasets in the source domain  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + FCN  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (can transfer to unseen target search spaces)  
***  
Title：[Delta-NAS: Difference of Architecture Encoding for Predictor-based Evolutionary Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10943463)  
Publication：IEEE/CVF Winter Conference on Applications of Computer Vision, 2025  
Source Code：-----  
Architecture Sampling：Random sampling of architecture pairs  
Architecture Representation：Difference-based encoding between architectures  
Objective Dimension：Single-objective  
Prediction Type：Relative performance  
Model Selection：-----  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Improving Routability Prediction via NAS Using a Smooth One-Shot Augmented Predictor](https://ieeexplore.ieee.org/abstract/document/11014419)    
Publication：International Symposium on Quality Electronic Design, 2025  
Source Code：-----  
Architecture Sampling：Sampling subnets from multiple one-shot networks  
Architecture Representation：Multi-modal (architecture features + Params + FLOPs + memory demand) Vector-based Fixed length  
Objective Dimension：Single-objective (AUC)  
Prediction Type：Absolute performance  
Model Selection：XGBoost  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Accelerating one-shot neural architecture search via constructing a sparse search space](https://www.sciencedirect.com/science/article/pii/S0950705124012541)  
Publication：Knowledge-Based Systems, 2024  
Source Code：-----  
Architecture Sampling：Random subnet sampling from supernet  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GBDT  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[CNNGen: A Generator and a Dataset for Energy-Aware Neural Architecture Search](https://inria.hal.science/hal-04957651/)  
Publication：European Symposium on Artificial Neural Networks, 2024  
Source Code：-----  
Architecture Sampling：CNNGen generates samples using grammar-based generation  
Architecture Representation：Multi-modal PNG image + Python code + Vector (architecture features + epochs + FLOPs + params)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Image predictor + Code predictor + Decision Tree  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Set-Nas: Sample-Efficient Training For Neural Architecture Search With Strong Predictor And Stratified Sampling](https://ieeexplore.ieee.org/abstract/document/10647243)  
Publication：IEEE International Conference on Image Processing, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Progressive Neural Predictor with Score-Based Sampling](https://ieeexplore.ieee.org/abstract/document/10651529)  
Publication：IJCNN, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: score-based sampling (score top-k)  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Semi-supervised accuracy predictor-based multi-objective neural architecture search](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438)  
Publication：Neurocomputing, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：SVR / RF / XGB  
Model Training：Semi-supervised  
Interaction with Search Algorithm：Online: update two datasets respectively with prediction accuracy + prediction confidence, also used to update predictor  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Inference latency prediction for CNNs on heterogeneous mobile devices and ML frameworks](https://www.sciencedirect.com/science/article/abs/pii/S0166531624000348)  
Publication：Performance Evaluation, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based  
Objective Dimension：Single-objective (end-to-end inference latency)  
Prediction Type：Absolute performance  
Model Selection：Lasso / Random Forest / GBDT / MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[CAP: A Context-Aware Neural Predictor for NAS](https://arxiv.org/abs/2406.02056)  
Publication：2024-arXiv  
Source Code：https://github.com/jihan4431/CAP  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GIN  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Fine-grained complexity-driven latency predictor in hardware-aware neural architecture search using composite loss](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972)  
Publication：Information Sciences, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal: Graph-based + Complexity vector (FLOPs, Params, Activations)  
Objective Dimension：Single-objective (Latency)  
Prediction Type：Absolute performance  
Model Selection：GCN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A Neural Architecture Predictor based on GNN-Enhanced Transformer](https://proceedings.mlr.press/v238/xiang24a.html)  
Publication：International Conference on Artificial Intelligence and Statistics, 2024  
Source Code：https://github.com/GNET  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GNN + Transformer  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Multi-population evolutionary neural architecture search with stacked generalization](https://www.sciencedirect.com/science/article/abs/pii/S0925231224004351)  
Publication：Neurocomputing, 2024  
Source Code：-----  
Architecture Sampling：All individuals from the first three generations for training set, first generation randomly generated  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：KNN + RF + SVM  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[GreenNAS: A Green Approach to the Hyperparameters Tuning in Deep Learning](https://www.mdpi.com/2227-7390/12/6/850)  
Publication：Mathematics, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF / SVR  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[On Latency Predictors for Neural Architecture Search](https://proceedings.mlsys.org/paper_files/paper/2024/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html)  
Publication：Proceedings of Machine Learning and Systems, 2024  
Source Code：https://github.com/abdelfattah-lab/nasflat  
Architecture Sampling：Encoding-based, select diverse samples using Cosine Similarity + KMeans clustering, then fine-tune on target device with a few samples  
Architecture Representation：Multi-modal: Graph-based + Hardware embeddings  
Objective Dimension：Single-objective (Inference latency)  
Prediction Type：Absolute performance  
Model Selection：GAT + Dense Graph Flow (DGF)  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Latency-Constrained Neural Architecture Search Method for Efficient Model Deployment on RISC-V Devices](https://www.mdpi.com/2079-9292/13/4/692)  
Publication：Electronics, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective (Inference latency)  
Prediction Type：Absolute performance  
Model Selection：DNN  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Towards Efficient Neural Networks Through Predictor-Assisted NSGA-III for Anomaly Traffic Detection of IoT](https://ieeexplore.ieee.org/abstract/document/10403928)  
Publication：IEEE Transactions on Cognitive Communications and Networking, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (Error rate, FLOPs, MAC)  
Prediction Type：Absolute performance  
Model Selection：CART  
Model Training：Supervised  
Interaction with Search Algorithm：Online: sampling from Pareto front  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[AutoGAN-DSP: Stabilizing GAN architecture search with deterministic score predictors](https://www.sciencedirect.com/science/article/abs/pii/S0925231223013103)  
Publication：Neurocomputing, 2024  
Source Code：https://github.com/APinCan/GAN_Architecture_Search_with_Predictors  
Architecture Sampling：Based on substructures selected by RL learner + generated complete GAN architectures  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective (2 predictors: Inception Score + Frechet Inception Distance)  
Prediction Type：Absolute performance  
Model Selection：GCN + LSTM + FCN  
Model Training：Supervised  
Interaction with Search Algorithm：Online: initially updated with RL-selected substructures, later not updated  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A Sampling Method for Performance Predictor Based on Contrastive Learning](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_18)  
Publication：Australasian Joint Conference on Artificial Intelligence, 2023  
Source Code：-----  
Architecture Sampling：Representative samples chosen via architecture augmentation and clustering  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + (RF / DT / SVR / KNN / GBRT)  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[PINAT: A Permutation INvariance Augmented Transformer for NAS Predictor](https://ojs.aaai.org/index.php/AAAI/article/view/26076)  
Publication：AAAI, 2023  
Source Code：https://github.com/ShunLu91/PINAT  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Transformer + FCN  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[How predictors affect the RL-based search strategy in Neural Architecture Search?](https://www.sciencedirect.com/science/article/abs/pii/S0957417423022443)    
Publication：Expert Systems with Applications, 2024  
Source Code：https://github.com/tjdeng/RPNASM  
Architecture Sampling：RL controller-based strategy sampling  
Architecture Representation：Single-modal Graph/Vector-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP / GCN / BANANAS / BOHAMIANN / BONAS / NAO / SemiNAS / Transformer  
Model Training：Supervised  
Interaction with Search Algorithm：Online: sampling based on continuously optimized controller strategy  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[DGL: Device Generic Latency Model for Neural Architecture Search on Mobile Devices](https://ieeexplore.ieee.org/abstract/document/10042973)  
Publication：TMC, 2023  
Source Code：-----  
Architecture Sampling：-----  
Architecture Representation：Multi-modal (Architecture + Device configuration)  
Objective Dimension：Single-objective (Latency)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Convolutional architecture search based on particle swarm algorithm for functional brain network classification](https://www.sciencedirect.com/science/article/abs/pii/S1568494623010670)  
Publication：Applied Soft Computing, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：SVM  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[LLM Performance Predictors are good initializers for Architecture Search](https://arxiv.org/abs/2310.16712)  
Publication：arXiv, 2023  
Source Code：https://github.com/UBC-NLP/llmas  
Architecture Sampling：Random  
Architecture Representation：Text (prompt)  
Objective Dimension：Single-objective (BLEU score)  
Prediction Type：Absolute performance  
Model Selection：LLM  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Pruning Large Language Models via Accuracy Predictor](https://arxiv.org/abs/2309.09507)  
Publication：arXiv, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GBDT  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Hidden Design Principles in Zero-Cost Performance Predictors for Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10191474)  
Publication：IJCNN, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Network hyperparameters (depth, width, group width, etc.)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Zero-cost predictors (grad_norm / SNIP / GRASP / SynFlow / Fisher / Jacov / logdet)  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[MPENAS: Multi-fidelity Predictor-guided Evolutionary Neural Architecture Search with Zero-cost Proxies](https://dl.acm.org/doi/abs/10.1145/3583131.3590513)  
Publication：GECCO, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (Architecture encoding + ZC proxy outputs + Learning curve data)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：NGBoost  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-n  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Fast Evolutionary Neural Architecture Search by Contrastive Predictor with Linear Regions](https://dl.acm.org/doi/abs/10.1145/3583131.3590452)  
Publication：GECCO, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：GCN  
Model Training：Supervised  
Interaction with Search Algorithm：Online: active learning-based  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Latency-Aware Neural Architecture Performance Predictor with Query-to-Tier Technique](https://ieeexplore.ieee.org/abstract/document/10155437)  
Publication：IEEE Transactions on Circuits and Systems for Video Technology, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (structural features, FLOPs, #parameters) Feature vector-based Fixed length  
Objective Dimension：Multi-objective (Accuracy + Latency)  
Prediction Type：Accuracy: Relative performance (column-aware) + Absolute performance (tiered scoring)  
Model Selection：Transformer  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Multi-Predict: Few Shot Predictors For Efficient Neural Architecture Search](https://arxiv.org/abs/2306.02459)  
Publication：arXiv, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Fixed length — Accuracy → Vector-based (zero-cost metrics: parameter gradients, gradient norms, activation sensitivity); Latency → Vector-based (latency on different devices)  
Objective Dimension：Single-objective with 2 predictors (Accuracy + Latency)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (fine-tuning on new search space)  
***  
Title：[Architecture Augmentation for Performance Predictor via Graph Isomorphism](https://ieeexplore.ieee.org/abstract/document/10109990)  
Publication：TCYB, 2023  
Source Code：-----  
Architecture Sampling：Random + isomorphism augmentation  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Pooling Graph Convolutional Networks for Structural Performance Prediction](https://link.springer.com/chapter/10.1007/978-3-031-25891-6_1)  
Publication：International Conference on Machine Learning, Optimization, and Data Science, 2022  
Source Code：https://github.com/wendli01/morp_gcn  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Graph-based (operations + kernel size, number of filters + node centrality measures such as harmonic centrality)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[DCLP: Neural Architecture Predictor with Curriculum Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29649)  
Publication：AAAI, 2023  
Source Code：https://github.com/Zhengsh123/DCLP  
Architecture Sampling：Random + data augmentation  
Architecture Representation：Single-modal Graph-based (positive samples perturbed by edge disturbance and attribute masking)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GIN  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A General-Purpose Transferable Predictor for Neural Architecture Search](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch81)  
Publication：Proceedings of the 2023 SIAM International Conference on Data Mining, 2023  
Source Code：-----  
Architecture Sampling：Random + non-random perturbation augmentation  
Architecture Representation：Single-modal Computational Graph (CG)-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GNN + MLP  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (fine-tuning on new search space)  
***  
Title：[Neural predictor-based automated graph classifier framework](https://link.springer.com/article/10.1007/s10994-022-06287-5)  
Publication：Machine Learning, 2022  
Source Code：-----  
Architecture Sampling：Component-based controlled hierarchical random sampling  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[PRE-NAS: Evolutionary Neural Architecture Search With Predictor](https://ieeexplore.ieee.org/abstract/document/9975797)  
Publication：TEVC, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF / GCN / SVR / MLP / Bayesian Ridge / Kernel Ridge / Linear  
Model Training：Supervised  
Interaction with Search Algorithm：Online: percentile-based stratified sampling according to prediction validation accuracy  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Work-in-Progress: Utilizing latency and accuracy predictors for efficient hardware-aware NAS](https://ieeexplore.ieee.org/abstract/document/9943117)  
Publication：International Conference on Hardware/Software Codesign and System Synthesis, 2022  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：-----  
Objective Dimension：Multi-objective (Accuracy + Latency)  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[BERTPerf: Inference Latency Predictor for BERT on ARM big.LITTLE Multi-Core Processors](https://ieeexplore.ieee.org/abstract/document/9919203)  
Publication：IEEE Workshop on Signal Processing Systems, 2022  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based (BERT hyperparameter configurations)  
Objective Dimension：Single-objective (Inference latency)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Generalized Global Ranking-Aware Neural Architecture Ranker for Efficient Image Classifier Search](https://dl.acm.org/doi/abs/10.1145/3503161.3548149)  
Publication：Proceedings of the 30th ACM International Conference on Multimedia, 2022  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (structural features, FLOPs, #parameters) Feature vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (column-aware) + Absolute performance (tiered scoring)  
Model Selection：Transformer  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Towards Leveraging Structure for Neural Predictor in NAS](https://cke.um.ac.ir/article_42708.html)  
Publication：Computer and Knowledge Engineering, 2022  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GBDT  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[MAPLE-Edge: A Runtime Latency Predictor for Edge Devices](https://openaccess.thecvf.com/content/CVPR2022W/EVW/html/Nair_MAPLE-Edge_A_Runtime_Latency_Predictor_for_Edge_Devices_CVPRW_2022_paper.html)  
Publication：CVPR, 2022  
Source Code：-----  
Architecture Sampling：Targeted uniform sampling  
Architecture Representation：Multi-modal (Hardware descriptors + Architecture encoding)  
Objective Dimension：Single-objective (Latency)  
Prediction Type：Absolute performance  
Model Selection：-----  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Arch-Graph: Acyclic Architecture Relation Predictor for Task-Transferable Neural Architecture Search](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Arch-Graph_Acyclic_Architecture_Relation_Predictor_for_Task-Transferable_Neural_Architecture_Search_CVPR_2022_paper.html)  
Publication：CVPR, 2022  
Source Code：https://github.com/Centaurus982034/Arch-Graph  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (Architecture encoding: Graph-based + Task embeddings: Vector-based including data distribution, task requirements, input/output types, etc.)  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware + column-aware)  
Model Selection：GCN (relation-aware) + Architecture relation graph construction (column-aware)  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (fine-tuning on new tasks)  
***  
Title：[WPNAS: Neural Architecture Search by jointly using Weight Sharing and Predictor](https://arxiv.org/abs/2203.02086)    
Publication：arXiv, 2022    
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector/Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP / RNN / Transformer  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Automated design of CNN architecture based on efficient evolutionary search](https://www.sciencedirect.com/science/article/abs/pii/S092523122200340X)  
Publication：Neurocomputing, 2022  
Source Code：-----  
Architecture Sampling：All individuals from the first three generations  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[CURIOUS: Efficient Neural Architecture Search Based on a Performance Predictor and Evolutionary Search](https://ieeexplore.ieee.org/abstract/document/9698855)  
Publication：IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2022  
Source Code：-----  
Architecture Sampling：Iterative method; first generation sampled randomly from initial pool generated by QMC, then trained predictor selects top-k from remaining pool until termination  
Architecture Representation：-----  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GBDT  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[TNASP: A Transformer-based NAS Predictor with a Self-evolution Framework](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7fa1575cbd7027c9a799983a485c3c2f-Abstract.html)  
Publication：NeurIPS, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based (Laplace matrix)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Transformer + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: using historical evaluation data as constraints to guide predictor updates  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Not All Operations Contribute Equally: Hierarchical Operation-Adaptive Predictor for Neural Architecture Search](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Not_All_Operations_Contribute_Equally_Hierarchical_Operation-Adaptive_Predictor_for_Neural_ICCV_2021_paper.html)  
Publication：ICCV, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Operation-adaptive attention module + Cell-level hierarchical gating module  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[FBNetV3: Joint Architecture-Recipe Search Using Predictor Pretraining](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html)  
Publication：CVPR, 2021  
Source Code：-----  
Architecture Sampling：Constraint-based iterative optimization  
Architecture Representation：Multi-modal (Architecture features + Training recipes)  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Self-supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[SSRNAS: Search Space Reduced One-shot NAS by a Recursive Attention-based Predictor with Cell Tensor-flow Diagram](https://ieeexplore.ieee.org/abstract/document/9533297)  
Publication：IJCNN, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Recursive GAT  
Model Training：Supervised  
Interaction with Search Algorithm：Online: sampling from iteratively optimized architecture pool  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[RANK-NOSH: Efficient Predictor-Based Architecture Search via Non-Uniform Successive Halving](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html)  
Publication：ICCV, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：GIN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Homogeneous Architecture Augmentation for Neural Predictor](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Homogeneous_Architecture_Augmentation_for_Neural_Predictor_ICCV_2021_paper.html)  
Publication：CoRR, 2021  
Source Code：https://github.com/lyq998/HAAP  
Architecture Sampling：Random + isomorphism augmentation  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html)  
Publication：NeurIPS, 2021  
Source Code：https://github.com/HayeonLee/HELP  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (Graph/Vector-based + Hardware embeddings)  
Objective Dimension：Single-objective (Latency)  
Prediction Type：Absolute performance  
Model Selection：GCN / MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (fine-tuning on new devices)  
***  
Title：[Pareto-Optimal Progressive Neural Architecture Search](https://dl.acm.org/doi/abs/10.1145/3449726.3463146)  
Publication：Proceedings of the Genetic and Evolutionary Computation Conference Companion, 2021  
Source Code：-----  
Architecture Sampling：Exhaustive search over all 1-block cells  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (Accuracy + Training time)  
Prediction Type：Absolute performance  
Model Selection：LSTM (Accuracy) + NNLS + Dynamic re-indexing (Training time)  
Model Training：Supervised  
Interaction with Search Algorithm：Online: sampling from Pareto front after each cell expansion  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Neural predictor based quantum architecture search](https://iopscience.iop.org/article/10.1088/2632-2153/ac28dd/meta)  
Publication：Machine Learning: Science and Technology, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Tensor-based Variable length  
Objective Dimension：Single-objective (Validation accuracy / Ground state energy)  
Prediction Type：Absolute performance  
Model Selection：CNN + LSTM  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (across different circuit depths and quantum bit scales)  
***  
Title：[A Novel Training Protocol for Performance Predictors of Evolutionary Neural Architecture Search Algorithms](https://ieeexplore.ieee.org/document/9336721)  
Publication：TEVC, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：SVM + GBDT + DTree + RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Reducing energy consumption of Neural Architecture Search: An inference latency prediction framework](https://www.sciencedirect.com/science/article/abs/pii/S221067072100041X)  
Publication：Sustainable Cities and Society, 2021  
Source Code：-----  
Architecture Sampling：Random + increased sampling probability for high-latency architectures  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective (Inference latency)  
Prediction Type：Absolute performance  
Model Selection：LSTM + LightGBM  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Accelerating neural architecture search using performance prediction](https://arxiv.org/pdf/1705.10823)  
Publication：2017-arXiv  
Source Code：-----  
Architecture Sampling：Random sampling  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective (Accuracy)  
Prediction Type：Absolute performance  
Model Selection：SVM and RF
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  



Title：[Neural Predictor for Neural Architecture Search](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_39)  
Publication：European conference on computer vision, 2020  
Source Code：https://github.com/ultmaster/neuralpredictor.pytorch  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted Evolutionary Deep Learning Using an End-to-End Random Forest-Based Performance Predictor](https://ieeexplore.ieee.org/abstract/document/8744404)  
Publication：TEVC, 2020  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A Generic Graph-Based Neural Architecture Encoding Scheme for Predictor-Based NAS](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_12)  
Publication：European conference on computer vision, 2020  
Source Code：https://github.com/czyczyyzc/GATES  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GNN + MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：With Transferability (usable across heterogeneous search spaces: operation-on-node and operation-on-edge)  
***  
Title：[NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/9723446)  
Publication：TNNLS,2022  
Source Code：-----  
Architecture Sampling：Random sampling on the key set  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN + GIN  
Model Training：Supervised    
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Efficient Sampling for Predictor-Based Neural Architecture Search](https://arxiv.org/abs/2011.12043)  
Publication：2021-arXiv  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：GCN  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[TAPAS: Train-Less Accuracy Predictor for Architecture Search](https://ojs.aaai.org/index.php/AAAI/article/view/4282)  
Publication：AAAI, 2019  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (architecture features + total layers + FLOPs + inference memory + cumulative predicted accuracy) Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：LSTM    
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：With Transferability (directly usable on new datasets)  
***  
Title：[Ranking-based architecture generation for surrogate-assisted neural architecture search](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051)  
Publication：Concurrency and Computation: Practice and Experience, 2024  
Source Code：https://github.com/outofstyle/RAGS-NAS  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted Evolutionary Neural Architecture Search with Architecture Knowledge Transfer](https://ieeexplore.ieee.org/abstract/document/11043035)  
Publication：CEC, 2025  
Source Code：-----  
Architecture Sampling：Latin Hypercube Sampling (LHS)  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RBFN  
Model Training：Supervised  
Interaction with Search Algorithm：Online: the global and local predictors each select one best individual for true evaluation and add to the pool; the global predictor is retrained on the entire pool, while the local predictor is retrained on the pool’s top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Benchmarking Quantum Architecture Search with Surrogate Assistance](https://arxiv.org/abs/2506.06762)  
Publication：2025-arXiv  
Source Code：https://github.com/SQuASH-bench/SQuASH  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based (GCN) / Vector-based (RF)   
Objective Dimension：Single-objective (circuit fidelity + ACC)  
Prediction Type：Absolute performance  
Model Selection：GCN / RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Pruning for efficient DenseNet via surrogate-model-assisted genetic algorithm considering neural architecture search proxies](https://www.sciencedirect.com/science/article/abs/pii/S2210650225001415)  
Publication：Swarm and Evolutionary Computation,2025  
Source Code：https://github.com/JingeunKim/DenseNetPruning  
Architecture Sampling：Random  
Architecture Representation：Single-modal Multi-dimensional tensor-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：CART  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top 10%  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[SiamNAS: Siamese Surrogate Model for Dominance Relation Prediction in Multi-objective Neural Architecture Search](https://dl.acm.org/doi/abs/10.1145/3712256.3726359)  
Publication：Proceedings of the Genetic and Evolutionary Computation Conference, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + #params + FLOPs)  
Prediction Type：Relative performance (relation-aware)    
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Multi-objective evolutionary neural architecture search for medical image classification](https://www.sciencedirect.com/science/article/abs/pii/S1568494625005903)  
Publication：Applied Soft Computing, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + computational efficiency + (medical) resource usage/deployability)  
Prediction Type：Absolute performance  
Model Selection：LLM-enhanced RF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: cluster-based sampling  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Transferrable Surrogates in Expressive Neural Architecture Search Spaces](https://arxiv.org/abs/2504.12971)    
Publication：2025-arXiv  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：RF: Multi-modal (GRAF encoding (structural features) + ZCP scores) Vector-based Fixed length;  LM: Single-modal String  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF + LM  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：With Transferability (cross-task/dataset without fine-tuning)  
***  
Title：[A Surrogate Model With Multiple Comparisons and Semi-Online Learning for Evolutionary Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10935345)  
Publication：IEEE Transactions on Emerging Topics in Computational Intelligence, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Semi-online: updates are triggered after a preset number of new individuals are truly trained; multiple-comparison sampling—if a new architecture can surpass any in the best pool, it is truly trained  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-assisted evolutionary neural architecture search based on smart-block discovery](https://www.sciencedirect.com/science/article/abs/pii/S0957417425008590)  
Publication：Expert Systems with Applications, 2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Regression model with SGD with Momentum  
Model Training：Supervised  
Interaction with Search Algorithm：Online: if prediction is unreliable (below threshold) / a new structure not seen in the benchmark, use it to update the surrogate  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[3D neural architecture search to optimize segmentation of plant parts](https://www.sciencedirect.com/science/article/pii/S27723755250001030)  
Publication：Smart Agricultural Technology,2025  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：-----  
Objective Dimension：Single-objective (mean IoU + latency + memory footprint)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate Modeling for Efficient Evolutionary Multi-Objective Neural Architecture Search in Super Resolution Image Restoration](https://www.scitepress.org/Papers/2024/129490/129490.pdf)  
Publication：Proceedings of the 16th International Joint Conference on Computational Intelligence, 2024  
Source Code：-----  
Architecture Sampling：Latin Hypercube Sampling (LHS)  
Architecture Representation：Single-modal Vector-based Fixed length    
Objective Dimension：Single-objective (Peak Signal-to-Noise Ratio)  
Prediction Type：Absolute performance  
Model Selection：XGBoost  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[An effective surrogate-assisted rank method for evolutionary neural architecture search](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011669)  
Publication：Applied Soft Computing, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance: relation-aware (triplets)  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: first half random, second half top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Pareto-Informed Multi-objective Neural Architecture Search](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_23)    
Publication：International Conference on Parallel Problem Solving from Nature, 2024  
Source Code：https://github.com/SYSU22214881/PiMO-NAS  
Architecture Sampling：Latin Hypercube Sampling (LHS)  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (Error + Params)  
Prediction Type：Absolute performance  
Model Selection：GP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: preference vectors dynamically generate sampling directions + batch sample selection by Expected Hypervolume Improvement (ΔHV)  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Wind power forecasting based on ensemble deep learning with surrogate-assisted evolutionary NAS and many-objective federated learning](https://www.sciencedirect.com/science/article/abs/pii/S036054422402797X)  
Publication：Energy, 2024  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (architecture features + lagged meteorological variables)   
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[One-Shot Surrogate for Evolutionary Multiobjective Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10611773)  
Publication：CEC, 2024  
Source Code：-----  
Architecture Sampling：super-surrogate (Pareto-aware sampling strategy); target surrogate: randomly sample a very small number of real samples  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (error + latency + energy consumption)  
Prediction Type：Absolute performance  
Model Selection：GP with RBF kernel  
Model Training：Supervised  
Interaction with Search Algorithm：Online: UCB acquisition function + non-dominated solutions  
Transferability on Task/Dataset/Search Space：With Transferability: across tasks/datasets/platforms via sub-surrogate sampling + meta-regression  
***  
Title：[Surrogate-Assisted Evolutionary Neural Architecture Search with Isomorphic Training and Prediction](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_16)  
Publication：International Conference on Intelligent Computing, 2024  
Source Code：-----  
Architecture Sampling：Random + isomorphism augmentation  
Architecture Representation：Two for evolution + surrogate: Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Any  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k every T generations  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Similarity surrogate-assisted evolutionary neural architecture search with dual encoding strategy](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf)  
Publication：Electronic Research Archive, 2024  
Source Code：-----  
Architecture Sampling：Random + isomorphism augmentation  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective   
Prediction Type：Relative performance: input triplets, output latent vectors; smaller vector distance ---> more similar performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Towards Full Forward On-Tiny-Device Learning: A Guided Search for a Randomly Initialized Neural Network](https://www.mdpi.com/1999-4893/17/1/22)  
Publication：Algorithms, 2024  
Source Code：https://github.com/andreapisa9/bayesian-elm-search  
Architecture Sampling：-----  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：RF / GP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: Bayesian Optimization with LCB (Lower Confidence Bound) acquisition function  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Maximal sparse convex surrogate-assisted evolutionary convolutional neural architecture search for image segmentation](https://link.springer.com/article/10.1007/s40747-023-01166-5)  
Publication：Complex & Intelligent Systems, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (ranking-aware)  
Model Selection：Maximal Sparse Convex (MSC)  
Model Training：Supervised  
Interaction with Search Algorithm：New individuals deemed promising by MSC are used to update the surrogate  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted Evolutionary Multiobjective Neural Architecture Search Based on Transfer Stacking and Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/10263998)  
Publication：TEVC, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Integer-matrix-based Fixed length  
Objective Dimension：Multi-objective (ACC + Params + FLOPs + latency)  
Prediction Type：Absolute performance  
Model Selection：Kriging / GP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: Bayesian Optimization with Expected Improvement  
Transferability on Task/Dataset/Search Space：With Transferability: across tasks/datasets/search spaces without fine-tuning, auto-adapting during search iterations  
***  
Title：[Efficient multi-objective evolutionary neural architecture search for U-Nets](https://www.sciencedirect.com/science/article/abs/pii/S1568494623008876)  
Publication：Applied Soft Computing, 2023  
Source Code：-----  
Architecture Sampling：Random subnet sampling from supernet  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + Params)  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: K-means cluster sampling  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Designing Convolutional Neural Networks using Surrogate-assisted Genetic Algorithm for Medical Image Classification](https://dl.acm.org/doi/abs/10.1145/3583133.3590678)  
Publication：Proceedings of the Companion Conference on Genetic and Evolutionary Computation, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：XGBoost  
Model Training：Supervised  
Interaction with Search Algorithm：Online: random sample 20%  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Crack detection of continuous casting slab by evolutionary topology backbone search](https://www.sciencedirect.com/science/article/abs/pii/S1568494623006816)  
Publication：Applied Soft Computing, 2023  
Source Code：-----  
Architecture Sampling：All individuals within the first five generations  
Architecture Representation：Single-modal Vector-based Variable length  
Objective Dimension：Single-objective (Average Precision)  
Prediction Type：Absolute performance    
Model Selection：Ensemble of DT + SVR + MLP + Logistic Regression  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-k every five generations  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[A surrogate evolutionary neural architecture search algorithm for graph neural networks](https://www.sciencedirect.com/science/article/abs/pii/S1568494623005033)  
Publication：Applied Soft Computing, 2023  
Source Code：https://github.com/chnyliu/CTFGNAS  
Architecture Sampling：Random   
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：MLP + GBDT + Random Forest (pick the best by Spearman each generation)  
Model Training：Supervised  
Interaction with Search Algorithm：Online: Random  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Gated Recurrent Unit Neural Networks for Wind Power Forecasting based on Surrogate-Assisted Evolutionary Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10166074)  
Publication：IEEE 12th Data Driven Control and Learning Systems Conference (DDCLS), 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Multi-modal (architecture features + lagged meteorological variables) Fixed length  
Objective Dimension：Single-objective   
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Multi-objective Bayesian Optimization for Neural Architecture Search](https://link.springer.com/chapter/10.1007/978-3-031-23492-7_13)  
Publication：International Conference on Artificial Intelligence and Soft Computing, 2022  
Source Code：https://github.com/PetraVidnerova/BayONet  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + Parameters)  
Prediction Type：Absolute performance  
Model Selection：GP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: sampling from Pareto front  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Novel Surrogate Measures Based on a Similarity Network for Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/10058912)  
Publication：IEEE Access, 2023  
Source Code：https://github.com/zekikus/Novel-Surrogate-Measures-based-on-a-Similarity-Network-for-Neural-Architecture-Search  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Non-deep predictor (similarity-network-driven): weighted average / linear regression model  
Model Training：Supervised  
Interaction with Search Algorithm：Online: add edges in similarity network and node types to decide true evaluation  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Human Activity Recognition Based on an Efficient Neural Architecture Search Framework Using Evolutionary Multi-Objective Surrogate-Assisted Algorithms](https://www.mdpi.com/2079-9292/12/1/50)  
Publication：Electronics, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (weighted F1 + FLOPs + Params)  
Prediction Type：Absolute performance  
Model Selection：MLP + CART + RBF + GP (pick the current best each generation)  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-8  
Transferability on Task/Dataset/Search Space：With Transferability: cross-task requires continued online updating  
***  
Title：[Surrogate-assisted evolutionary neural architecture search with network embedding](https://link.springer.com/article/10.1007/s40747-022-00929-w)  
Publication：Complex & Intelligent Systems, 2022  
Source Code：https://github.com/HandingWangXDGroup/SAENAS-NE  
Architecture Sampling：Random  
Architecture Representation：Single-modal formal-grammar string architecture description (converted with graph2vec) -----> Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：Skip-gram model for embeddings (graph2vec), RankNet MLP as surrogate  
Model Training：Unsupervised for embeddings, Supervised for surrogate  
Interaction with Search Algorithm：Online: after nondominated sorting on fitness + uncertainty, take top-k  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted Multiobjective Neural Architecture Search for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/9916102)  
Publication：IEEE Transactions on Artificial Intelligence, 2023  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + Latency)  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: obtain candidate set via NSGA-II; prioritize latency-uniform coverage by Kolmogorov–Smirnov (KS) sampling, then fill up to K by ACC  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Block-Level Surrogate Models for Inference Time Estimation in Hardware-Aware NAS](https://link.springer.com/chapter/10.1007/978-3-031-26419-1_28)  
Publication：Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2022  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + Latency)  
Prediction Type：Absolute performance  
Model Selection：Linear Regression / RF / Boosted Trees / Dense NAS (MLP)  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted Neuroevolution](https://dl.acm.org/doi/abs/10.1145/3512290.3528703)  
Publication：Proceedings of the Genetic and Evolutionary Computation Conference, 2022  
Source Code：-----  
Architecture Sampling：All individuals of each generation until Kendall’s tau reaches a threshold  
Architecture Representation：Formal-grammar string architecture description (Net2Tensor conversion) -----> Tensor-based Variable length * Fixed width  
Objective Dimension：Single-objective  
Prediction Type：Absolute performance  
Model Selection：LSTM  
Model Training：Supervised  
Interaction with Search Algorithm：Online: after Kendall’s tau reaches threshold, active learning, top 25%  
Transferability on Task/Dataset/Search Space：With Transferability: cross-dataset without fine-tuning  
***  
Title：[Bi-fidelity Multi-objective Neural Architecture Search for Adversarial Robustness with Surrogate as a Helper-objective](https://federated-learning.org/fl-ijcai-2022/Papers/FL-IJCAI-22_paper_22.pdf)  
Publication：IJCAI, 2022  
Source Code：-----  
Architecture Sampling：Latin Hypercube Sampling (LHS)  
Architecture Representation：Single-modal Graph-based (GATES encoding) ----> Vector-based Fixed length  
Objective Dimension：Multi-objective (error on clean data + error on adversarial data + composite score)  
Prediction Type：Absolute performance  
Model Selection：GATES (GNN) + MLP / RBF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: every G generations, select solutions with high uncertainty + likely to enter Pareto front  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Pareto Rank Surrogate Model for Hardware-aware Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/9804643)  
Publication：IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2022  
Source Code：https://github.com/IHIaadj/HW-PR-NAS  
Architecture Sampling：Random  
Architecture Representation：Multi-modal Hand-crafted features (FLOPs, Params, depth, input size, #downsamplings, first/last channels) + LSTM sequence encoding (operator sequence) + GCN graph encoding (graph-based representation)  
Objective Dimension：Multi-objective (ACC + Latency)  
Prediction Type：Absolute performance  
Model Selection：GCN (more suitable for ACC) + LSTM (more suitable for Latency) + Regressors (MLP / XGBoost / LightGBM)  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Augmenting Novelty Search with a Surrogate Model to Engineer Meta-diversity in Ensembles of Classifiers](https://link.springer.com/chapter/10.1007/978-3-031-02462-7_27)  
Publication：International Conference on the Applications of Evolutionary Computation (Part of EvoStar), 2022    
Source Code：-----    
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Six distance measures (diversity)  
Prediction Type：Predicts distance between two architectures, not performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Action Command Encoding for Surrogate-Assisted Neural Architecture Search](https://ieeexplore.ieee.org/abstract/document/9521985)  
Publication：IEEE TRANSACTIONS ON COGNITIVE AND DEVELOPMENTAL SYSTEMS, 2022  
Source Code：https://github.com/anonymone/ACE-NAS  
Architecture Sampling：Random  
Architecture Representation：Single-modal ACEncoding sequence (action commands), converted by LSTM to Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：LSTM + RankNet  
Model Training：LSTM (unsupervised), RankNet (supervised)  
Interaction with Search Algorithm：Online: update RankNet every N generations based on ranking   
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Evolving graph convolutional networks for neural architecture search](https://link.springer.com/article/10.1007/s00521-021-05979-8)  
Publication：Neural Computing and Applications, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Graph-based  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware: greater than, less than / approximately equal)  
Model Selection：GCN  
Model Training：Supervised  
Interaction with Search Algorithm：Offline  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[EMONAS-Net: Efficient multiobjective NAS using surrogate-assisted evolutionary algorithm for 3D medical image segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0933365721001470)  
Publication：Artificial Intelligence in Medicine, 2021  
Source Code：-----  
Architecture Sampling：All individuals from the top N generations  
Architecture Representation：Single-modal Vector-based Fixed length   
Objective Dimension：Multi-objective (ESE + Params)  
Prediction Type：Absolute performance  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: solutions selected by PBI + nondomination + minimum error + maximum uncertainty  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Fast Evolutionary Neural Architecture Search Based on Bayesian Surrogate Model](https://ieeexplore.ieee.org/abstract/document/9504999)  
Publication：CEC, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：RF  
Model Training：Supervised  
Interaction with Search Algorithm：Online: top-1  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Enhancing Multi-objective Evolutionary Neural Architecture Search with Surrogate Models and Potential Point-Guided Local Searches](https://link.springer.com/chapter/10.1007/978-3-030-79457-6_39)  
Publication：International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems, 2021  
Source Code：-----  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ERROR + Param (NAS-Bench-101) / FLOPs (NAS-Bench-201))  
Prediction Type：Absolute performance  
Model Selection：MLP  
Model Training：Supervised  
Interaction with Search Algorithm：Online: update every 10 generations (architectures with prediction accuracy above a dynamic threshold in past generations are truly evaluated)  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[Surrogate-Assisted PSO for Evolving Variable-Length Transferable Blocks for Image Classification](https://ieeexplore.ieee.org/abstract/document/9349967)  
Publication：TNNLS, 2020  
Source Code：-----  
Architecture Sampling：Random + downsampling  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Single-objective  
Prediction Type：Relative performance (relation-aware)  
Model Selection：SVM  
Model Training：Supervised  
Interaction with Search Algorithm：Online: when surrogate accuracy exceeds a threshold, if the new solution’s predicted accuracy is better, perform true evaluation  
Transferability on Task/Dataset/Search Space：Without Transferability  
***  
Title：[NSGANetV2: Evolutionary Multi-objective Surrogate-Assisted Neural Architecture Search](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_3)  
Publication：European conference on computer vision, 2020  
Source Code：https://github.com/mikelzc1990/nsganetv2  
Architecture Sampling：Random  
Architecture Representation：Single-modal Vector-based Fixed length  
Objective Dimension：Multi-objective (ACC + #MAdds)  
Prediction Type：Absolute performance  
Model Selection：MLP / CART / RBF / GP (pick the current best each generation)  
Model Training：Supervised  
Interaction with Search Algorithm：Online: among non-dominated candidates, choose those with the highest predicted ACC; along the #MAdds axis, select samples from sparse regions  
Transferability on Task/Dataset/Search Space：Without Transferability  






# 9.16
NAS-Bench-101
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [RATs-NAS](https://ieeexplore.ieee.org/abstract/document/10685480) | 94.41 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [CARL](https://arxiv.org/abs/2506.04001) | 94.23 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 94.23 | ----- | ----- | ----- | ----- | ----- | 410 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 94.27 | ----- | 95.06 | ----- | ----- | ----- | 96 | ----- | ----- | ----- |
| [DIMNP](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036) | 94.23 | ----- | ----- | ----- | ----- | ----- | 423 | ----- | ----- | ----- |
| [HEP-ENAS](https://ieeexplore.ieee.org/abstract/document/10805362) | 94.23 | ----- | ----- | ----- | ----- | ----- | 350 | ----- | ----- | ----- |
| [SET-NAS](https://ieeexplore.ieee.org/abstract/document/10647243) | 93.98 | ----- | ----- | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [PNSS](https://ieeexplore.ieee.org/abstract/document/10651529) | 93.98 | ----- | ----- | ----- | ----- | ----- | 423 | ----- | ----- | ----- |
| [PNSS](https://ieeexplore.ieee.org/abstract/document/10651529) | 94.25 | ----- | ----- | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.14 | ----- | ----- | 0.448 | 10.5 | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.14 | ----- | ----- | 0.391 | 10.6 | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-GBDT](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.06 | ----- | ----- | 0.265 | 10.3 | ----- | 1000 | ----- | ----- | ----- |
| [CAP](https://arxiv.org/abs/2406.02056) | 94.18 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [CLCDLP](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 93.3 | ----- | ----- | 0.0000201 | ----- | ----- | 424 | ----- | ----- | ----- |
| [SPCL](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_18) | 93.94 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [NARQ2T](https://ieeexplore.ieee.org/abstract/document/10155437) | 94.11 | ----- | ----- | 0.00242 | ----- | ----- | 4236 | ----- | ----- | ----- |
| [RFGIAug](https://ieeexplore.ieee.org/abstract/document/10109990) | 94.23 | ----- | ----- | ----- | ----- | ----- | 424 | ----- | ----- | ----- |
| [RFGIAug](https://ieeexplore.ieee.org/abstract/document/10109990) | 94.20 | ----- | ----- | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [DCLP+RL](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 94.14 | ----- | ----- | ----- | ----- | ----- | 300 | ----- | ----- | ----- |
| [DCLP+RS](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 94.17 | ----- | ----- | ----- | ----- | ----- | 300 | ----- | ----- | ----- |
| [CL-fine-tune](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch81) | 94.23 | ----- | ----- | ----- | ----- | ----- | 700 | ----- | ----- | ----- |
| [Towards Leveraging...](https://cke.um.ac.ir/article_42708.html) | 94.21 | ----- | 94.9 | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [RANK-NOSH](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html) | 93.97 | ----- | ----- | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [HAAP](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Homogeneous_Architecture_Augmentation_for_Neural_Predictor_ICCV_2021_paper.html) | 94.09 | ----- | ----- | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [NPENAS-NP](https://ieeexplore.ieee.org/abstract/document/9723446) | 94.14 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [NPENAS-BO](https://ieeexplore.ieee.org/abstract/document/9723446) | 94.1 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [RAGS-NAS](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051) | 94.22 | ----- | ----- | ----- | ----- | ----- | 608 | ----- | ----- | ----- |
| [MT-MSAENAS](https://ieeexplore.ieee.org/abstract/document/11043035) | 94.21  | ----- | ----- | ----- | ----- | ----- | 424 | ----- | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 94.15 | ----- | ----- | 0.00509 | 13.35 | ----- | 734 | ----- | ----- | ----- |
| [HYTES-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0957417425008590) | 92.28 | ----- | 93.02 | 0.00418 | ----- | ----- | ----- | ----- | ----- | ----- |
| [TCMR-ENAS](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011669) | 94.23 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | 4.02 | ----- |
| [ITP-ENAS](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_16) | 94.18 | ----- | ----- | ----- | ----- | ----- | 424 | ----- | ----- | ----- |
| [SSENAS](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf) | 93.83 | ----- | ----- | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [SSENAS](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf) | 93.18 | ----- | ----- | ----- | ----- | ----- | 40 | ----- | ----- | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | 94.08 | ----- | 94.72 | ----- | ----- | ----- | 150 | ----- | ----- | ----- |

NAS-Bench-201
--------------------
**CIFAR-10**
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [RATs-NAS](https://ieeexplore.ieee.org/abstract/document/10685480) | 94.36 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [CARL](https://arxiv.org/abs/2506.04001) | 94.37 | ----- | ----- | ----- | ----- | ----- | 80 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 94.35 | ----- | 91.52 | ----- | ----- | ----- | 110 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 94.37 | ----- | 91.57 | ----- | ----- | ----- | 410 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 94.37 | ----- | 91.61 | ----- | ----- | ----- | 64 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 94.37 | ----- | 91.61 | ----- | ----- | ----- | 128 | ----- | ----- | ----- |
| [DIMNP](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036) | 94.13 | ----- | 91.51 | ----- | ----- | ----- | 781 | ----- | ----- | ----- |
| [HEP-ENAS](https://ieeexplore.ieee.org/abstract/document/10805362) | 94.37 | ----- | 91.61 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [SET-NAS](https://ieeexplore.ieee.org/abstract/document/10647243) | 94.37 | ----- | ----- | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [PNSS](https://ieeexplore.ieee.org/abstract/document/10651529) | 94.30 | ----- | 91.47 | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 93.97 | ----- | 90.80 | 0.00985 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 93.79 | ----- | 91.03 | 0.0259 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.06 | ----- | 90.86 | 0.0562 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.22 | ----- | 91.37 | 0.247 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.10 | ----- | 91.22 | 0.594 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.30 | ----- | 91.44 | 0.847 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.14 | ----- | 91.33 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.32 | ----- | 91.45 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.36 | ----- | 91.57 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.31 | ----- | 91.47 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.12 | ----- | 91.13 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 94.09 | ----- | 91.13 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [CAP](https://arxiv.org/abs/2406.02056) | 94.34 | ----- | 91.54 | 0.000181 | ----- | ----- | 50 | ----- | ----- | ----- |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 86.69 | ----- | ----- | ----- | ----- | 0.00063 | 1000 | ----- | ----- | 77.47 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 88.24 | ----- | ----- | ----- | ----- | 0.00127 | 1000 | ----- | ----- | 173.44 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 91.26 | ----- | ----- | ----- | ----- | 0.00234 | 1000 | ----- | ----- | 374.60 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 90.08 | ----- | ----- | ----- | ----- | 0.0000235 | 10 | ----- | ----- | 318.12 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 91.22 | ----- | ----- | ----- | ----- | 0.000237 | 100 | ----- | ----- | 373.58 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 91.50 | ----- | ----- | ----- | ----- | 0.00118 | 500 | ----- | ----- | 412.77 |
| [SPCL](https://link.springer.com/chapter/10.1007/978-981-99-8388-9_18) | 93.78 | ----- | 91.01 | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [RPNASM](https://www.sciencedirect.com/science/article/abs/pii/S0957417423022443) | 94.33 | ----- | 91.51 | ----- | ----- | 0.46 | 100+2*N | ----- | ----- | ----- |
| [MPENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590513) | ----- | ----- | 91.23 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [Fast-ENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590452) | 93.74 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | ----- | ----- |
| [NARQ2T](https://ieeexplore.ieee.org/abstract/document/10155437) | 94.35 | ----- | 91.48 | 0.00188 | ----- | ----- | 1000 | ----- | ----- | ----- |
| [RFGIAug](https://ieeexplore.ieee.org/abstract/document/10109990) | 94.25 | ----- | 91.43 | ----- | ----- | ----- | 424 | ----- | ----- | ----- |
| [DCLP+RL](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 94.29 | ----- | ----- | ----- | ----- | 0.00029 | 50 | ----- | ----- | ----- |
| [DCLP+RS](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 94.34 | ----- | ----- | ----- | ----- | 0.0001667 | 50 | ----- | ----- | ----- |
| [CL-fine-tune](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch81) | 94.37 | ----- | ----- | ----- | ----- | ----- | 90 | ----- | ----- | ----- |
| [PRE-NAS](https://ieeexplore.ieee.org/abstract/document/9975797) | 94.04 | ----- | 91.37 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSRNAS](https://ieeexplore.ieee.org/abstract/document/9533297) | 93.94 | ----- | 91.23 | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [RANK-NOSH](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html) | 94.26 | ----- | 91.4 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [HAAP](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Homogeneous_Architecture_Augmentation_for_Neural_Predictor_ICCV_2021_paper.html) | 94.00 | ----- | 91.18 | ----- | 0.0694 | ----- | ----- | ----- | ----- | ----- |
| [NPENAS-NP](https://ieeexplore.ieee.org/abstract/document/9723446) | 94.31 | ----- | ----- | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [RAGS-NAS](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051) | 94.37 | ----- | 91.61 | 0.00410 | ----- | ----- | 608 | ----- | ----- | ----- |
| [MT-MSAENAS](https://ieeexplore.ieee.org/abstract/document/11043035) | 94.36 | ----- | 91.50  | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SiamNAS](https://dl.acm.org/doi/abs/10.1145/3712256.3726359) | 94.37 | ----- | ----- | 0.01 | ----- | ----- | 600 | ----- | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 94.33 | ----- | 91.57 | 0.00129 | 0.0849 | ----- | 211 | ----- | ----- | ----- |
| [HYTES-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0957417425008590) | 93.38 | ----- | 90.19 | 0.00845 | ----- | ----- | ----- | ----- | ----- | ----- |
| [TCMR-ENAS](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011669) | 94.15 | ----- | 91.11 | ----- | 0.0694  | ----- | ----- | ----- | ----- | ----- |
| [TCMR-ENAS](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011669) | 94.14 | ----- | 91.29 | ----- | 0.139 | ----- | ----- | ----- | ----- | ----- |
| [ITP-ENAS](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_16) | 94.35 | ----- | 91.59 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSENAS](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf) | 94.37 | ----- | 91.61 | ----- | ----- | ----- | 400 | ----- | ----- | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 94.39 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.64 | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 90.85 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.15 | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | 94.34 | ----- | 91.58 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |

***
**CIFAR-100**
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [RATs-NAS](https://ieeexplore.ieee.org/abstract/document/10685480) | 73.50 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [CARL](https://arxiv.org/abs/2506.04001) | 73.51 | ----- | ----- | ----- | ----- | ----- | 80 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 73.31 | ----- | 73.28 | ----- | ----- | ----- | 110 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 73.35 | ----- | 73.6 | ----- | ----- | ----- | 410 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 73.51 | ----- | 73.49 | ----- | ----- | ----- | 64 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 73.51 | ----- | 73.49 | ----- | ----- | ----- | 128 | ----- | ----- | ----- |
| [DIMNP](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036) | 72.50 | ----- | 72.3 | ----- | ----- | ----- | 781 | ----- | ----- | ----- |
| [HEP-ENAS](https://ieeexplore.ieee.org/abstract/document/10805362) | 73.51 | ----- | 73.49 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [SET-NAS](https://ieeexplore.ieee.org/abstract/document/10647243) | 73.48 | ----- | ----- | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [PNSS](https://ieeexplore.ieee.org/abstract/document/10651529) | 73.13 | ----- | 72.92 | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 71.72 | ----- | 71.39 | 0.00985 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 71.91 | ----- | 71.58 | 0.0259 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.19 | ----- | 72.12 | 0.0562 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.07 | ----- | 71.64 | 0.247 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.06 | ----- | 71.78 | 0.594 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.10 | ----- | 71.71 | 0.847 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.93 | ----- | 72.72 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 73.51 | ----- | 73.49 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 73.51 | ----- | 73.49 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.81 | ----- | 72.50 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 72.87 | ----- | 72.56 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 73.23 | ----- | 72.66 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [CAP](https://arxiv.org/abs/2406.02056) | 73.41 | ----- | 73.41 | 0.000181 | ----- | ----- | 50 | ----- | ----- | ----- |
| [CLCDLP-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 70.1 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 12 |
| [CLCDLP-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 72.4 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 20 |
| [CLCDLP-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 73.5 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 34 |
| [CLCDLP-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 70.1 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 17 |
| [CLCDLP-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 71.6 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 19 |
| [CLCDLP-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 71.8 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 24 |
| [NASFLAT-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 68.53 | ----- | ----- | 0.00034 | ----- | ----- | 20 | ----- | ----- | 14.4 |
| [NASFLAT-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 72.08 | ----- | ----- | 0.00034 | ----- | ----- | 20 | ----- | ----- | 22.2 |
| [NASFLAT-Google Pixel2](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 73.5 | ----- | ----- | 0.00034 | ----- | ----- | 20 | ----- | ----- | 34 |
| [NASFLAT-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 69.92 | ----- | ----- | 0.000178 | ----- | ----- | 20 | ----- | ----- | 17.10 |
| [NASFLAT-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 71.45 | ----- | ----- | 0.000178 | ----- | ----- | 20 | ----- | ----- | 20.47 |
| [NASFLAT-Titan RTX](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 71.9 | ----- | ----- | 0.000178 | ----- | ----- | 20 | ----- | ----- | 26.61 |
| [RPNASM](https://www.sciencedirect.com/science/article/abs/pii/S0957417423022443) | 72.89 | ----- | 73.08 | ----- | ----- | 0.86 | 100+5*N | ----- | ----- | ----- |
| [DGL-Sdm 450](https://ieeexplore.ieee.org/abstract/document/10042973) | 73.2 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 49.5 |
| [DGL-Sdm 450](https://ieeexplore.ieee.org/abstract/document/10042973) | 71.6 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 27.9 |
| [DGL-Sdm 675](https://ieeexplore.ieee.org/abstract/document/10042973) | 72.4 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 41.3 |
| [DGL-Sdm 675](https://ieeexplore.ieee.org/abstract/document/10042973) | 69.7 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 21.3 |
| [DGL-Sdm 855](https://ieeexplore.ieee.org/abstract/document/10042973) | 72.8 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 15.1 |
| [DGL-Sdm 855](https://ieeexplore.ieee.org/abstract/document/10042973) | 71.5 | ----- | ----- | ----- | ----- | ----- | 500 | ----- | ----- | 11.6 |
| [MPENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590513) | ----- | ----- | 72.61 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [Fast-ENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590452) | 72.00 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | ----- | ----- |
| [NARQ2T](https://ieeexplore.ieee.org/abstract/document/10155437) | 73.22 | ----- | 72.73 | 0.00188 | ----- | ----- | 1000 | ----- | ----- | ----- |
| [DCLP+RL](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 72.83 | ----- | ----- | ----- | ----- | 0.00029 | 50 | ----- | ----- | ----- |
| [DCLP+RS](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 73.5 | ----- | ----- | ----- | ----- | 0.0001667 | 50 | ----- | ----- | ----- |
| [PRE-NAS](https://ieeexplore.ieee.org/abstract/document/9975797) | 72.02 | ----- | 71.95 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSRNAS](https://ieeexplore.ieee.org/abstract/document/9533297) | 72.16 | ----- | 71.73 | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [RANK-NOSH](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html) | 73.51 | ----- | 73.49 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [HELP-Google Pixel2](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 67.4 | ----- | ----- | 0.00145 | ----- | ----- | 10 | ----- | ----- | 13 |
| [HELP-Google Pixel2](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 70.6 | ----- | ----- | 0.00145 | ----- | ----- | 10 | ----- | ----- | 19 |
| [HELP-Google Pixel2](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 73.5 | ----- | ----- | 0.00145 | ----- | ----- | 10 | ----- | ----- | 34 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 69.3 | ----- | ----- | 0.00128 | ----- | ----- | 10 | ----- | ----- | 18 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 71.6 | ----- | ----- | 0.00128 | ----- | ----- | 10 | ----- | ----- | 19 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 71.8 | ----- | ----- | 0.00128 | ----- | ----- | 10 | ----- | ----- | 25 |
| [NPENAS-NP](https://ieeexplore.ieee.org/abstract/document/9723446) | 73.46 | ----- | ----- | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [RAGS-NAS](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051) | 73.49 | ----- | 73.51 | 0.00410 | ----- | ----- | 608 | ----- | ----- | ----- |
| [MT-MSAENAS](https://ieeexplore.ieee.org/abstract/document/11043035) | 73.46 | ----- | 73.34 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SiamNAS](https://dl.acm.org/doi/abs/10.1145/3712256.3726359) | 72.97 | ----- | ----- | 0.01 | ----- | ----- | 600 | ----- | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 73.51 | ----- | 73.49 | 0.00129 | 0.0849 | ----- | 211 | ----- | ----- | ----- |
| [HYTES-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0957417425008590) | 70.47 | ----- | 70.39 | 0.00845 | ----- | ----- | ----- | ----- | ----- | ----- |
| [ITP-ENAS](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_16) | 73.36 | ----- | 73.34 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSENAS](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf) | 73.51 | ----- | 73.49 | ----- | ----- | ----- | 400 | ----- | ----- | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 73.84 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.99 | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 66.59 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.14 | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | 73.46 | ----- | 73.46 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |

***
**ImageNet-16**
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [RATs-NAS](https://ieeexplore.ieee.org/abstract/document/10685480) | 47.07 | ----- | ----- | ----- | ----- | ----- | 150 | ----- | ----- | ----- |
| [CARL](https://arxiv.org/abs/2506.04001) | 47.31 | ----- | ----- | ----- | ----- | ----- | 80 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 46.21 | ----- | 45.75 | ----- | ----- | ----- | 110 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 46.52 | ----- | 46.42 | ----- | ----- | ----- | 410 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 46.64 | ----- | 46.38 | ----- | ----- | ----- | 64 | ----- | ----- | ----- |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | 47.31 | ----- | 46.73 | ----- | ----- | ----- | 128 | ----- | ----- | ----- |
| [DIMNP](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036) | 46.35 | ----- | 45.88 | ----- | ----- | ----- | 781 | ----- | ----- | ----- |
| [HEP-ENAS](https://ieeexplore.ieee.org/abstract/document/10805362) | 47.21 | ----- | 46.60 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [SET-NAS](https://ieeexplore.ieee.org/abstract/document/10647243) | 47.25 | ----- | ----- | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [PNSS](https://ieeexplore.ieee.org/abstract/document/10651529) | 46.73 | ----- | 46.23 | ----- | ----- | ----- | 1000 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 45.61 | ----- | 45.05 | 0.00985 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 45.58 | ----- | 46.24 | 0.0259 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-SVR](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.25 | ----- | 46.47 | 0.0562 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.74 | ----- | 46.02 | 0.247 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.46 | ----- | 45.80 | 0.594 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-KNNS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.40 | ----- | 45.65 | 0.847 | ----- | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.50 | ----- | 46.11 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.38 | ----- | 45.97 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-RF](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.62 | ----- | 46.40 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.54 | ----- | 45.89 | 0.115 | ----- | ----- | 100 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.45 | ----- | 45.75 | 0.146 | ----- | ----- | 300 | ----- | ----- | ----- |
| [SAPMNAS-XGB](https://www.sciencedirect.com/science/article/abs/pii/S0925231224012438) | 46.41 | ----- | 45.42 | 0.195 | 20 | ----- | 500 | ----- | ----- | ----- |
| [CAP](https://arxiv.org/abs/2406.02056) | 46.44 | ----- | 46.47 | 0.000181 | ----- | ----- | 50 | ----- | ----- | ----- |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 31.16 | ----- | ----- | ----- | ----- | 0.000792 | 1000 | ----- | ----- | 45.15 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 32.72 | ----- | ----- | ----- | ----- | 0.00119 | 1000 | ----- | ----- | 75.38 |
| [LC-NAS](https://www.mdpi.com/2079-9292/13/4/692) | 39.31 | ----- | ----- | ----- | ----- | 0.00184 | 1000 | ----- | ----- | 111.55 |
| [RPNASM](https://www.sciencedirect.com/science/article/abs/pii/S0957417423022443) | 46.44 | ----- | 46.34 | ----- | ----- | 2.31 | 100+5*N | ----- | ----- | ----- |
| [MPENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590513) | ----- | ----- | 46.44 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [Fast-ENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590452) | 45.74 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | ----- | ----- |
| [NARQ2T](https://ieeexplore.ieee.org/abstract/document/10155437) | 46.93 | ----- | 46.24 | 0.00188 | ----- | ----- | 1000 | ----- | ----- | ----- |
| [DCLP+RL](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 46.49 | ----- | ----- | ----- | ----- | 0.00029 | 50 | ----- | ----- | ----- |
| [DCLP+RS](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 46.54 | ----- | ----- | ----- | ----- | 0.0001667 | 50 | ----- | ----- | ----- |
| [PRE-NAS](https://ieeexplore.ieee.org/abstract/document/9975797) | 45.34 | ----- | 45.16 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSRNAS](https://ieeexplore.ieee.org/abstract/document/9533297) | 45.65 | ----- | 45.46 | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [RANK-NOSH](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html) | 46.34 | ----- | 46.37 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [NPENAS-NP](https://ieeexplore.ieee.org/abstract/document/9723446) | 46.48 | ----- | ----- | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [RAGS-NAS](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051) | 46.61 | ----- | 46.64 | 0.00410 | ----- | ----- | 608 | ----- | ----- | ----- |
| [MT-MSAENAS](https://ieeexplore.ieee.org/abstract/document/11043035) | 46.68 | ----- | 46.53 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SiamNAS](https://dl.acm.org/doi/abs/10.1145/3712256.3726359) | 46.37 | ----- | ----- | 0.01 | ----- | ----- | 600 | ----- | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 46.71 | ----- | 46.52 | 0.00129 | 0.0849 | ----- | 211 | ----- | ----- | ----- |
| [HYTES-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0957417425008590) | 43.44 | ----- | 42.95 | 0.00845 | ----- | ----- | ----- | ----- | ----- | ----- |
| [ITP-ENAS](https://link.springer.com/chapter/10.1007/978-981-97-5581-3_16) | 46.40 | ----- | 46.10 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SSENAS](https://www.aimspress.com/aimspress-data/era/2024/2/PDF/era-32-02-050.pdf) | 46.24 | ----- | 46.72 | ----- | ----- | ----- | 400 | ----- | ----- | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 47.19 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.82 | ----- |
| [T<sup>2</sup>MONAS](https://ieeexplore.ieee.org/abstract/document/10263998) | 40.08 | ----- | ----- | ----- | ----- | ----- | 55 | ----- | 0.21 | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | 46.36 | ----- | 46.59 | ----- | ----- | ----- | 100 | ----- | ----- | ----- |

***
NAS-Bench-301
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [WeakPNAG](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_17) | ----- | ----- | 95.01 | ----- | ----- | ----- | 96 | ----- | ----- | ----- |
| [CL-fine-tune](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch81) | ----- | ----- | 94.83 | ----- | ----- | ----- | 800 | ----- | ----- | ----- |
| [RAGS-NAS](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.8051) | ----- | ----- | 94.89 | ----- | ----- | ----- | 300 | ----- | ----- | ----- |
| [TCMR-ENAS](https://www.sciencedirect.com/science/article/abs/pii/S1568494624011669) | 94.47 | ----- | 94.97 | ----- | ----- | ----- | 200 | ----- | ----- | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | ----- | ----- | 95.01 | ----- | ----- | ----- | 300 | ----- | ----- | ----- |

***

DARTS 
--------------------
**ImageNet**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CARL](https://arxiv.org/abs/2506.04001) | 76.1 | 92.8 | ----- | ----- | ----- | 0.25 | 100 | ----- | 5.3 | ----- |
| [SDGP](https://ieeexplore.ieee.org/abstract/document/10438213) | 76.2 | 92.8 | ----- | 0.01 | ----- | ----- | 400(NB101-C10) | 546 | 5.41 | ----- |
| [Fast-ENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590452) | 75.7 | 92.5 | ----- | ----- | ----- | 0.026  | 1000 | ----- | 5.9 | ----- |
| [RFGIAug](https://ieeexplore.ieee.org/abstract/document/10109990) | 73.4 | ----- | ----- | ----- | 1.5 | ----- | 100 | ----- | 4.8 | ----- |
| [PRE-NAS](https://ieeexplore.ieee.org/abstract/document/9975797) | 76 | 92.2 | ----- | ----- | ----- | 0.6 | 564 | ----- | 6.2 | ----- |
| [WPNAS-A](https://arxiv.org/abs/2203.02086) | 76.22 | 92.70 | ----- | ----- | ----- | 1.5 | ----- | 550 | 5.03 | ----- |
| [WPNAS-B](https://arxiv.org/abs/2203.02086) | 76.61 | 92.98 | ----- | ----- | ----- | 2.0 | ----- | 848 | 7.56 | ----- |

***
**CIFAR-10**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CARL](https://arxiv.org/abs/2506.04001) | 97.67 | ----- | ----- | ----- | ----- | 0.25 | 100 | ----- | 3.7 | ----- |
| [SR-NAS](https://arxiv.org/abs/2505.15832) | 97.34 | ----- | ----- | 0.01 | ----- | ----- | 100 | ----- | 3.9 | ----- |
| [GMAE-NAS (AE)](https://www.ijcai.org/proceedings/2022/0432.pdf)  | 97.44 | ----- | ----- | ----- | ----- | 3.3 | 100 | ----- | 4.0 | ----- |
| [GMAE-NAS (BO)](https://www.ijcai.org/proceedings/2022/0432.pdf)  | 97.50 | ----- | ----- | ----- | ----- | 3.3 | 100 | ----- | 3.8 | ----- |
| [DIMNP](https://www.sciencedirect.com/science/article/abs/pii/S1566253524007036) | 97.55 | ----- | ----- | ----- | ----- | 0.1 | 100 | ----- | 3.5 | ----- |
| [SDGP](https://ieeexplore.ieee.org/abstract/document/10438213) | 97.6 | ----- | ----- | 0.01 | ----- | ----- | 400(NB101-C10) | ----- | 3.4 | ----- |
| [CAP](https://arxiv.org/abs/2406.02056) | 97.58 | ----- | ----- | ----- | ----- | 3.3 | 100 | ----- | 3.3 | ----- |
| [GNET](https://proceedings.mlr.press/v238/xiang24a.html) | 97.61 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [PINAT](https://ojs.aaai.org/index.php/AAAI/article/view/26076) | 97.58 | ----- | ----- | ----- | ----- | 0.3  | 1000 | ----- | 3.6 | ----- |
| [Fast-ENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590452) | 97.50 | ----- | ----- | ----- | ----- | 0.02  | 1000 | ----- | 4.2 | ----- |
| [DCLP+RL](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 97.5 | ----- | ----- | 0.00333 | 0.1667  | ----- | ----- | ----- | ----- | ----- |
| [DCLP+RS](https://ojs.aaai.org/index.php/AAAI/article/view/29649) | 97.52 | ----- | ----- | 0.00208 | 0.1667  | ----- | ----- | ----- | ----- | ----- |
| [PRE-NAS](https://ieeexplore.ieee.org/abstract/document/9975797) | 97.51 | ----- | ----- | ----- | ----- | 0.6 | 564 | ----- | 4.5 | ----- |
| [WPNAS-A](https://arxiv.org/abs/2203.02086) | 97.45 | ----- | ----- | ----- | ----- | 1.5 | ----- | ----- | 2.4 | ----- |
| [WPNAS-B](https://arxiv.org/abs/2203.02086) | 97.70 | ----- | ----- | ----- | ----- | 2.0 | ----- | ----- | 4.8 | ----- |
| [SSRNAS](https://ieeexplore.ieee.org/abstract/document/9533297) | 97.53 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 3.8 | ----- |
| [RANK-NOSH](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_RANK-NOSH_Efficient_Predictor-Based_Architecture_Search_via_Non-Uniform_Successive_Halving_ICCV_2021_paper.html) | 97.47 | ----- | ----- | ----- | ----- | ----- | 50 | ----- | 3.5 | ----- |
| [NPENAS-BO](https://ieeexplore.ieee.org/abstract/document/9723446) | 97.36 | ----- | ----- | ----- | ----- | 2.5 | 150 | ----- | 4.0 | ----- |
| [NPENAS-NP](https://ieeexplore.ieee.org/abstract/document/9723446) | 97.46 | ----- | ----- | ----- | ----- | 1.8 | 100 | ----- | 3.5 | ----- |
| [SAENAS-NE](https://link.springer.com/article/10.1007/s40747-022-00929-w) | 97.48 | ----- | ----- | ----- | ----- | 9 | 100 | ----- | 2.9 | ----- |
| [MORAS-SHNet-M](https://federated-learning.org/fl-ijcai-2022/Papers/FL-IJCAI-22_paper_22.pdf) | 85.8 | ----- | ----- | ----- | ----- | 3 | G/2 + 200 | 1634 | 5.22 | ----- |
| [MORAS-SHNet-M](https://federated-learning.org/fl-ijcai-2022/Papers/FL-IJCAI-22_paper_22.pdf) | 86.0 | ----- | ----- | ----- | ----- | 3 | G/2 + 200 | 1525 | 5.60 | ----- |
| [Augmenting Novelty...](https://link.springer.com/chapter/10.1007/978-3-031-02462-7_27) | 83.885 | ----- | ----- | 0.0564 | 0.2096  | ----- |3200 | ----- | ----- | ----- |




 TransNAS-Bench-101 Micro
--------------------
| 算法 | Cls. Obj. (%) | Cls. Scene (%) | Auto. SSIM (10^-3) | Normal. SSIM (10^-3) | Sem. Seg. mIoU (%) | Room. L2 loss (10^-3) | Jigsaw (%) | Quary |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CARL](https://arxiv.org/abs/2506.04001) | 45.69 | 54.78 | 57.01 | 57.77 | 25.80 | 60.92 | 94.84 | 50 |
| [MPENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590513) | 45.24 | 54.77 | 56.82 | 58.14 | 25.38 | 60.69 | 94.93 | 200 |
| [Arch-Graph](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Arch-Graph_Acyclic_Architecture_Relation_Predictor_for_Task-Transferable_Neural_Architecture_Search_CVPR_2022_paper.html) | 45.81 | 54.90 | 56.58 | 58.27 | 25.69 | 60.08 | ----- | 80 |


 TransNAS-Bench-101 Macro
--------------------
| 算法 | Cls. Obj. (%) | Cls. Scene (%) | Auto. SSIM (10^-3) | Normal. SSIM (10^-3) | Sem. Seg. mIoU (%) | Room. L2 loss (10^-3) | Jigsaw (%) | Quary |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CARL](https://arxiv.org/abs/2506.04001) | 47.45 | 56.92 | 74.12 | 62.44 | 29.22 | 57.68 | 96.89 | 50 |
| [MPENAS](https://dl.acm.org/doi/abs/10.1145/3583131.3590513) | 47.23 | 56.49 | 73.26 | 61.54 | 28.70 | 58.74 | 96.67 | 200 |
| [Arch-Graph](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Arch-Graph_Acyclic_Architecture_Relation_Predictor_for_Task-Transferable_Neural_Architecture_Search_CVPR_2022_paper.html) | 47.44 | 56.98 | 75.90 | 64.35 | 29.19 | 57.75 | ----- | 80 |

NAS-Bench-NLP
--------------------
| 算法 | Log PPL | Quary |
| :----: | :----: | :----: |
| [CARL](https://arxiv.org/abs/2506.04001) | 4.572 | 100 |

OFA  
--------------------
**CIFAR-10**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 98.20 | ----- | ----- | 1.4 | ----- | ----- | 400 | ----- | 6.33 | ----- |
***
**CIFAR-100**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 87.26 | ----- | ----- | 1.6 | ----- | ----- | 400 | ----- | 6.74 | ----- |
***
**ImageNet**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [SR-NAS](https://arxiv.org/abs/2505.15832) | 76.65 | ----- | ----- | ----- | ----- | ----- | 100 | ----- | ----- | ----- |
| [SPNAS](https://ieeexplore.ieee.org/abstract/document/10841460) | 78.62 | 94.07 | ----- | 0.37 | ----- | ----- | 400 | ----- | 6.64 | ----- |
| [CL-fine-tune](https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch81) | 79.2 | ----- | ----- | ----- | ----- | ----- | 50 fine-tune | ----- | ----- | ----- |
| [PiMO-NAS-O-T](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_23) | 75.2 | ----- | ----- | ----- | ----- | 0.81 | 260 | 153 | 5.6 | ----- |
| [PiMO-NAS-O-T](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_23) | 77.7 | ----- | ----- | ----- | ----- | 0.81 | 260 | 362 | 6.1 | ----- |
| [PiMO-NAS-O-T](https://link.springer.com/chapter/10.1007/978-3-031-70071-2_23) | 80.2 | ----- | ----- | ----- | ----- | 0.81 | 260 | 576 | 7.44 | ----- |

**ImageNet-1k**

--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CLCDLP](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 77.0 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 25.2 |
| [CLCDLP](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 77.8 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 27.4 |
| [CLCDLP](https://www.sciencedirect.com/science/article/abs/pii/S0020025524006972) | 78.0 | ----- | ----- | ----- | ----- | ----- | 900 | ----- | ----- | 27.6 |
| [DGL-Google Pixel3A](https://ieeexplore.ieee.org/abstract/document/10042973) | 77.6 | 93.7 | ----- | ----- | ----- | ----- | 500 | ----- | 6.8 | 105 |
| [DGL-Google Pixel3A](https://ieeexplore.ieee.org/abstract/document/10042973) | 77.8 | 93.9 | ----- | ----- | ----- | ----- | 500 | ----- | 7.5 | 114 |
| [DGL-Google Pixel3A](https://ieeexplore.ieee.org/abstract/document/10042973) | 78.3 | 94.1 | ----- | ----- | ----- | ----- | 500 | ----- | 8.7 | 121 |
| [DGL-Samsung Galaxy Note10](https://ieeexplore.ieee.org/abstract/document/10042973) | 77.7 | 93.8 | ----- | ----- | ----- | ----- | 500 | ----- | 7.8 | 30 |
| [DGL-Samsung Galaxy Note10](https://ieeexplore.ieee.org/abstract/document/10042973) | 77.9 | 94.0 | ----- | ----- | ----- | ----- | 500 | ----- | 7.5 | 41 |
| [DGL-Samsung Galaxy Note10](https://ieeexplore.ieee.org/abstract/document/10042973) | 78.2 | 94.1 | ----- | ----- | ----- | ----- | 500 | ----- | 7.7 | 50 |
| [DGL-Huawei Enjoy 20 pro](https://ieeexplore.ieee.org/abstract/document/10042973) | 77.5 | 93.9 | ----- | ----- | ----- | ----- | 500 | ----- | 7.8 | 45 |
| [DGL-Huawei Enjoy 20 pro](https://ieeexplore.ieee.org/abstract/document/10042973) | 78.0 | 94.0 | ----- | ----- | ----- | ----- | 500 | ----- | 8.5 | 53 |
| [DGL-Huawei Enjoy 20 pro](https://ieeexplore.ieee.org/abstract/document/10042973) | 78.3 | 94.1 | ----- | ----- | ----- | ----- | 500 | ----- | 8.1 | 57 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 76.0 | ----- | ----- | 0.0002917 | ----- | ----- | 10 | ----- | ----- | 20.3 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 23.1 | ----- | ----- | 0.0002917 | ----- | ----- | 10 | ----- | ----- | 23.1 |
| [HELP-Titan RTX](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 77.9 | ----- | ----- | 0.0002917 | ----- | ----- | 10 | ----- | ----- | 28.6 |
| [HELP-Intel Xeon Gold 6226](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 77.6 | ----- | ----- | 0.00333 | ----- | ----- | 20 | ----- | ----- | 147 |
| [HELP-Intel Xeon Gold 6226](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 78.1 | ----- | ----- | 0.00333 | ----- | ----- | 20 | ----- | ----- | 171 |
| [HELP-Intel Xeon Gold 6226](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 75.9 | ----- | ----- | 0.00125 | ----- | ----- | 10 | ----- | ----- | 67.4 |
| [HELP-Intel Xeon Gold 6226](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html) | 76.7 | ----- | ----- | 0.00125 | ----- | ----- | 10 | ----- | ----- | 76.4 |
***

ENAS
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [HOP](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Not_All_Operations_Contribute_Equally_Hierarchical_Operation-Adaptive_Predictor_for_Neural_ICCV_2021_paper.html) | 97.48 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | 3.9 | ----- |
| [GATES](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_12) | 97.42 | ----- | ----- | ----- | ----- | ----- | 600 | ----- | 4.1 | ----- |

CIFAR-10
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CIMNet](https://ieeexplore.ieee.org/abstract/document/10551739) | 94.7 | 99.8 | ----- | ----- | ----- | ----- | 3200 | 188 | 8.9 | 98.6 |
| [A holistic approach...](https://www.sciencedirect.com/science/article/pii/S1568494625001437) | ----- | ----- | 75 | ----- | ----- | 23 | 55291 | ----- | ----- | ----- |
| [MPE-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224004351) | 96.53 | ----- | ----- | ----- | ----- | 0.78 | ----- | ----- | 6.4 | ----- |
| [grad_norm](https://ieeexplore.ieee.org/abstract/document/10191474) | 88.1 | ----- | ----- | ----- | ----- | ----- | ----- | 45.60 | 0.667 | ----- |
| [snip](https://ieeexplore.ieee.org/abstract/document/10191474) | 84.6 | ----- | ----- | ----- | ----- | ----- | ----- | 24.24 | 0.114 | ----- |
| [grasp](https://ieeexplore.ieee.org/abstract/document/10191474) | 84.1 | ----- | ----- | ----- | ----- | ----- | ----- | 33.50 | 0.487 | ----- |
| [synflow](hhttps://ieeexplore.ieee.org/abstract/document/10191474) | 88.1 | ----- | ----- | ----- | ----- | ----- | ----- | 43.09 | 0.344 | ----- |
| [fisher](https://ieeexplore.ieee.org/abstract/document/10191474) | 82.7 | ----- | ----- | ----- | ----- | ----- | ----- | 31.05 | 0.456 | ----- |
| [jacov](https://ieeexplore.ieee.org/abstract/document/10191474) | 90.21 | ----- | ----- | ----- | ----- | ----- | ----- | 21.96 | 0.066 | ----- |
| [logdet](https://ieeexplore.ieee.org/abstract/document/10191474) | 90.32 | ----- | ----- | ----- | ----- | ----- | ----- | 50.00 | 0.081 | ----- |
| [Automated design...](https://www.sciencedirect.com/science/article/abs/pii/S092523122200340X) | 95.78 | ----- | ----- | ----- | ----- | 0.66 | 120 | ----- | 2.8 | ----- |
| [CURIOUS-20](https://ieeexplore.ieee.org/abstract/document/9698855) | 97.20 | ----- | ----- | ----- | ----- | 2.5 | 120 | ----- | 26.1 | ----- |
| [CURIOUS-20](https://ieeexplore.ieee.org/abstract/document/9698855) | 97.05 | ----- | ----- | ----- | ----- | 2.5 | 120 | ----- | 4.7 | ----- |
| [CURIOUS-40](https://ieeexplore.ieee.org/abstract/document/9698855) | 97.31 | ----- | ----- | ----- | ----- | 2.5 | 120 | ----- | 47.1 | ----- |
| [POPNAS](https://dl.acm.org/doi/abs/10.1145/3449726.3463146) | ----- | ----- | 74.33 | ----- | ----- | 0.0264 | 1280 | ----- | ----- | ----- |
| [E2EPP](https://dl.acm.org/doi/abs/10.1145/3449726.3463146) | 94.7 | ----- | ----- | ----- | ----- | 8.5 | ----- | ----- | ----- | ----- |
| [TAP](https://ojs.aaai.org/index.php/AAAI/article/view/4282) | 93.67 | ----- | ----- | 0.00463 | ----- | ----- | 800 | ----- | ----- | ----- |
| [SMA-GA-NP](https://www.sciencedirect.com/science/article/abs/pii/S2210650225001415) | 94.28 | ----- | ----- | ----- | ----- | 0.38 | ----- | 0.67 | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 97.12 | ----- | ----- | ----- | ----- | 1.32 | ----- | ----- | 3.46 | ----- |
| [Designing Convolutional...](https://dl.acm.org/doi/abs/10.1145/3583133.3590678) | 96.66 | ----- | ----- | ----- | ----- | 0.8 | 100+0.2*N | ----- | 0.67 | ----- |
| [MO-BayONet](https://link.springer.com/chapter/10.1007/978-3-031-23492-7_13) | 74.29 | ----- | ----- | ----- | ----- | ----- |N<sub>init</sub> + 100 | ----- | 0.081 | ----- |
| [MO-BayONet](https://link.springer.com/chapter/10.1007/978-3-031-23492-7_13) | 76.45 | ----- | ----- | ----- | ----- | ----- |N<sub>init</sub> + 100 | ----- | 0.196 | ----- |
| [MO-BayONet](https://link.springer.com/chapter/10.1007/978-3-031-23492-7_13) | 76.46 | ----- | ----- | ----- | ----- | ----- |N<sub>init</sub> + 100 | ----- | 0.455 | ----- |
| [ACEncoding+Seq2Rank](https://ieeexplore.ieee.org/abstract/document/9521985) | 97.1 | ----- | ----- | ----- | ----- | 1.5 | ----- | ----- | 2.4 | ----- |
| [ACEncoding+Seq2Rank](https://ieeexplore.ieee.org/abstract/document/9521985) | 97.68 | ----- | ----- | ----- | ----- | 1.5 | ----- | ----- | 9.8 | ----- |
| [FENAS](https://ieeexplore.ieee.org/abstract/document/9504999) | 96.16 | ----- | ----- | ----- | ----- | 0.16 | ----- | ----- | 16.5 | ----- |
| [EffPnet](https://ieeexplore.ieee.org/abstract/document/9349967) | 96.424 | ----- | ----- | ----- | ----- | <3 | ----- | ----- | 2.68 | ----- |
| [MSuNAS](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_3) | 98.4 | ----- | ----- | ----- | ----- | ----- | 350 | ----- | ----- | ----- |

CIFAR-100
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CIMNet](https://ieeexplore.ieee.org/abstract/document/10551739) | 74.4 | 95.7 | ----- | ----- | ----- | ----- | 3200 | 101 | 8.5 | 152.3 |
| [MPE-NAS](https://www.sciencedirect.com/science/article/abs/pii/S0925231224004351) | 80.41 | ----- | ----- | ----- | ----- | 0.81 | ----- | ----- | 6.6 | ----- |
| [grad_norm](https://ieeexplore.ieee.org/abstract/document/10191474) | 63.9 | ----- | ----- | ----- | ----- | ----- | ----- | 31.42 | 0.469 | ----- |
| [snip](https://ieeexplore.ieee.org/abstract/document/10191474) | 55.41 | ----- | ----- | ----- | ----- | ----- | ----- | 40.02 | 0.575 | ----- |
| [grasp](https://ieeexplore.ieee.org/abstract/document/10191474) | 62.32 | ----- | ----- | ----- | ----- | ----- | ----- | 30.93 | 0.468 | ----- |
| [synflow](https://ieeexplore.ieee.org/abstract/document/10191474) | 69.36 | ----- | ----- | ----- | ----- | ----- | ----- | 46.41 | 0.497 | ----- |
| [fisher](https://ieeexplore.ieee.org/abstract/document/10191474) | 53.29 | ----- | ----- | ----- | ----- | ----- | ----- | 35.79 | 0.517 | ----- |
| [jacov](https://ieeexplore.ieee.org/abstract/document/10191474) | 27.57 | ----- | ----- | ----- | ----- | ----- | ----- | 34.71 | 0.037 | ----- |
| [logdet](https://ieeexplore.ieee.org/abstract/document/10191474) | 43.85 | ----- | ----- | ----- | ----- | ----- | ----- | 49.70 | 0.058 | ----- |
| [Automated design...](https://www.sciencedirect.com/science/article/abs/pii/S092523122200340X) | 77.02 | ----- | ----- | ----- | ----- | 0.92 | 120 | ----- | 3.4 | ----- |
| [E2EPP](https://dl.acm.org/doi/abs/10.1145/3449726.3463146) | 77.98 | ----- | ----- | ----- | ----- | 8.5 | ----- | ----- | ----- | ----- |
| [TAP](https://ojs.aaai.org/index.php/AAAI/article/view/4282) | 81.01 | ----- | ----- | 0.00463 | ----- | ----- | 800 | ----- | ----- | ----- |
| [SMA-GA-NP](https://www.sciencedirect.com/science/article/abs/pii/S2210650225001415) | 74.86 | ----- | ----- | ----- | ----- | 0.38 | ----- | 0.67 | ----- | ----- |
| [SMCSO](https://ieeexplore.ieee.org/abstract/document/10935345) | 80.65 | ----- | ----- | ----- | ----- | 2 | ----- | ----- | 3.72 | ----- |
| [FENAS](https://ieeexplore.ieee.org/abstract/document/9504999) | 79.43 | ----- | ----- | ----- | ----- | 0.16 | ----- | ----- | 16.5 | ----- |

IMAGENET
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [LRENAS](https://www.sciencedirect.com/science/article/abs/pii/S2210650225001142) | 79.2 | 94.6 | ----- | ----- | ----- | 0.7 | 300 | ----- | 6.1 | ----- |
| [MSuNAS](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_3) | 75.9 | ----- | ----- | ----- | ----- | ----- | 350 | ----- | ----- | ----- |


IMAGENET-1K
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [SparseNAS](https://www.sciencedirect.com/science/article/pii/S0950705124012541) | 76.7 | ----- | ----- | 1.06 | ----- | 13.1 | 1000 | ----- | ----- | ----- |
| [FBNetV3-A](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html) | 79.1 | 94.5 | ----- | ----- | ----- | 445.83 | 240 | 357 | ----- | ----- |
| [FBNetV3-C](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html) | 80.5 | 95.1 | ----- | ----- | ----- | 445.83 | 240 | 557 | ----- | ----- |
| [FBNetV3-E](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html) | 81.3 | 95.5 | ----- | ----- | ----- | 445.83 | 240 | 762 | ----- | ----- |
| [FBNetV3-G](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.html) | 82.8 | 96.3 | ----- | ----- | ----- | 445.83 | 240 | 2100 | ----- | ----- |

Tiny ImageNet
--------------------
| 算法 | Top-1 Test Accuracy (%) | Top-5 Test Accuracy (%) | Valid ACC (%) | Search Cost A (GPU days) | Search Cost B (GPU days) | Search Cost C (GPU days) | Quary | FLOPs (M) | Params (M) | inference latency (ms) |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [CIMNet](https://ieeexplore.ieee.org/abstract/document/10551739) | 64.6 | 85.4 | ----- | ----- | ----- | ----- | 3200 | 158 | 8.9 | 97.9 |

***







