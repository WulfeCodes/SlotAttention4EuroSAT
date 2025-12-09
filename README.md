# SlotAttention4EuroSAT
This is grid search code for a MobileViT backbone + slot-attention classification algorithm that achieved 98.54% test accuracy compared to current SoTA publication results. with a 36% parameter reduction compared to current SOTA performance: Automated classification of remote sensing satellite images using deep learning based vision transformer 
https://link.springer.com/article/10.1007/s10489-024-05818-y

OPTIMIZED ATTENTION MECHANISM

i. Motivation
    While transformers are undeniably a key standing point for leveraging large data for regression tasks, there are many flaws which are key research topics amongst the deep learning community. One of which is the ‘rank collapse’, this occurs when query, key, value projection matrices share much of the same domain, thus constraining the information transfer and limiting the exploration of weight combinations. This often occurs due to the input signal homogeneity into the embedding dimension. In a typical stand-alone encoder gpt the same input is given into the q,k,v matrices, which can be redundant depending on the weight configuration. As such, various training experiments were performed to improve the capacity of the dot product operations, particularly ‘slot-attention’ in its initial development succeeded at creating explainable attention heat maps along with a unique capacity for unsupervised object detection. Though it was not until the implementation of SCOUTER[5] that it was directly applied to classification tasks. 

ii. Slot-attention algorithm

  Slot-attention replaces the query input as n m shaped vectors, these vectors are often multiplied by a patch embedding from some image or previous image encoder. For our encoder and vit base mobilevit-v2 [6] was chosen for it’s o(k) run-time greatly outperforming global multi-head attention which typically operated in o(k2) in this case mobilevit’s 075 checkpoint from the timm library was used as our primary layer. The last backbone is then extracted to serve as inputs into the key and query embeddings. Slot-attention operates differently than normal attention in a few regimes, for one the normalization is not about the key’s column elements. It instead normalizes across each n row column vector. Each slot attention vector thus assigns some importance to each region of the patch embedding, creating a specialization according to the type of objects regressed upon. Once the attention operation is computed it is then projected into a n class,1 row vector. Cross entropy is then performed to evaluate against the one-hot encoding of the given distribution. This loss is then summed with the attention loss of featured in the primary slot-attention loss, this helps to keep gradients normal to ensure explainable attention heat maps when analyzing a given query.

iii. Experiments

   To conduct the evaluation, many combinations of hyper-parameters were chosen including lamda (the scaling of the attention loss), the dropout of the vit, the number of slot-attention vectors per class, the amount of iterations of slot-attention, and the hidden dimension of the slot-attention component. Some problems do arise due to these tunings, in regard to the attention loss’ objective of minimizing the attention value many of the gradients tend to vanish leading to slower no learning behaviors at this component. In addition, the initialization values of these slot-attention vectors greatly impact the learning capacity of the model, this concept is expanded upon and improved in latter works such as e-scouter [7] for few-shot inference. E-scouter pre-initializes these slot-attention vectors with distributions about the input patch leading to increased robustness across class numbers and different datasets. 

i.	Results

a.	Experimental configuration

The performance of the optimized attention mechanism was evaluated using the mobilevitv2_075 backbone. The specific slot-attention configuration had: Lambda: 0.05, Dropout: 0.1, Slots Per Class: 1, Slot Iterations: 5, Stage 1 Epoch: 55, LR: 4e-4, Stage 2 Epoch: 45, LR: 1e-4 MobileViT parameter Count: ~2.2Million, Slot Attention parameter Count ~500k

b.	Quantitative analysis

The training procedure was divided into two distinct stages to ensure stable feature alignment before full-network optimization.

1. stage 1: frozen backbone

In the initial phase, the mobilevit backbone weights were frozen, training only the slot attention module. This stage lasted for 55 epochs. Despite the backbone being non-trainable, the slot-attention mechanism successfully extracted relevant features, achieving a peak validation accuracy of 90.96%. This confirms the capacity of the slot-attention vectors to learn significant representations even when constrained by a static feature extractor.

2. stage 2: fine-tuning

Following the initialization of the attention slots, the entire network (including the backbone) was unfrozen for end-to-end fine-tuning. The model reached over 45 epochs for a total of 100 epochs, stabilizing with a best validation accuracy of 98.67%.

c.	 Final Performance

The final test set evaluation yielding an accuracy of 98.54%. These results validate that a lightweight backbone combined with a properly regularized (lambda=0.05) slot-attention head can achieve high-fidelity classification performance while maintaining computational efficiency.

Metric	 (MobileViT v2 + Slot Attn)	Proposed Paper Model (ViT + LSA)
Best Accuracy	98.67% (Val) / 98.54% (Test)	98.5% 
Total Parameters	~2.66 Million (2,664,049)	~4.17 Million (4,166,151) 
Architecture Base	Hybrid (ViT Backbone + Slot Attention)	Pure ViT (Vision Transformer) 3
Attention Mechanism	Slot Attention (Iterative, Grouping)	Local Self-Attention (LSA) (Window-based) 44
