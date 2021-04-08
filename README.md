## MATE: Multimodal Dialogue Systems via Capturing Context-aware Dependencies of Semantic Elements
<img src="images/pytorch.jpeg" width="30%"> 
<img src="images/ustc.jpeg" width="30%">
<img src="images/huawei.jpeg" width="30%">

This is the PyTorch implementation of the paper:
**Multimodal Dialogue Systems via Capturing Context-aware Dependencies of Semantic Elements**. Weidong He, Zhi Li, Dongcai Lu, Enhong Chen, Tong Xu, Baoxing  Huai, Jing Nicholas Yuan. ***ACM MM 2020***. 
[[PDF]](https://dl.acm.org/doi/abs/10.1145/3394171.3413679?casa_token=NAfGkcF9aD4AAAAA:RycuI3YzktrxbcAiq10TPiJ3VseRsO_b7VhvTZM_5XZQX3k9Kqrqv8x1_BM3fKBJvC9XWK_tXvY)

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{he2020multimodal,
  title={Multimodal Dialogue Systems via Capturing Context-aware Dependencies of Semantic Elements},
  author={He, Weidong and Li, Zhi and Lu, Dongcai and Chen, Enhong and Xu, Tong and Huai, Baoxing and Yuan, Jing},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2755--2764},
  year={2020}
}
</pre>

## Abstract
Recently, multimodal dialogue systems have engaged increasing attention in several domains such as retail, travel, etc. In spite of the promising performance of pioneer works, existing studies usually focus on utterance-level semantic representations with hierarchical structures, which ignore the context-aware dependencies of multimodal semantic elements, i.e., words and images. Moreover, when integrating the visual content, they only consider images of the current turn, leaving out ones of previous turns as well as their ordinal information. To address these issues, we propose a Multimodal diAlogue systems with semanTic Elements, MATE for short. Specifically, we unfold the multimodal inputs and devise a Multimodal Element-level Encoder to obtain the semantic representation at element-level. Besides, we take into consideration all images that might be relevant to the current turn and inject the sequential characteristics of images through position encoding. Finally, we make comprehensive experiments on a public multimodal dialogue dataset in the retail domain, and improve the BLUE-4 score by 9.49, and NIST score by 1.8469 compared with state-of-the-art methods.

## Model Architecture
<p align="center">
<img src="images/model.png" width="80%" />
</p>
The architecture of the proposed MATE model, which includes two main components:

**Multimodal Element-level Encoder**: In this component, all images from the dialog history and the user query are organized as dialog image memory. Then, we allocate related images to each turn and obtain image-enhanced text embeddings through an attention mechanism. Meanwhile, all images are integrated with a user query to get a query-enhanced image embeddings. Finally, all embeddings are concatenated as multimodal semantic element embeddings.

**Knowledge-aware Two-Stage Decoder**: It is a variant of a transformer decoder for generating better responses. The first-stage decoder focuses on the multimodal conversation context from the encoder, while the second-stage decoder takes domain knowledge and results from the first decoder to further refine the responses.

## Dependency
Check the packages needed or simply run the command.
```console
❱❱❱ pip install -r requirements.txt
```

## Data
<p align="center">
<img src="images/dataset.png" width="70%" />
</p>

Download the MMD dataset and process with the following command.
```console
❱❱❱ python3 create_data.py
```
The final 

## Train and test
Training
```console
❱❱❱ python3 myTrain.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1
```
Testing
```console
❱❱❱ python3 myTest.py -path=${save_path}
```
* -bsz: batch size
* -dr: drop out ratio
* -lr: learning rate
* -le: loading pretrained embeddings
* -path: model saved path
