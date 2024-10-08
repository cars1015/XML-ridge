# XML-ridge(A Simple but Effective Closed-form Solution for Extreme Multi-label Learning)

## dataset

- Bibtex
- Delicious200K
- Eurlex-4K
- Wiki10-31K
- AmazonCat-13K

Bibtex and Delicious200K are provided on the [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html#ba-pair),
and others are provided on the [X-transformer GitHub](https://github.com/OctoberChang/X-Transformer).

##Embedding
In our paper's experiments, we utilize embeddings obtained using the XR-transformer method, augmented with TF-IDF. For size reduction, we employ random projection for AmazonCat13K and SVD for delicious200K.

The embeddings can be accessed from the following sources:
https://drive.google.com/drive/folders/1gLjFM7JZfuj4yDOoYsDZ4bDvT--YXEGa?usp=sharing

Regarding AmazonCat13k, since we can't save it in my storage space, if you want to replicate it, please download the data provided by AttentionXML and conduct random projection.

## Train and predict

The following script is for training and evaluation on the Wiki10-31K data.
<br>
`python3 main.py 
--data "wiki10-31k"
--lambda 4`

if you use propensity score, add this script
`--w_flg`
if you use dence embedding...
`--c_flg`

## Hyper parameter

Wiki10
| DATASET | λ | λ (w-flg)|
|----------|----------|----------|
|Bibtex|92|93|
|Eurlex-4K|0.5|6|
|Wiki10-31K|4|8|
|AmazonCat-13K|0.1|0.1|
|Delicious200K|0.1|0.1|

## Reference

The evaluation code was quoted from a (https://github.com/FutureComputing4AI/Learning-with-Holographic-Reduced-Representations?tab=MIT-1-ov-file#readme)

This repository is still being edited.
論文公開までに、yamlファイル追加します。
