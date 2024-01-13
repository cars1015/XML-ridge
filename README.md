# XL-ridge(A Simple but Effective Closed-form Solution for Extreme Multi-label Learning)

## dataset

- Bibtext
- Eurlex-4K
- Wiki10-31K
- AmazonCat-13K
  Bibtex is provided on the [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html#ba-pair), and others provided on the X-transformer GitHub.
  ##train and EXperiment

## Train and predict
This is an example run on Wiki10-31K using TF-IDF feature
<br>
'python3 main.py \
--data "Wiki10-31K"\
--lambda 1.3\
--A 0.55\
--B 1.5\'
if you use propensity score, add this script
'--w_flg'
if you use XLNet embedding...
'--c_flg'

This repository is still being edited.
