## ARAE Tensorflow Code

Code for the paper [Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223) by Zhao, Kim, Zhang, Rush and LeCun

### Features
 - Supports generating discrete values (short sentences)
 - Based on auto-regressive LSTM (tf.nn.dynamic_rnn)

### Usage example
`python train.py --data_path=data_snli/`

### Sentence Generations (After 1 epoch)
```
a man is trying to get a picture of his friends .
a man wearing a blue shirt and a woman is standing outside of a building while a man is watching him .
a man is sleeping .
a woman with brown hair and a man in a blue shirt and black pants , is in a blue shirt and a woman in a red shirt .
a group of people are gathered .
a woman wearing a pink shirt and a pink shirt is holding a baby while another girl is holding a baby .
a human doing an orange and yellow , while being held while being held by a crowd while an older man watches .
a man in red , white , and black , is running down the street , while another is being held by his dog .
a woman wearing a red shirt and a red shirt is getting ready to make a large piece of water .
a man is holding up a piece of wood while holding a large piece of wood while holding a large crowd of people .
a woman wearing a pink shirt and a pink shirt is doing a trick while another man watches .
a woman is trying to get her picture taken while she is walking down her bike while another man is walking down a street .
```
