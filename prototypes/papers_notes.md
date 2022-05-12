### Fully Quantized Transformer for Machine Translation

quant-aware training strategy
pruning and retraining increases performance 
    pruning in tandem with training too

what to quantize

8bit same or better than FP
    6 bit too, but no HW

### Q8BERT: Quantized 8Bit BERT
quant-aware training during fine-tuning phase
    maybe this will do? to not do whole training, but only quant-aware fine-tuning?
    quant-aware training - fake quantization layer simulates rounding effect of range quant

We have released our work as part of our open source model library
NLP Architect - https://github.com/IntelLabs/nlp-architect
https://intellabs.github.io/nlp-architect/quantized_bert.html#quantization-aware-training


Comparison post-training dyn. quant. and quant.-aware training 




