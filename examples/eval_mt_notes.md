# Notes for **eval_mt** requirements and implementation

* evaluation framework
    * should allow for evaluation of: NMT quality by at least one metric, inference speed, generating a summary on chosen domains
* implementation
    * at least one method of NMT speed-up: quantization, distilation, adaptive inference
* comparison
    * of quality/speed of implemented methods on various domains

## Evaluation  framework

Prototypes:
*   BLEU score
*   Eval time (transformers natively)

TODO:
*   summary - in domain vs. different domain

## Implementation
Various quantization modes

Prototypes:
*   dynamic quant. of pretrained model
*   MarianMT training FP16

TODO:
*   static quant. of pretrained model
*   quant-aware training

## Comparison 

Prototypes:
*   dynamic quant. of pretrained vs full precision


# Scenarios to support


Pretrained model -> eval -> post-train quantize -> eval quant. -> compare
Model -> training -> pretrained model -> ...
Model -> quant-aware training -> ... 



# Inteface ideas

1. To allow for easy loading, evaluation and comparison (quant/fp)
2. To support:
   0. https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
   1. https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
   2. post-training quantization
      1. dynamic
      2. static
   3. quantization-aware training
      1. fine-tuning of pretrained?
3. training own model to eval?



## Main classes
Model()

Evaluator(model)








   


