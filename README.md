# master-thesis
Master thesis repo

# Work outcome:
* evaluation framework
    * should allow for evaluation of: NMT quality by at least one metric, inference speed, generating a summary on chosen domains
* implementation
    * at least one method of NMT speed-up: quantization, distilation, adaptive inference
* comparison
    * of quality/speed of implemented methods on various domains

Vystup prace by mal byt:
* Evaluačný framework: umožňujúci evaluáciu kvality strojového prekladu s pomocou aspoň jednej povrchovej, syntaktickej a sémantickej metriky, evaluáciu rýchlosti predikcie a generáciu prehľadu na vybraných doménach
* Implementáciu aspoň jednej metódy zrýchlenia inferencie za pomoci kvantizácie, distilácie alebo adaptívnej inferencie
* Porovnanie kvality a rýchlosti implementovanej metódy na viacerých doménach.


## Notes

* Directory enmt/modified_transformers_files
    * this directory contains modified files of Transformers lib
    * modeling_marian_quant* are quantizable Marian Models - support Static Quant. and Quant.-Aware Training
 
* in some cases it is needed to have **PyTorch installed ahead of this repo**


## Reproducing results:

Scripts with prefix `runner` are scripts from our experiments.
You can run them to reproduce results.


## Prototypes:

### Evaluation of pretrained model 
[Jupyter notebook](examples/train_and_eval/eval_pretrained.ipynb)

INT8 quantized model(MarianMT) has nearly same BLEU score and is 1.7times faster than in FP


### GLUE Bert Quantization
[Jupyter notebook](examples/glue_quantization/notes_examples.ipynb)

from: https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html


### Loding and quantization of pretrained models
[Jupyter notebook](examples/loadings/pretrained_model_quant.ipynb)


Also contains dataset preprocessing