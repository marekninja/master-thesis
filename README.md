# Master Thesis Repo

Despite the precision of the large language models, the deployment of these models still faces some practical issues. Except for being memory-demanding, the main issue lays in the speed of prediction. In the case of generative language models, the time of auto-regressive generation scales with the output length. Another significant limitation of translation models remains in their domain-specificity given by the domain of the training data.
    
Our work investigates the impact of model quantization on these issues. In theory, quantisation holds a potential to address these problems through lower bit-width computations allowing for model compression, speed-up, and regularization incorporated in training. Specifically, we inspect the effect that quantization has on Transformer neural language translation model. 
% Our results demonstrate that a quantised instance of Transformer is less prone to domain overfitting 
    
In addition to the obtained measurements, the contributions of this work are also in the implementations of quantized Transformer and the reusable framework for evaluation of speed, memory requirements, and distributional robustness of generative language models.

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


## Repo Notes

* Directory enmt/modified_transformers_files
    * this directory contains modified files of Transformers lib
    * modeling_marian_quant* are quantizable Marian Models - support Static Quant. and Quant.-Aware Training

## Instalation notes
 
* in some cases it is needed to have **PyTorch installed ahead of this repo**
    * tested PyTorch version: 1.11.0.dev20210929+cu102

* there might be some non-linux compatible libraries (e.g. pywin*), just skip them when it fails...


```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


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