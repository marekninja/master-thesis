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

### GLUE Bert Quantization
[Jupyter notebook](examples/glue_quantization/notes_examples.ipynb)

from: https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html
