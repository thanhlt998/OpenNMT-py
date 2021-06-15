python translate.py -model models/v2/checkpoints/viwikisplit_bert_step_18000.pt \
-src data/vi_wikisplit_200k/test_data.complex \
-output data/vi_wikisplit_200k/test_data.simple.pred \
-gpu 0 \
-verbose