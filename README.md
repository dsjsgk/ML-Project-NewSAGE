# ML-Project-NewSAGE

This is an implementation for the paper **Using Attention Method In GraphSAGE Model**.

To run experiments, python 3.7+, pytorch 1.11 and matplotlib are required.

If you meet all the requirements, simply run 
```
python3 main.py
```

To use attention aggregator, add
```
--attention
```

To change dataset, add
```
--dataset=0/1/2
```
0 stands for Cora, 1 stands for Citeseer and 2 stands for Pubmed.

To use new sampling method, you may change the variable `use_new_sampling` in `sampling.py` to `True`.
