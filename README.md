# MMID-CNN-Analysis
## Homogeneity Scores

To produce homogeneity scores you can run "run_k_means.py" followed by the
two languages you wish to run them on. You will need the CNN folder for
that specific language or you will run into errors. An example run that will out
the homogeneity scores for Italian to German will be

```python3
python3 run_k_means.py it de
```

If the file "it_to_de_homogeneity_scores.tsv" already exists, then the script
will print so, else it will creat the file. To overwrite the file you can add
"y" to the input arguments as so

```python3
python3 run_k_means.py it de y
```
