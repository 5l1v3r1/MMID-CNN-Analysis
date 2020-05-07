# MMID-CNN-Analysis


## Downloading the data
The CNN image matrices can be downloaded using the scripts in the `download` folder
in batches or via the MMID website directly:

https://multilingual-images.org/downloads.html

## Dictionaries
The dictionaries were processed from the dictionaries found
[here](http://www.seas.upenn.edu/~nlp/resources/TACL-data-release/dictionaries.tar.gz).
For each language pair, a new dictionary was created with the name
`{source_language}_to_{target_language}.tsv`. The source language's
dictionary with the name `dict.{source_language}` was translated to the
`target language` with the
[Google Translate API](https://cloud.google.com/translate/docs/basic/setup-basic). Note that translation was done one word at a time as not to provide
context that may skew the translation. When processing the scores, if a
the Google Translated word was not found, the score will be replaced with `Nan`.

## Homogeneity Scores

The calculation for the homogeneity scores is contained in `homogeneity.py`.
You will need to have downloaded the CNN matrices for both of the languages
in any language pair you run, except for English, where you will only need
to have downloaded one package for the non-English language.
The result will be written to a TSV in the Score-Results folder. The paths
for the language package folder and Score-Results folder must be defined in
`PATHS.py`. For example, to get scores between Italian and German, you can execute the following command:

```python3
python3 homogeneity.py it de
```
Note that
Italian to German vs. German to Italian, may yield different scores as it is
not guaranteed the translation mappings work forward and backward.



If the file `it_to_de_homogeneity_scores.tsv` already exists, then the
script will print that the file already exists and will stop execution; however,
if it doesn't then the file will be created. To overwrite the file you
can add `y` to the input arguments as so

```python3
python3 homogeneity.py it de y
```

## Median Max Cosine Similarity

The calculation for median max cosine similarities can be found in
```med_max_cosine.py```. As with the homogeneity scores, make sure to have
downloaded the CNN matrices for the language and also have defined the proper
paths in PATHS.py. An example execution would be

```python3
python3 med_max_cosine.py tr
```

Again you can choose to overwrite the file if it exists by adding y.


```python3
python3 med_max_cosine.py tr y
```

The result will be written to  ```tr_med_max_cosine.tsv```.

## Available Scores and Dictionaries
The following languages have already been processed:
- English
- French
- Arabic
- Azerbaijani
- Spanish
- Indonesian
- German
- Turkish
- Hindi
- Italian
- Vietnamese
- Thai
- Welsh
