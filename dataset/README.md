## Dataset
* `combined` contains the curated sentences.
    * `original.csv` contains winograd style sentences.
    * `control.csv` contains sentences where the number of the verb can disambiguate the pronoun.
    * `synonym_1.csv` and `synonym_2.csv` contain sentences where context words are synonyms.
    * `prompt.csv` contains sentences for prompt methods (for GPTs)

## Code
* `Winogrande_preprocess.py` reads Winogrande sentences and creates pairs.