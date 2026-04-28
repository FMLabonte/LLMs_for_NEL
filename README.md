
## The Broader Idea
We want to train Relation Extraction (RE) models for biomedical texts, which is a relatively hard task requiring expensive human annotators. The problem is that only a limited amount of training data is available. The idea is to use LLMs to create synthetic data by prompting them with a list of relations that should be described in an abstract, then having the model generate the abstract. However, this creates a chicken-and-egg problem: no models exist to verify that the LLM actually follows the input and provides it correctly in the synthetic abstract.
### The Solution
We need a way to verify this synthetic data. Luckily, we have around 500 annotated abstracts that we can treat as a gold standard. Using these, we can switch annotations and text, treating the annotated abstract as if it were the generation of a model. We can then introduce perturbations in the annotations to introduce errors in a controlled way—for example, flipping a label from negative to positive. We can then train a small model to identify those errors by providing the text and the expected label, turning this into a binary classification task.
Producing synthetic data is cheap, and with an effective filter, we can ensure data quality.

#### Example 
We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene. -> From this we can take that VEGFA is negatively correlated SRY since its deactivation leads to an increase of the other. expressed as an annotation VEGFA negative_correlation SRY 

Now we can treat this annotation and text pair as input and output in put being Relation: [VEGFA negative_correlation SRY] and our theoretical synthetic data creation model would have written out: text: [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] <- this reversed pair is our gold sample

Relation: [VEGFA negative_correlation SRY] Text: [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] Label: [true] 

We want to see if we can train a model that would spot a missmatch between input relation pair and output Sentence. changing sentences is an controlable manner is hard, Changing annotations is not. 
VEGFA negative_correlation SRY can be changed to VEGFA positive_correlation SRY. Now the model is given this and the input text, with the quetion is this tripplet correctly represented by the text ?

Relation: [VEGFA positive_correlation SRY] Text: [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] Label: [wrong] 

Using this we can create different errors: 
Swapping labels, Connecting enteties that arent connected, Connecting enteties with enteties that are not in the text etc. and figure out which of those the models can spot and how accurately.


## Example to run the demo.:

`python3 pubtator_parser.py Data/BioRED/Dev.PubTator `

Example usage of the custom data loader this allows you to load all 3 datasets that we are interested in uniformely. Making it easier to work with down the line
```python
from pubtator_parser import parse_pubtator, save_dataframes, load_dataframes, enrich_relations

# Parse — now returns 3 DataFrames
meta, anns, rels = parse_pubtator("Data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt")

# Join annotations with metadata
combined = anns.merge(meta, on="pmid")

# Filter to just Chemical entities (example)
chemicals = anns[anns["entity_type"] == "Chemical"]

# Enrich relations with human-readable mention names
rels_named = enrich_relations(rels, anns)

# Save — pass rels as the third argument
save_dataframes(meta, anns, rels, prefix="CDR_test", output_dir="output/")

# Load returns 3 DataFrames metadata(abstract and PID), Annotations, Relations 
meta, anns, rels = load_dataframes(prefix="CDR_test", input_dir="output/")
``` 
