# Training Custom HiT Models

1. Include your custom ontology within the `data` directory. For this example, we will simply use `custom.owl` as our custom ontology.

2. Generate the local entity lexicon (for training) according to your pre-processing preferences with:

```sh
# generates the entity_lexicon (the set of named classes)
python scripts/load_taxonomy.py --ontology data/custom.owl --training-data data/hit_dataset_custom

# pre-processes (and transforms) the output of the previous step to enable training local HiT models for multi-hop inference
python scripts/preprocess_entity_lexicon.py --input data/hit_dataset_custom/multi/entity_lexicon.json \
    --output data/hit_dataset_custom/multi/local_entity_lexicon.json \
    --strip-parens \
    --to-lower-case

# pre-processes (and transforms) the output of the previous step to enable training local HiT models for mixed-hop prediction
python scripts/preprocess_entity_lexicon.py --input ./data/hit_dataset_custom/mixed/entity_lexicon.json \
    --output ./data/hit_dataset_custom/mixed/local_entity_lexicon.json \
    --strip-parens \
    --to-lower-case
```

3. Modify the training parameters within [lib/hierarchy_transformers/scripts/config.yaml](../lib/hierarchy_transformers/scripts/config.yaml):

```yaml
dataset_path: "./data/hit_dataset_custom"
dataset_name: "mixed" # or multi

# pre-trained base model from Hugging Face
model_path: "sentence-transformers/all-MiniLM-L12-v2"

# training config
num_train_epochs: 20
train_batch_size: 256
eval_batch_size: 512
learning_rate: 1e-5 
hit_loss: 
  clustering_loss_weight: 1.0
  clustering_loss_margin: 5.0
  centripetal_loss_weight: 1.0
  centripetal_loss_margin: 0.1
```

4. Run the training procedure:

```sh
python lib/hierarchy_transformers/scripts/train_hit.py -c lib/hierarchy_transformers/scripts/config.yaml
```