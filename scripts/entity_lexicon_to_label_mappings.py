from pathlib import Path
from tqdm import tqdm

from hroov.utils.data_utils import (
    _regex_parens,
    strip_parens,
    load_json,
    save_json
)

# input

print("Preparing I/O ... ")

data_dir = "./data"
entity_lexicon_fp = Path(f"{data_dir}/preprocessed_entity_lexicon.json")

# output

verbalisation_list_fp = Path(f"{data_dir}/label_verbalisations.json")
entity_map_fp = Path(f"{data_dir}/entity_map.json")
entity_mappings_list_fp = Path(f"{data_dir}/entity_mappings_list.json")

# main:

print("Loading data ... ")

entity_lexicon = load_json(entity_lexicon_fp)
iris = entity_lexicon.keys()
entity_map = {}
entity_verbalisation_list = []
list_of_entity_mappings = []

print("Producing normalised entity verbalisations (labels) ... ")

for entity_idx, entity_iri in enumerate(tqdm(iris)):
    entity_map[str(entity_idx)] = {
        "mapping_id": str(entity_idx),
        "label": entity_lexicon[entity_iri].get('name'), # type: ignore
        "verbalisation": strip_parens(str(entity_lexicon[entity_iri].get('name'))).lower(), # type: ignore
        "iri": entity_iri
    }
    entity_verbalisation_list.append(strip_parens(str(entity_lexicon[entity_iri].get('name'))).lower()) # type: ignore
    list_of_entity_mappings.append(entity_map[str(entity_idx)])

print("Saving to disk ... ")

save_json(verbalisation_list_fp, entity_verbalisation_list)
save_json(entity_map_fp, entity_map)
save_json(entity_mappings_list_fp, list_of_entity_mappings)