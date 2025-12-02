# Original Authors: Hui Yang, Jiaoyan Chen
# Credited Repo: https://github.com/HuiYang1997/OnT
# Credited Paper: https://arxiv.org/abs/2507.14334
# For license details, see the associated repositories, OnT and HiT
# 
# Initial modifications include an initial attempt at introducing hard negatives
# for NF1 only (at present) to try and guide the model to be more discriminative
# w.r.t. easy and hard examples during contrastive learning.
# Modified by: Jon Dilworth
# Status of modifications: under testing, WiP

import numpy as np
import json
import os
from random import sample
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_names(concept_file, role_file):
    with open(concept_file, 'r') as f:
        concept_names = json.load(f)
    with open(role_file, 'r') as f:
        role_names = json.load(f)
    return concept_names, role_names

def process_npy_file(file_path, concept_names, role_names, kind):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Process the data to generate sentence pairs
    # (This is where you will apply your specific formula)
    sentence_pairs = []
    conjunction_paris = []
    existential_pairs = []
    concept_ids = []


    # Example logic (to be replaced with actual processing):
    for item in data:
        # Generate sentence pairs based on item and names
        if kind == 'nf1':
            # in this case, data is a matrix of the shape (n, 2), where n is the number of sentences, and 2 is the number of concepts in each sentence
            sentence1 = concept_names[str(item[0])]
            sentence2 = concept_names[str(item[1])]
            concept_ids.append(item[0])
        elif kind == 'nf2':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqcap 1 \sqsubseteq 2
            sentence1 = f"{concept_names[str(item[0])]} and {concept_names[str(item[1])]}"
            sentence2 = concept_names[str(item[2])]
            conjunction_paris.append((concept_names[str(item[0])], concept_names[str(item[1])]))
            concept_ids.append(item[0])
        elif kind == 'nf3':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqsubseteq \exists 1. 2
            sentence1 = concept_names[str(item[0])]
            sentence2 = f"{role_names[str(item[1])]} some {concept_names[str(item[2])]}"
            existential_pairs.append((role_names[str(item[1])], concept_names[str(item[2])]))
            concept_ids.append(item[0])
        elif kind == 'nf4':
            # in this case, data is a matrix of the shape (n, 3), the axioms means \exists 0. 1 \sqsubseteq  2
            role_name = role_names[str(item[0])]
            sentence1 = f"{role_name} some {concept_names[str(item[1])]}"
            sentence2 = concept_names[str(item[2])]
            existential_pairs.append((role_name, concept_names[str(item[1])]))
            concept_ids.append(item[1])
        else:
            raise ValueError(f"Unknown axiom kind: {kind}")
        sentence_pairs.append((sentence1, sentence2))
    
    return sentence_pairs, conjunction_paris, existential_pairs, concept_ids

def process_test_val_data(file_path, concept_names, role_names, kind):
     # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Process the data to generate sentence pairs
    # (This is where you will apply your specific formula)
    sentences = []
    answer_ids = []
    roles, cons = [], []
    con1s, con2s = [], []

    # Example logic (to be replaced with actual processing):
    for item in data:
        # Generate sentence pairs based on item and names
        if kind == 'nf1':
            # in this case, data is a matrix of the shape (n, 2), where n is the number of sentences, and 2 is the number of concepts in each sentence
            sentences.append(concept_names[str(item[0])])
            answer_ids.append(int(item[1]))
        elif kind == 'nf2':
            sentences.append(f"{concept_names[str(item[0])]} and {concept_names[str(item[1])]}")
            answer_ids.append(int(item[2]))
            con1s.append(concept_names[str(item[0])])
            con2s.append(concept_names[str(item[1])])
        elif kind == 'nf3':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqsubseteq \exists 1. 2
            sentences.append(f"{role_names[str(item[1])]} some {concept_names[str(item[2])]}")
            answer_ids.append(int(item[0]))
            roles.append(role_names[str(item[1])])
            cons.append(concept_names[str(item[2])])
        elif kind == 'nf4':
            role_name = role_names[str(item[0])]
            # in this case, data is a matrix of the shape (n, 3), the axioms means \exists 0. 1 \sqsubseteq  2
            roles.append(role_name)
            sentences.append(f"{role_name} some {concept_names[str(item[1])]}")
            answer_ids.append(int(item[2]))
            cons.append(concept_names[str(item[1])])
        else:
            raise ValueError(f'Unknown kind: {kind}')
    
    return sentences, answer_ids, roles, cons, con1s, con2s

def load_val_test(data_dir, out_dir, load_type = 'val'):
    concept_file = f'{data_dir}/concept_names.json'
    role_file = f'{data_dir}/role_names.json'
    inverse_role_file = f'{data_dir}/role_inverse_mapping.json'    
    
    # Load names
    concept_names, role_names = load_names(concept_file, role_file) 

    # process test or val data
    processed_data = {'query_sentences':{}, 'answer_ids':{}}
    data_dir = f'{data_dir}/ont_tmp/{load_type}'
    file_names_to_process = [f'nf{i}.npy' for i in range(1, 5)]
    for file_name in file_names_to_process:
        file_path = os.path.join(data_dir, file_name)
        sentences, answer_ids, roles, cons, con1s, con2s = process_test_val_data(file_path, concept_names, role_names, kind = file_name[:-4])
        processed_data['answer_ids'][file_name[:-4]] = answer_ids
        if file_name == 'nf1.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":s} for s in sentences]
        elif file_name == 'nf2.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":sentences[i], "con1":con1s[i], "con2":con2s[i]} for i in range(len(sentences))]
        elif file_name == 'nf3.npy' or file_name == 'nf4.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":sentences[i], "role":roles[i], "con":cons[i]} for i in range(len(sentences))]
        else:
            print('error')

    with open(os.path.join(out_dir, f'{load_type}.json'), 'w') as f:
        json.dump(processed_data, f, indent=8)

    return


# NOTE: this is a pretty complicated function
# changes made to this local fork still being tested
# bugs may or may not have been introduced... BEWARE.
# TODO: consider potential for hard negatives for NF2 -> NF4
def load_train(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    concept_file = f'{data_dir}/concept_names.json'
    role_file = f'{data_dir}/role_names.json'
    
    # Load names
    concept_names, role_names = load_names(concept_file, role_file)
    
    # Load hard negatives if available
    hard_neg_path = os.path.join(data_dir, 'hard_negatives.json')
    use_hard_negatives = os.path.exists(hard_neg_path)
    
    hard_negatives: dict[str, list[str]] = {}
    if use_hard_negatives:
        print("Loading hard negative structure...")
        with open(hard_neg_path, 'r') as f:
            hard_neg_data = json.load(f)
        hard_negatives = hard_neg_data['hard_negatives']
    else:
        print("No hard negative structure found, using random sampling")

    # Translate data
    file_names_to_process = [f'nf{i}.npy' for i in range(1, 5)]

    conjunction_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    existential_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    sentence_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    
    concept_id_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
  
    nf_data_dir = f'{data_dir}/ont_tmp/train'
    for file_name in file_names_to_process:
        file_path = os.path.join(nf_data_dir, file_name)
        sent_pairs, conj_pairs, exist_pairs, nf1_concept_ids = process_npy_file(
            file_path, concept_names, role_names, kind = file_name[:-4]
        )
        sentence_pair_dict[file_name[:-4]]+=sent_pairs
        conjunction_pair_dict[file_name[:-4]]+=conj_pairs
        existential_pair_dict[file_name[:-4]]+=exist_pairs
        concept_id_dict[file_name[:-4]]+=nf1_concept_ids
    
    # save each nf data separately
    all_data = []
    for key, sentence_pairs in sentence_pair_dict.items():
        roles = existential_pair_dict[key]
        conj_pairs = conjunction_pair_dict[key]
        concept_ids = concept_id_dict[key]
        output_file = f'{out_dir}/train_{key}.jsonl'
        with open(output_file, 'w') as out_file:
            for ind, pair in enumerate(sentence_pairs):
                # randomly select 10 names from concept_names as negative samples
                if key == 'nf3':
                    role_name = f"{roles[ind][0]} some "
                else:
                    role_name = ""
                negative_samples = sample(list(concept_names.values()), k=10)
                negative_samples = [role_name + negs for negs in negative_samples] 
                # ^ do we want role + negs \forall negs \in samples for HiT data?
                # --- 
                # OLD CODE:
                # data_for_hit = {"child": pair[0], "parent": pair[1], "negative": negative_samples} # <- roles would be added here, no?
                # all_data.append(data_for_hit)
                # END OLD CODE
                # --- 
                if use_hard_negatives and str(concept_ids[ind]) in hard_negatives:
                    # neg_concept_ids = hard_negatives[concept_ids[ind]]
                    neg_concept_ids = hard_negatives[str(concept_ids[ind])]
                    hard_negatives_for_nf1 = [
                        concept_names[str(neg_id)] 
                        for neg_id in neg_concept_ids 
                        if str(neg_id) in concept_names
                    ]
                    if len(hard_negatives_for_nf1) < 10:
                        # continue sampling until we've got 10
                        remaining_negatives_required = 10 - len(hard_negatives_for_nf1)
                        sampled_random_negatives = sample(list(concept_names.values()), k=remaining_negatives_required)
                        hard_negatives_for_nf1.extend(sampled_random_negatives)
                    # else: we've got the complete list of 10, in any case:
                    negative_samples = hard_negatives_for_nf1
                else: # fallback to random sampling (as in the original)
                    negative_samples = sample(list(concept_names.values()), k=10)
                data_for_hit = {"child": pair[0], "parent": pair[1], "negative": negative_samples}
                all_data.append(data_for_hit)
                # !! IMPORTANT !! Verify the contents of the output files !!
                negative_samples_for_NF2 = sample(list(concept_names.values()), k=10)
                negative_samples_for_NF3_NF4 = [role_name + negs for negs in negative_samples_for_NF2]
                if key == 'nf1':
                    data_ours = data_for_hit
                elif key == 'nf2':
                    data_ours = { "con1":conj_pairs[ind][0], "con2":conj_pairs[ind][1], "parent":pair[1], "negative": negative_samples_for_NF2}
                elif key == 'nf3':
                    # negative_samples = [role_name + negs for negs in negative_samples] 
                    data_ours = {"atomic":pair[0], "role":roles[ind][0], "con":roles[ind][1], "negative": negative_samples_for_NF3_NF4}
                elif key == 'nf4':
                    data_ours = {"atomic":pair[1], "role":roles[ind][0], "con":roles[ind][1], "negative": negative_samples_for_NF3_NF4}
                else:
                    raise ValueError(f"Unknown axiom kind: {key}")

                out_file.write(json.dumps(data_ours) + "\n")
    
    # save all data
    output_file = f'{out_dir}/train.jsonl'
    with open(output_file, 'w') as out_file:
        for data in all_data:
            out_file.write(json.dumps(data) + "\n")
    
    # save conjunction pairs
    output_file = f'{out_dir}/train_conj.jsonl'
    with open(output_file, 'w') as out_file:
        # Write sentence pairs to the output file
        for key, conj_pairs in conjunction_pair_dict.items():
            for pair in conj_pairs:
                data = {"Concept": f"{pair[0]} and {pair[1]}", "con1": pair[0], "con2": pair[1]}
                out_file.write(json.dumps(data) + "\n")
    

    # save existential pairs
    output_file = f'{out_dir}/train_exist.jsonl'
    with open(output_file, 'w') as out_file:
        # Write sentence pairs to the output file
        for key, existential_pairs in existential_pair_dict.items():
            for pair in existential_pairs:
                data = {"Concept": f"{pair[0]} some {pair[1]}", "role": pair[0], "con": pair[1]}
                out_file.write(json.dumps(data) + "\n")
    
    # save all entities
    output_file = f'{out_dir}/concept_names.json'
    with open(output_file, 'w') as out_file:
        json.dump(concept_names, out_file, indent=4)

    # save all roles
    output_file = f'{out_dir}/role_names.json'
    with open(output_file, 'w') as out_file:
        json.dump(role_names, out_file, indent=4)

    
    return

if __name__ == "__main__":
    all_data_dir = '/data/Hui/HiT_new/HierarchyTransformers/data_new'
    load_train(all_data_dir, task='OnT')
    load_val_test(all_data_dir, task='OnT', load_type = 'val')
    load_val_test(all_data_dir, task='OnT', load_type = 'test')