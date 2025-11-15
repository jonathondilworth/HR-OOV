from pathlib import Path
import json
import copy
import latextable
from latextable import texttable
import statistics
from sklearn.metrics import auc as sk_auc
import numpy as np
# ^ python imports
# module/lib imports:
from hroov.utils.retrievers import (
    TFIDFRetriever,
    BM25Retriever,
    SBERTRetriever,
    HiTRetriever,
    OnTRetriever
)
from hroov.utils.math_functools import (
  batch_cosine_similarity,
  batch_euclidian_l2_distance,
  batch_poincare_dist_with_adaptive_curv_k,
  entity_subsumption,
  concept_subsumption
)
from hroov.utils.math_functools import (
   macro_pr_curve,
   compute_ndcg_at_k
)
from hroov.utils.data_utils import load_json
from hroov.utils.query_utils import QueryObjectMapping, QueryResult
from hierarchy_transformers import HierarchyTransformer
from OnT.OnT import OntologyTransformer
from sentence_transformers import SentenceTransformer

embeddings_dir = "./embeddings"

common_map = Path(f"{embeddings_dir}/entity_mappings.json")
common_verbalisations = Path(f"{embeddings_dir}/verbalisations.json")

verbalisations = load_json(common_verbalisations)

## Lexical Baseline Retrievers (TF-IDF, BM25):

tfidf_ret = TFIDFRetriever(common_verbalisations, common_map)
bm25_ret = BM25Retriever(common_verbalisations, common_map, k1=1.3, b=0.7)

# SBERT

sbert_plm_hf_string = "all-MiniLM-L12-v2"

sbert_ret_plm_w_cosine_sim = SBERTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/sbert-plm-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_cosine_similarity
)

sbert_ret_plm_w_euclid_dist = SBERTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/sbert-plm-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_str="all-MiniLM-L12-v2",
  score_fn=batch_euclidian_l2_distance
)

# HiT SNOMED 25 (Full) Random Negatives

hit_snomed_25_model_fp = './models/snomed_models/HiT-mixed-SNOMED-25/final'
hit_SNOMED25_model_path = Path(hit_snomed_25_model_fp)

hit_ret_snomed_25_w_hyp_dist = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_SNOMED25_model_path,
  score_fn=batch_poincare_dist_with_adaptive_curv_k
)

hit_ret_snomed_25_w_ent_sub = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-25-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_SNOMED25_model_path,
  score_fn=entity_subsumption
)

# HiT SNOMED CT (Full) Hard Negatives

hit_snomed_hard_model_fp = './models/snomed_models/HiT_mixed_hard_negatives/'
hit_snomed_hard_model_path = Path(hit_snomed_hard_model_fp)

hit_snomed_hard_w_hyp_dist = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-hard-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_snomed_hard_model_path,
  score_fn=batch_poincare_dist_with_adaptive_curv_k
)

hit_snomed_hard_w_ent_sub = HiTRetriever(
  embeddings_fp=Path(f"{embeddings_dir}/hit-snomed-hard-embeddings.npy"),
  meta_map_fp=common_map,
  verbalisations_fp=common_verbalisations,
  model_fp=hit_snomed_hard_model_path,
  score_fn=entity_subsumption
)

# OnT-96 SNOMED CT 2025 (Full)

ont_snomed_96_model_fp = "./models/snomed_models/OnT-96"
ont_snomed_96_model_path = Path(ont_snomed_96_model_fp)

ont_snomed_96_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-96-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_snomed_96_model_path,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ont_snomed_96_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-96-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ont_snomed_96_model_path,
    score_fn=concept_subsumption
)

# OnT Mini SNOMED CT 2025 (M-128)

ontr_snomed_minified_128_model_fp = './models/snomed_models/OnTr-m-128'
ontr_snomed_minified_128_model_fp = Path(ontr_snomed_minified_128_model_fp)

ontr_ret_snomed_minified_128_w_hyp_dist = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_128_model_fp,
    score_fn=batch_poincare_dist_with_adaptive_curv_k
)

ontr_ret_snomed_minified_128_w_con_sub = OnTRetriever(
    embeddings_fp=Path(f"{embeddings_dir}/ont-snomed-minified-128-embeddings.npy"),
    meta_map_fp=common_map,
    verbalisations_fp=common_verbalisations,
    model_fp=ontr_snomed_minified_128_model_fp,
    score_fn=concept_subsumption
)

#### EXPERIMENTAL RUNS

# load query objects via ORM-style QueryObjectMapping:
data_query_mapping = QueryObjectMapping(Path("./data/eval_dataset_50.json"))
equiv_queries, subsumpt_queries = data_query_mapping.get_queries()

# specify the 'cutoff_depth' (i.e. 'd' parameter)
global_cutoff_depth = 3

# set up the 'models dict' ready for experimental runs (single target):

models_dict = {
  # BASELINES
  "BoW Lex-BASE TFIDF": tfidf_ret,
  "BoW Lex-BASE BM25": bm25_ret,
  # BASELINE CONTEXTUALISED EMBEDDINGS
  "SBERT Contx-BASE cos-sim": sbert_ret_plm_w_cosine_sim,
  # HiT (Full)
  "HiT SNO-25(F-R) d_k": hit_ret_snomed_25_w_hyp_dist,
  "HiT SNO-25(F-R) s_e": hit_ret_snomed_25_w_ent_sub,
  "HiT SNO-25(F-H) d_k": hit_snomed_hard_w_hyp_dist,
  "HiT SNO-25(F-H) s_e": hit_snomed_hard_w_ent_sub,
  # OnT SNOMED Models (Full, batch_size=64, Mini, batch_size=[32,64,128])
  "OnT SNO-25(F) d_k": ont_snomed_96_w_hyp_dist,
  "OnT SNO-25(F) s_c": ont_snomed_96_w_con_sub,
  "OnT SNO-25(M-128) d_k": ontr_ret_snomed_minified_128_w_hyp_dist,
  "OnT SNO-25(M-128) s_c": ontr_ret_snomed_minified_128_w_con_sub,
} 

print("Running for d=3:")

# PREP TABLE START #
experiment_table = texttable.Texttable()
experiment_table.set_deco(texttable.Texttable.HEADER)
experiment_table.set_precision(2)
experiment_table.set_cols_dtype(['t', 't', 't', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'])
experiment_table.set_cols_align(["l", "l", "l", "c", "c", "c", "c", "c", "c", "c", "c", "c"])
experiment_table.header(["Model", "Variant", "Metric", "mAP", "MRR", "H@1", "H@3", "H@5", "Med", "MR", "nDCG@10", "R@100"])
# END-PREP TABLE #

ks      = [1, 3, 5, 10, 100, len(verbalisations)]
MAX_K   = max(ks)

all_results = {}
macro_avg_PR_AUC_data = {}

centripetal_weight = 0.1

for model_name, model in models_dict.items():
    
    # init accumulators
    results = {
      "MRR": 0.0, # Mean Reciprical Rank
      "MAP": 0.0, # Mean Average Precision
      **{f"Hits@{k}": 0.0 for k in ks},
      **{f"P@{k}": 0.0 for k in ks}, # Precision@k
      **{f"R@{k}": 0.0 for k in ks}, # Recall@k
      **{f"F1@{k}": 0.0 for k in ks}, # F1@k
      **{f"nDCG@{k}": 0.0 for k in ks}, # normalised Discounted Cumlative Gain @ k
      "MR": 0.0, # Mean Rank
      "aRP": 0.0  # R-Precision
    }
    # AUC-PR, Median Rank & Coverage are calculated during the test procedure
    hit_count = 0 # for coverage
    total_possible_hits = 0 # for coverage := hit_count / total_possible_hits .. essentially: recall@k, when k = MAX_K
    all_ranks = [] # for median rank
    # @depricationWarning : AUC-PR, previous implementation was rough approximation
    per_query_rels_for_PR = []

    if model_name == "HiT SNO-25(F-H) s_e":
       centripetal_weight = 0.15
    elif model_name == "OnT SNO-25(F) s_c":
       centripetal_weight = 0.37
    else:
       centripetal_weight = 0.10

    for q_idx, query in enumerate(subsumpt_queries):
        
        qstr = query.get_query_string()
        gold_targets = query.get_unique_sorted_subsumptive_targets(key="depth", reverse=False, depth_cutoff=global_cutoff_depth) # [*parents, *ancestors]
        g_target_iris = set([x["iri"] for x in gold_targets])
        num_targets = len(g_target_iris)
        total_possible_hits += num_targets
        average_precision = 0.0
        hit_count_this_query = 0
        hit_count_lt_or_eq_num_targets = 0

        ranked_results: list[QueryResult] = [] # empty lists (are unlikely to exist) but are treated as full misses
        
        # TODO: replace with match (?) - i.e. switch
        if isinstance(model, HiTRetriever):
          if model._score_fn == entity_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=centripetal_weight)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, OnTRetriever):
          if model._score_fn == concept_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=centripetal_weight)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, SBERTRetriever):
          if model._score_fn == batch_cosine_similarity:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True)
          else:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False)
        #
        elif isinstance(model, BM25Retriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        elif isinstance(model, TFIDFRetriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        else:
           raise ValueError("No appropriate retriever has been set.")

        retrieved_iris = [iri for (_, iri, _, _) in ranked_results] # type: ignore

        # (macro) PR-AUC
        rel_binary = []
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
              rel_binary.append(1)
            else:
              rel_binary.append(0)
        per_query_rels_for_PR.append((rel_binary, num_targets))

        # MRR & Mean Rank (on the first hit)
        rank_pos = None
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
                rank_pos = rank_idx
                results["MRR"] += 1.0 / rank_idx
                results["MR"] += rank_idx
                break
        
        # Average Precision (this query), for use in calculating mAP
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count += 1
              hit_count_this_query += 1
              average_precision += hit_count_this_query / rank_idx
        average_precision /= num_targets
        results["MAP"] += average_precision

        # R-Precision (this query)
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count_lt_or_eq_num_targets += 1
           if rank_idx == num_targets: # then we need to calculate the precision @ this index
              results["aRP"] += hit_count_lt_or_eq_num_targets / num_targets
              break

        # include a penalty to appropriately offset the MR
        # rather than artifically inflating the performance
        # by simply dropping queries that do not contain 
        # (unlikely in this case)
        if rank_pos is None:
            results["MR"] += MAX_K + 1 # penalty: rank := MAX_K + 1

        for k in ks:
            hit = 1 if (rank_pos is not None and rank_pos <= k) else 0
            results[f"Hits@{k}"] += hit
            top_k_results = set(retrieved_iris[:k])
            total_hits_at_k = len(g_target_iris.intersection(top_k_results))
            p_at_k = total_hits_at_k / k # Precision@K
            results[f"P@{k}"] += p_at_k
            r_at_k = total_hits_at_k / num_targets
            results[f"R@{k}"] += r_at_k
            if (p_at_k + r_at_k) > 0:
               results[f"F1@{k}"] += 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)
            iDCG, targets_with_dcg = query.get_targets_with_dcg(type="exp", depth_cutoff=global_cutoff_depth)
            results[f"nDCG@{k}"] += compute_ndcg_at_k(ranked_results, targets_with_dcg, k) # type: ignore

        final_rank = rank_pos if rank_pos is not None else MAX_K + 1
        all_ranks.append(final_rank)

    # (macro) PR-AUC
    R_grid, P_macro = macro_pr_curve(per_query_rels_for_PR, recall_points=101)
    macro_pr_auc = float(np.trapezoid(P_macro, R_grid))

    # normalise over queries & compute coverage
    N = len(subsumpt_queries)
    normalized = {metric: value / N for metric, value in results.items()}
    normalized['Cov'] = (hit_count / total_possible_hits) # calculate the coverage of this model
    normalized['Med'] = statistics.median(all_ranks) # median rank
    # area under precision-recall curve (trapezodial rule)
    recall_at_k_xs    = [normalized[f"R@{k}"] for k in ks]
    # check for monotonic recall
    if any(r2 < r1 for r1, r2 in zip(recall_at_k_xs, recall_at_k_xs[1:])):
        raise ValueError(f"Recall must be non-decreasing for PR-AUC")
    precision_at_k_xs = [normalized[f"P@{k}"] for k in ks]
    normalized["AUC"] = float(sk_auc(recall_at_k_xs, precision_at_k_xs))
    normalized["MacroPR_AUC"] = macro_pr_auc

    print(f"Model: {model_name}")
    print(f" mAP: \t  {normalized['MAP']:.2f}") # Mean Average Precision
    print(f" MRR*:   {normalized['MRR']:.2f}") # MRR at first hit ranks
    for k in [1, 3, 5]:
        print(f"  Hits@{k}:    {normalized[f'Hits@{k}']:.2f}")
    print(f" Med:     {normalized['Med']:.2f}") # Median Rank
    print(f" MR:      {normalized['MR']:.2f} ") # Mean Rank
    print(f" nDCG@10: {normalized['nDCG@10']:.2f}") # nDCG@10
    print(f" PR-AUC:  {normalized['AUC']:.2f}") # area under precision-recall curve
    print(f" mPR-AUC: {normalized['MacroPR_AUC']:.2f}") # PR-AUC (macro averaged)
    print(f" R@100:   {normalized['R@100']:.2f}") # Recall@100
    print("-"*60)

    all_results[model_name] = normalized

    model_metric_string = model_name.split()

    experiment_table.add_row([
      model_metric_string[0],
      model_metric_string[1],
      model_metric_string[2],
      normalized['MAP'], 
      normalized['MRR'],
      normalized['Hits@1'], 
      normalized['Hits@3'], 
      normalized['Hits@5'],
      normalized['Med'], 
      normalized['MR'],
      normalized['nDCG@10'],
      normalized['R@100']
    ])

    macro_avg_PR_AUC_data[model_name] = {
      "recall": R_grid.tolist(),
      "precision": P_macro.tolist()
    }

Path('../logs').mkdir(parents=True, exist_ok=True)
output_file = '../logs/oov_entity_mentions_d_equals_three_results.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"All results saved to {output_file}")

output_macro_pr_auc_file = '../logs/oov_entity_mentions_d_equals_three__PR_AUC_POINTS_4_PLOT.json'
with open(output_macro_pr_auc_file, 'w') as f:
    json.dump(macro_avg_PR_AUC_data, f, indent=2)

print(f"Macro PR AUC plot data dumped to {output_macro_pr_auc_file}")

print(f"Printing table: \n\n")

print(experiment_table.draw())

print("\n\n Printing LaTeX: \n\n")

print(latextable.draw_latex(
    table=experiment_table,
    caption="Performance of fetching multiple relevant targets using OOV mentions with lambda=0.1 & 0.37 (50 Queries)",
    use_booktabs=True, position="H", caption_above=True, caption_short="Multi target performance of OOV mentions, lambda={0.1,0.37}",
    label="tab:multi-target-oov-weighted-3"
  )
)

print("Running for d=5:")

global_cutoff_depth = 5

# PREP TABLE START #
experiment_table_two = texttable.Texttable()
experiment_table_two.set_deco(texttable.Texttable.HEADER)
experiment_table_two.set_precision(2)
experiment_table_two.set_cols_dtype(['t', 't', 't', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f'])
experiment_table_two.set_cols_align(["l", "l", "l", "c", "c", "c", "c", "c", "c", "c", "c", "c"])
experiment_table_two.header(["Model", "Variant", "Metric", "mAP", "MRR", "H@1", "H@3", "H@5", "Med", "MR", "nDCG@10", "R@100"])
# END-PREP TABLE #

ks      = [1, 3, 5, 10, 100, len(verbalisations)]
MAX_K   = max(ks)

all_results = {}
macro_avg_PR_AUC_data = {}

centripetal_weight = 0.1

for model_name, model in models_dict.items():
    
    # init accumulators
    results = {
      "MRR": 0.0, # Mean Reciprical Rank
      "MAP": 0.0, # Mean Average Precision
      **{f"Hits@{k}": 0.0 for k in ks},
      **{f"P@{k}": 0.0 for k in ks}, # Precision@k
      **{f"R@{k}": 0.0 for k in ks}, # Recall@k
      **{f"F1@{k}": 0.0 for k in ks}, # F1@k
      **{f"nDCG@{k}": 0.0 for k in ks}, # normalised Discounted Cumlative Gain @ k
      "MR": 0.0, # Mean Rank
      "aRP": 0.0  # R-Precision
    }
    # AUC-PR, Median Rank & Coverage are calculated during the test procedure
    hit_count = 0 # for coverage
    total_possible_hits = 0 # for coverage := hit_count / total_possible_hits .. essentially: recall@k, when k = MAX_K
    all_ranks = [] # for median rank
    # @depricationWarning : AUC-PR, previous implementation was rough approximation
    per_query_rels_for_PR = []

    if model_name == "HiT SNO-25(F-H) s_e":
       centripetal_weight = 0.15
    elif model_name == "OnT SNO-25(F) s_c":
       centripetal_weight = 0.37
    else:
       centripetal_weight = 0.10

    for q_idx, query in enumerate(subsumpt_queries):
        
        qstr = query.get_query_string()
        gold_targets = query.get_unique_sorted_subsumptive_targets(key="depth", reverse=False, depth_cutoff=global_cutoff_depth) # [*parents, *ancestors]
        g_target_iris = set([x["iri"] for x in gold_targets])
        num_targets = len(g_target_iris)
        total_possible_hits += num_targets
        average_precision = 0.0
        hit_count_this_query = 0
        hit_count_lt_or_eq_num_targets = 0

        ranked_results: list[QueryResult] = [] # empty lists (are unlikely to exist) but are treated as full misses
        
        # TODO: replace with match (?) - i.e. switch
        if isinstance(model, HiTRetriever):
          if model._score_fn == entity_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=centripetal_weight)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, OnTRetriever):
          if model._score_fn == concept_subsumption:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True, model=model._model, weight=centripetal_weight)
          elif model._score_fn == batch_poincare_dist_with_adaptive_curv_k: 
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False, model=model._model)
        #
        elif isinstance(model, SBERTRetriever):
          if model._score_fn == batch_cosine_similarity:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=True)
          else:
            ranked_results = model.retrieve(qstr, top_k=MAX_K, reverse_candidate_scores=False)
        #
        elif isinstance(model, BM25Retriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        elif isinstance(model, TFIDFRetriever):
          ranked_results = model.retrieve(qstr, top_k=MAX_K)
        #
        else:
           raise ValueError("No appropriate retriever has been set.")

        retrieved_iris = [iri for (_, iri, _, _) in ranked_results] # type: ignore

        # (macro) PR-AUC
        rel_binary = []
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
              rel_binary.append(1)
            else:
              rel_binary.append(0)
        per_query_rels_for_PR.append((rel_binary, num_targets))

        # MRR & Mean Rank (on the first hit)
        rank_pos = None
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
            if iri in g_target_iris:
                rank_pos = rank_idx
                results["MRR"] += 1.0 / rank_idx
                results["MR"] += rank_idx
                break
        
        # Average Precision (this query), for use in calculating mAP
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count += 1
              hit_count_this_query += 1
              average_precision += hit_count_this_query / rank_idx
        average_precision /= num_targets
        results["MAP"] += average_precision

        # R-Precision (this query)
        for rank_idx, iri in enumerate(retrieved_iris, start=1):
           if iri in g_target_iris:
              hit_count_lt_or_eq_num_targets += 1
           if rank_idx == num_targets: # then we need to calculate the precision @ this index
              results["aRP"] += hit_count_lt_or_eq_num_targets / num_targets
              break

        # include a penalty to appropriately offset the MR
        # rather than artifically inflating the performance
        # by simply dropping queries that do not contain 
        # (unlikely in this case)
        if rank_pos is None:
            results["MR"] += MAX_K + 1 # penalty: rank := MAX_K + 1

        for k in ks:
            hit = 1 if (rank_pos is not None and rank_pos <= k) else 0
            results[f"Hits@{k}"] += hit
            top_k_results = set(retrieved_iris[:k])
            total_hits_at_k = len(g_target_iris.intersection(top_k_results))
            p_at_k = total_hits_at_k / k # Precision@K
            results[f"P@{k}"] += p_at_k
            r_at_k = total_hits_at_k / num_targets
            results[f"R@{k}"] += r_at_k
            if (p_at_k + r_at_k) > 0:
               results[f"F1@{k}"] += 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)
            iDCG, targets_with_dcg = query.get_targets_with_dcg(type="exp", depth_cutoff=global_cutoff_depth)
            results[f"nDCG@{k}"] += compute_ndcg_at_k(ranked_results, targets_with_dcg, k) # type: ignore

        final_rank = rank_pos if rank_pos is not None else MAX_K + 1
        all_ranks.append(final_rank)

    # (macro) PR-AUC
    R_grid, P_macro = macro_pr_curve(per_query_rels_for_PR, recall_points=101)
    macro_pr_auc = float(np.trapezoid(P_macro, R_grid))

    # normalise over queries & compute coverage
    N = len(subsumpt_queries)
    normalized = {metric: value / N for metric, value in results.items()}
    normalized['Cov'] = (hit_count / total_possible_hits) # calculate the coverage of this model
    normalized['Med'] = statistics.median(all_ranks) # median rank
    # area under precision-recall curve (trapezodial rule)
    recall_at_k_xs    = [normalized[f"R@{k}"] for k in ks]
    # check for monotonic recall
    if any(r2 < r1 for r1, r2 in zip(recall_at_k_xs, recall_at_k_xs[1:])):
        raise ValueError(f"Recall must be non-decreasing for PR-AUC")
    precision_at_k_xs = [normalized[f"P@{k}"] for k in ks]
    normalized["AUC"] = float(sk_auc(recall_at_k_xs, precision_at_k_xs))
    normalized["MacroPR_AUC"] = macro_pr_auc

    print(f"Model: {model_name}")
    print(f" mAP: \t  {normalized['MAP']:.2f}") # Mean Average Precision
    print(f" MRR*:   {normalized['MRR']:.2f}") # MRR at first hit ranks
    for k in [1, 3, 5]:
        print(f"  Hits@{k}:    {normalized[f'Hits@{k}']:.2f}")
    print(f" Med:     {normalized['Med']:.2f}") # Median Rank
    print(f" MR:      {normalized['MR']:.2f} ") # Mean Rank
    print(f" nDCG@10: {normalized['nDCG@10']:.2f}") # nDCG@10
    print(f" PR-AUC:  {normalized['AUC']:.2f}") # area under precision-recall curve
    print(f" mPR-AUC: {normalized['MacroPR_AUC']:.2f}") # PR-AUC (macro averaged)
    print(f" R@100:   {normalized['R@100']:.2f}") # Recall@100
    print("-"*60)

    all_results[model_name] = normalized

    model_metric_string = model_name.split()

    experiment_table_two.add_row([
      model_metric_string[0],
      model_metric_string[1],
      model_metric_string[2],
      normalized['MAP'], 
      normalized['MRR'],
      normalized['Hits@1'], 
      normalized['Hits@3'], 
      normalized['Hits@5'],
      normalized['Med'], 
      normalized['MR'],
      normalized['nDCG@10'],
      normalized['R@100']
    ])

    macro_avg_PR_AUC_data[model_name] = {
      "recall": R_grid.tolist(),
      "precision": P_macro.tolist()
    }

Path('../logs').mkdir(parents=True, exist_ok=True)
output_file = '../logs/oov_entity_mentions_d_equals_five_results.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"All results saved to {output_file}")

output_macro_pr_auc_file = '../logs/oov_entity_mentions_d_equals_five__PR_AUC_POINTS_4_PLOT.json'
with open(output_macro_pr_auc_file, 'w') as f:
    json.dump(macro_avg_PR_AUC_data, f, indent=2)

print(f"Macro PR AUC plot data dumped to {output_macro_pr_auc_file}")

print(f"Printing table: \n\n")

print(experiment_table_two.draw())

print("\n\n Printing LaTeX: \n\n")

print(latextable.draw_latex(
    table=experiment_table_two,
    caption="Performance of fetching multiple relevant targets (d=5) using OOV mentions with lambda=0.1 & 0.37 (50 Queries)",
    use_booktabs=True, position="H", caption_above=True, caption_short="Multi target performance of OOV mentions, lambda={0.1,0.37}",
    label="tab:multi-target-oov-weighted-5"
  )
)