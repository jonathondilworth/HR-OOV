from __future__ import annotations
from typing import Iterable
from hierarchy_transformers import HierarchyTransformer # type: ignore # <- temporary (TODO)
from OnT.OnT import OntologyTransformer
import numpy as np
import torch
import math


def identity(x):
    return x


# specific to calculating nDCG
def add(a, b, key='dcg'):
    return a + b[key]


def batch_euclidian_l2_distance(u: np.ndarray, vs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(u - vs, axis=1)


def l2_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.sum(x**2))


def batch_l2_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.asarray(np.sqrt(np.sum(x**2, axis=1)))


def inner_product(p_u: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    u = np.asarray(p_u, dtype=np.float32)
    v = np.asarray(p_v, dtype=np.float32)
    return np.inner(u, v)


def batch_inner_product(p_u: np.ndarray, p_vs: np.ndarray) -> np.ndarray:
    u = np.asarray(p_u, dtype=np.float32).ravel()
    vs = np.asarray(p_vs, dtype=np.float32)
    return vs.dot(u)


def cosine_similarity(u, v, normalised=True):
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return np.inner(u, v) if normalised else np.inner(u, v) / (l2_norm(u) * l2_norm(v))


def batch_cosine_similarity(p_u, p_vs, normalised=True):
    u  = np.asarray(p_u,  dtype=np.float32)
    vs = np.asarray(p_vs, dtype=np.float32)
    return batch_inner_product(u, vs) if normalised else batch_inner_product(u, vs) / (l2_norm(u) * batch_l2_norm(vs))


def efficient_batch_poincare_distance_with_curv_k(u: np.ndarray, vs: np.ndarray, k: np.float32 | np.float64):
    '''
    Temporary notes (resources):
      1. np contiguous arrays: https://medium.com/@heyamit10/understanding-numpy-ascontiguousarray-with-practical-examples-a71d639fe65a
      2. linalg GEMV: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html
      3. CUDA-based distance calc: https://stackoverflow.com/questions/29752994/computing-all-pairs-distances-between-points-in-different-sets-with-cuda
    '''
    vs = np.ascontiguousarray(vs, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    eps_offset = 1e-7
    # more efficient than element-wise ops:
    u_norm_sqd = np.dot(u, u)
    vs_norm_sqd = np.matmul(vs, vs)
    # see (2,3): ||x-y||^2 = ||x||^2+||y||^2-2*<x,y>
    denom_term_A = 1 - (k * u_norm_sqd + eps_offset)
    denom_term_B = 1 - (k * vs_norm_sqd + eps_offset)
    vs_u = vs @ u 
    l2_dist_sqd = vs_norm_sqd + u_norm_sqd - 2.0 * vs_u
    # acosh, bound to [1, inf)
    acosh_arg = np.maximum((1.0 + (2.0 * k * l2_dist_sqd) / (denom_term_A * denom_term_B)), 1.0)
    scaling_factor_k = 1.0 / np.sqrt(k)
    return scaling_factor_k * np.arccosh(acosh_arg)


def batch_poincare_distance_with_curv_k(u: np.ndarray, vs: np.ndarray, k: np.float64 | np.float32) -> np.float64 | np.float32:
    u_norm_sqd = np.sum(u**2)
    vs_norms_sqd = np.sum(vs**2, axis=1)
    l2_dist_sqd = np.sum((u - vs)**2, axis=1)
    offset = 1e-7 # eps guard
    arg = 1 + ((2 * k * l2_dist_sqd) / ((1 - (k * u_norm_sqd + offset)) * (1 - (k * vs_norms_sqd + offset)))) # acosh
    arg = np.maximum(1.0, arg) # bounds check
    acosh_scaling = np.float64(1) / np.float64(np.sqrt(k)) # kappa
    return (acosh_scaling * np.arccosh(arg, dtype=np.float64)) # 1 / sqrt(k) * acosh(arg)


def batch_poincare_distance_with_curv_k_torch(u: np.ndarray, vs: np.ndarray, k: float, * , device: str | None = None, dtype: torch.dtype | None = None, batch_size: int | None = None) -> np.ndarray:
    '''
    defaults device to "cuda:0"
    defaults dtype to torch.float32
    defaults batch_size to the size of the vs array being passed in

    TODO: migrate default arguments to function signature where possible
    '''
    
    # defaults to single GPU
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # defaults to float32
    if dtype is None:
        dtype = torch.float32
    # defaults to the size of the 'embedding store' (assuming vs := embedding store)
    if batch_size is None or batch_size <= 0:
       batch_size = len(vs)
    
    eps = 1e-7 # eps guard
    scaling_factor_k = 1.0 / math.sqrt(k)
    u_t = torch.as_tensor(u, device=device, dtype=dtype)
    u_norm_sqd = (u_t * u_t).sum()
    outs = []
    
    with torch.inference_mode():
        # reflects prior implementation via torch
        for start in range(0, len(vs), batch_size):
            vs_chunk = vs[start:start + batch_size]
            vs_t = torch.as_tensor(vs_chunk, device=device, dtype=dtype)
            vs_norms_sqd = (vs_t * vs_t).sum(dim=1)
            l2_dist_sqd = ((vs_t - u_t) ** 2).sum(dim=1)
            denom = (1.0 - (k * u_norm_sqd + eps)) * (1.0 - (k * vs_norms_sqd + eps))
            arg = 1.0 + (2.0 * k * l2_dist_sqd) / denom
            arg = torch.clamp_min(arg, 1.0)
            d = scaling_factor_k * torch.acosh(arg)
            outs.append(d.detach().to("cpu"))

    return torch.cat(outs, dim=0).numpy()


def batch_poincare_dist_with_adaptive_curv_k(u: np.ndarray, vs:np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    # TODO: accept curvature kappa as an argument, rather than the models themselves
    if isinstance(model, HierarchyTransformer):
        k = np.float64(model.get_circum_poincareball(model.embed_dim).c)
    elif isinstance(model, OntologyTransformer):
        hierarchy_model = model.hit_model
        hierarchy_poincare_ball = hierarchy_model.get_circum_poincareball(hierarchy_model.embed_dim)
        k = np.float64(hierarchy_poincare_ball.c)    
    else:
        raise Exception("Hyperbolic distance should only be only calculated in B^n")
    return np.asarray(batch_poincare_distance_with_curv_k(u, vs, k))


def batch_poincare_dist_with_adaptive_curv_k_torch(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    if not torch.cuda.is_available():
       print("Warning: explicit call to torch-based batch poincare dist fn, but CUDA is not available.")
       batch_poincare_dist_with_adaptive_curv_k(u, vs, model=model, **kwargs)
    # else: (CUDA is available)
    if isinstance(model, HierarchyTransformer):
        k = np.float64(model.get_circum_poincareball(model.embed_dim).c)
    elif isinstance(model, OntologyTransformer):
        hierarchy_model = model.hit_model
        hierarchy_poincare_ball = hierarchy_model.get_circum_poincareball(hierarchy_model.embed_dim)
        k = np.float64(hierarchy_poincare_ball.c)
    else:
        raise Exception("Hyperbolic distance should only be only calculated in B^n or H^n")
    # happy path:
    device = kwargs.get("device", "cuda:0")
    dtype = kwargs.get("dtype", torch.float32)
    batch_size = kwargs.get("batch_size", None) # => batch_size := len(vs)
    return np.asarray(batch_poincare_distance_with_curv_k_torch(u, vs, float(k), device=device, dtype=dtype, batch_size=batch_size))


def gpu_cpu_batch_poincare_dist_with_adaptive_curv_k(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer | OntologyTransformer, **kwargs):
    if torch.cuda.is_available():
        return batch_poincare_dist_with_adaptive_curv_k_torch(u, vs, model, **kwargs)
    # else:
    return batch_poincare_dist_with_adaptive_curv_k(u, vs, model, **kwargs)
    

def subsumption_score_hit(hit_transformer: HierarchyTransformer, child_emb: np.ndarray | torch.Tensor, parent_emd: np.ndarray | torch.Tensor, centri_weight: float = 1.0):
    child_emb_t = torch.Tensor(child_emb)
    parent_emb_t = torch.Tensor(parent_emd)
    dists = hit_transformer.manifold.dist(child_emb_t, parent_emb_t)
    child_norms = hit_transformer.manifold.dist0(child_emb_t)
    parent_norms = hit_transformer.manifold.dist0(parent_emb_t)
    return -(dists + centri_weight * (parent_norms - child_norms))


def subsumption_score_ont(ontology_transformer: OntologyTransformer, child_emb: np.ndarray | torch.Tensor, parent_emb: np.ndarray | torch.Tensor, weight_lambda: float = 1.0):
    child_emb_t = torch.Tensor(child_emb)
    parent_emb_t = torch.Tensor(parent_emb)
    return ontology_transformer.score_hierarchy(child_emb_t, parent_emb_t, weight_lambda)


def entity_subsumption(u: np.ndarray, vs: np.ndarray, model: HierarchyTransformer, *, weight: float = 0.4, **kwargs):
    return np.asarray(subsumption_score_hit(model, u, vs, centri_weight=weight))


def concept_subsumption(u: np.ndarray, vs: np.ndarray, model: OntologyTransformer, *, weight: float = 0.4, **kwargs):
    return np.asarray(subsumption_score_ont(model, u, vs, weight_lambda=weight))


def dcg_exp_relevancy_at_pos(relevancy: int, rank_position: int) -> float:
    if relevancy <= 0:
      return float(0.0)
    numerator = (2**relevancy) - 1
    denominator = math.log2(rank_position + 1)
    return float(numerator / denominator)


def compute_ndcg_at_k(results: list[tuple[int, str, float, str]], targets_with_dcg_exp: list[dict], k: int = 20) -> float:
    relevance_map = {target['iri']: target['relevance'] for target in targets_with_dcg_exp}
    dcg = 0.0
    for rank, (idx, iri, score, label) in enumerate(results[:k], start=1):
        rel = relevance_map.get(iri, 0)
        dcg += dcg_exp_relevancy_at_pos(rel, rank)
    ideal_dcg = sum(target['dcg'] for target in targets_with_dcg_exp[:k])
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg

