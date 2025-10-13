import numpy as np
import itertools
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
rng = np.random.default_rng()
import pandas as pd
from dadapy import Data
from dadapy._utils import utils as ut
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial import distance

from sentence_transformers import SentenceTransformer, util
import random
import xml.etree.ElementTree as ET

import openai
import gc
from scipy import stats
import torch
import os
import json
from openai import OpenAI

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

criterias = {
'Paranoid PD' : [
 
 
    "Suspects, without sufficient basis, that others are exploiting, harming, or deceiving him or her.",
    "Is preoccupied with unjustified doubts about the loyalty or trustworthiness of friends or associates.",
...
],
# Schizotypal Personality Disorder Criteria
'Schizotypal' : [
    "Ideas of reference (excluding delusions of reference).",
    "Odd beliefs or magical thinking that influences behavior and is inconsistent with subcultural norms (e.g., superstitiousness, belief in clairvoyance, telepathy, or 'sixth sense'; in children and adolescents, bizarre fantasies or preoccupations).",
    "Unusual perceptual experiences, including bodily illusions.",
    "Odd thinking and speech (e.g., vague, circumstantial, metaphorical, overelaborate, or stereotyped).",
...
],
# Schizoid Personality Disorder Criteria
'Schizoid' : [
    "Neither desires nor enjoys close relationships, including being part of a family.",
    "Almost always chooses solitary activities.",
...
],
# Avoidant Personality Disorder Criteria
'Avoidant' : [
    "Avoids occupational activities that involve significant interpersonal contact because of fears of criticism, disapproval, or rejection.",
    "Is unwilling to get involved with people unless certain of being liked.",
...
],
# Borderline Personality Disorder Criteria
'Borderline PD' : [
    "Frantic efforts to avoid real or imagined abandonment.",
    "A pattern of unstable and intense interpersonal relationships characterized by alternating between extremes of idealization and devaluation.",
...
],
# Obsessive-Compulsive Personality Disorder Criteria
'OCPD' : [
    "Is preoccupied with details, rules, lists, order, organization, or schedules to the extent that the major point of the activity is lost.",
    "Shows perfectionism that interferes with task completion (e.g., is unable to complete a project because his or her own overly strict standards are not met).",
...
],
# Narcissistic Personality Disorder Criteria
'Narcissistic PD' : [
    "Has a grandiose sense of self-importance (e.g., exaggerates achievements and talents, expects to be recognized as superior without commensurate achievements).",
...
],
### Antisocial Personality Disorder Criteria
'Anti Social' : [
'Failure to conform to social norms with respect to lawful behaviors (repeatedly performing acts that are grounds for arrest)',
'Deceitfulness, as indicated by repeated lying, use of aliases, or conning others for personal profit or pleasure',
...
],
### Histrionic Personality Disorder Criteria
'Histrionic' : [
'Is uncomfortable in situations in which they are not the center of attention',
'Interaction with others is often characterized by inappropriate sexually seductive or provocative behavior',
...
],
### Dependent Personality Disorder Criteria
'Dependent' : [
'Has difficulty making everyday decisions without an excessive amount of advice and reassurance from others',
'Needs others to assume responsibility for most major areas of their life',
...
]}


def return_ids_kstar_binomial(data, embeddings, initial_id=None, Dthr=23, r='opt', n_iter = 10):
    if initial_id is None:
        data.compute_id_2NN(algorithm='base')
    else:
        data.compute_distances()
        data.set_id(initial_id)

    ids = np.zeros(n_iter)
    ids_err = np.zeros(n_iter)
    kstars = np.zeros((n_iter, data.N), dtype=int)
    log_likelihoods = np.zeros(n_iter)
    ks_stats = np.zeros(n_iter)
    p_values = np.zeros(n_iter)

    for i in range(n_iter):
      # compute kstar
      data.compute_kstar(Dthr)
      # print("iteration ", i)
      # print("id ", data.intrinsic_dim)

      # set new ratio
      r_eff = min(0.95,0.2032**(1./data.intrinsic_dim)) if r == 'opt' else r
      # compute neighbourhoods shells from k_star
      rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
      rn = rk * r_eff
      n = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)
      # compute id
      id = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
      # compute id error
      id_err = ut._compute_binomial_cramerrao(id, data.kstar-1, r_eff, data.N)
      # compute likelihood
      log_lik = ut.binomial_loglik(id, data.kstar - 1, n - 1, r_eff)
      # model validation through KS test
      n_model = rng.binomial(data.kstar-1, r_eff**id, size=len(n))
      ks, pv = ks_2samp(n-1, n_model)
      # set new id
      data.set_id(id)

      ids[i] = id
      ids_err[i] = id_err
      kstars[i] = data.kstar
      log_likelihoods[i] = log_lik
      ks_stats[i] = ks
      p_values[i] = pv

    data.intrinsic_dim = id
    data.intrinsic_dim_err = id_err
    data.intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())

    return ids, kstars[(n_iter - 1), :]#, ids_err, log_likelihoods, ks_stats, p_values

def find_Kstar_neighs(kstars, embeddings):
    nn = NearestNeighbors(metric = 'cosine', n_jobs=-1)
    nn.fit(embeddings)

    neighs_ind = []
    for i, obs in enumerate(embeddings):
        distance, ind = nn.kneighbors([obs], n_neighbors=kstars[i] + 1)

        k_neighs = ind[0][1:]
        neighs_ind.append(k_neighs.tolist())
    return neighs_ind

def find_single_k_neighs(embeddings, index, k, cosine=True):
    target_embedding = embeddings[index]
    
    if cosine: 
        # Compute cosine distance
        all_distances = np.array([distance.cosine(target_embedding, emb) for emb in embeddings])
        # Sort by ascending order (smaller distance = higher similarity)
        nearest_indices = np.argsort(all_distances)[1:k+1]  
    else:
        # Compute dot product
        all_scores = util.dot_score(target_embedding, embeddings)[0].cpu().tolist()
        # Sort by descending order (larger score = higher similarity)
        nearest_indices = np.argsort(all_scores)[::-1][1:k+1]  

    return nearest_indices.tolist()
def extract_docs(results):

    unique_set = set()
    for sublist in results:
        unique_set.update(sublist)
    unique_docs = list(unique_set)

    return unique_docs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


warnings.filterwarnings('ignore')
set_seed(0)
model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
cosine = True
def get_embedding(text, model):
    return model.encode(text)


predictions = {}
for user in users:
    file = 'jsons_final_evidences/' + user + '.json'
    print(file)

    with open(file, 'r', encoding='utf-8') as file:
        final_diz = json.load(file)
    posts = [post for post in final_diz['evidences']] #+ final_diz['no_evidences'] 
    diagnoses = [diagnosis for diagnosis in list(diz[user].keys())]
    doc_embeddings = get_embedding(posts, model)
    predictions[user] = {}
    
    for diagn in diagnoses:
        print(diagn)
        pred_criteria = []
        
        for i in range(len(criterias[diagn])):
            if len(posts) > 10:
                item = model.encode(criterias[diagn][i])
                embs = np.concatenate((np.array(item).reshape(1, -1), doc_embeddings))

                data = Data(embs)
                try:
                    ids, kstars = return_ids_kstar_binomial(
                        data, 
                        doc_embeddings, 
                        initial_id=None, 
                        Dthr=23.928, 
                        r='opt', 
                        n_iter=10
                    )
                    nns = find_single_k_neighs(embs, 0, kstars[0], cosine)
                    docs_retrieved = np.array(posts)[np.array(nns) - 1].tolist()
                except ValueError as e:
                    print(f"Error in return_ids_kstar_binomial: {e}")
                    # fallback -> usa direttamente i posts
                    docs_retrieved = posts
            else:
                docs_retrieved = posts

            llm = client.chat.completions.create(
                model='openai/gpt-4o-mini', 
                messages=[
                    {"role": "user","content": f""""Analyze the provided Reddit posts against a specific SCID-5-PD diagnostic criteria listed below.

                    # Diagnostic Criterion for {diagn}
                    {criterias[diagn][i]}

                    # Scoring
                    For the criterion above, assign:
                    - **0** = No evidence OR subthreshold evidence (the behavior/pattern is absent, unclear, or does not meet the threshold for clinical significance)
                    - **1** = Clear evidence (the posts demonstrate a consistent, unambiguous pattern that meets the criterion)

                    # Rules
                    - Base assessment only on explicit post content
                    - Look for consistent patterns, not isolated incidents
                    - Consider context (jokes vs. genuine expressions)

                    # Reddit Posts
                    {docs_retrieved}

                    # Output
                    Return ONLY a scores 0/1
                    """}
                ], 
                temperature=0.0, 
                max_tokens=20
            )
            pred_criteria.append(llm.choices[0].message.content)
        
        predictions[user][diagn] = pred_criteria
        
