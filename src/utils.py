from ast import IsNot
import json
import math
import itertools
from glob import glob
import time
from openai import OpenAI
from config_loader import get_effective_api_config

# Initialize OpenAI client (v1+ SDK) - only used for subjective evaluation
api_config = get_effective_api_config()
client = OpenAI(
    api_key=api_config['api_key'],
    base_url=api_config['base_url'],
)

PROMPT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg}[/INST]"

def get_prompt(source, query):
    system_prompt = """You are a helpful, respectful and honest assistant.
Given a web source, and context, your only purpose is to summarize the source, and extract topics that may be relevant to the context. Even if a line is distinctly relevant to the context, include that in the summary. It is preferable to pick chunks of text, instead of isolated lines.
"""

    user_msg = f"### Context: ```\n{query}\n```\n\n ### Source: ```\n{source}\n```\n Now summarize the text in more than 1000 words, keeping in mind the context and the purpose of the summary. Be as detailed as possible.\n"

    return PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_msg=user_msg)

import re
import nltk



def get_num_words(line):
    return len([x for x in line if len(x)>2])

def extract_citations_new(text):
    def ecn(sentence):
        citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'

        return [int(re.findall(r'\d+', citation)[0]) for citation in re.findall(citation_pattern, sentence)]

    paras = re.split(r'\n\n', text)

    # Split each paragraph into sentences
    sentences = [nltk.sent_tokenize(p) for p in paras]

    # Split each sentence into words
    words = [[(nltk.word_tokenize(s), s, ecn(s)) for s in sentence] for sentence in sentences]
    return words

def impression_wordpos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])

            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores

def impression_word_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
                

def impression_para_based(sentences, n = 5, normalize = True, alpha = 1.1, beta = 1.5, gamma2 = 1/math.e):
    scores = [0 for _ in range(n)]
    power_scores = [1 for _ in range(n)]
    average_lines = sum([len(x) for x in sentences])/len(sentences)
    for i, para in enumerate(sentences):
        citation_counts = [0 for _ in range(n)]
        for sent in para:
            for c in sent[2]:
                try:
                    citation_counts[c-1] += get_num_words(sent[0])
                except Exception as e:
                    print(f"Citation Hallucinated: {c}")
        if sum(citation_counts)==0:
            continue
        
        for cit_num, cit in enumerate(citation_counts):
            if cit==0: continue
            score = cit/sum(citation_counts)
            
            score *= beta**(len(para)/average_lines - 1)
            
            if i == 0:
                score *= 1
            elif i != len(sentences)-1:
                score *= math.exp(-1 * i / (len(sentences)-2))
            else:
                score *= gamma2

            try:
                power_scores[cit_num] *= (alpha) ** (cit/sum(citation_counts))
                scores[cit_num] += score
            except:
                print(f'Citation Hallucinated: {cit}')
         
    final_scores = [x*y for x, y in zip(scores, power_scores)]
    return [x/sum(final_scores) for x in final_scores] if normalize and sum(final_scores)!=0 else final_scores


subj_cache_file = None
def impression_subjpos_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'subjpos_detailed')

def impression_diversity_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'diversity_detailed')

def impression_uniqueness_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'uniqueness_detailed')

def impression_follow_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'follow_detailed')

def impression_influence_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'influence_detailed')

def impression_relevance_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'relevance_detailed')

def impression_subjcount_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'subjcount_detailed')
    
def impression_subjective_impression(sentences, query, n = 5, normalize = True, idx = 0, metric = 'subjective_impression'):
    # print(hash((sentences, query, n, idx)))
    # 3/0
    def returnable_score_from_scores(scores):
        avg_score = sum(scores.values())/len(scores.values())
        if metric != 'subjective_impression':
            avg_score = scores[metric]
        return [avg_score if _==idx else 0 for _ in range(n)]

    
    global subj_cache_file
    cache_file = 'gpt-eval-scores-cache_new-new.json'

    if os.environ.get('SUBJ_STATIC_CACHE', None) is not None:
        if subj_cache_file is None:
            try:
                subj_cache_file = json.load(open(cache_file, encoding='utf-8'), strict=False)
            except json.JSONDecodeError as e:
                print(f"Warning: Subjective cache corrupted at {e}. Creating new cache.")
                subj_cache_file = dict()
    else:
        if os.path.exists(cache_file):
            try:
                subj_cache_file = json.load(open(cache_file, encoding='utf-8'), strict=False)
            except json.JSONDecodeError as e:
                print(f"Warning: Subjective cache corrupted at {e}. Creating new cache.")
                subj_cache_file = dict()
        else:
            subj_cache_file = dict()
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(subj_cache_file, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error creating subjective cache file: {e}")
    cache = subj_cache_file
    # TODO: Fix str(idx) issue
    # from pdb import set_trace
    if str((sentences, query)) in cache:
        if str(idx) in cache[str((sentences, query))]:
            print('Okay we have a hit!')
            # new_scores = []
            # for idx in range(5):
            #     sc = cache[str((sentences, query))][str(idx)]
            #     new_scores.append(sum(sc.values())/len(sc.values()))
            # return [x/sum(new_scores) for x in new_scores] if normalize else new_scores
            return returnable_score_from_scores(cache[str((sentences, query))][str(idx)])
    # TODO: If we don't have a hit, fine, just return 0 or something
    # set_trace()
    return [0 if _==idx else 0 for _ in range(n)]
    def convert_to_number(x, min_val = 1.0):
        try: return max(min(5, float(x)), min_val)
        except: return min_val
    scores = dict()
    for prompt_file in glob('geval_prompts/*.txt'):
        prompt = open(prompt_file).read()
        prompt = prompt.replace('[1]',f'[{idx+1}]')
        cur_prompt = prompt.format(query = query, answer = sentences)
        while True:
            try:
                _response = client.completions.create(
                    model='gpt-3.5-turbo-instruct',
                    prompt=cur_prompt,
                    temperature=0.0,
                    max_tokens=3,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    logprobs=5,
                    n=1,
                )
                # print(_response.usage)
                # time.sleep(0.5)
                logprobs = _response['choices'][0]['logprobs']['top_logprobs'][0]
                total_sum = sum([((math.e)**v) for v in logprobs.values()])
                avg_score = sum([convert_to_number(k) * ((math.e)**v)/total_sum for k,v in logprobs.items()])
                scores[os.path.split(prompt_file)[-1].split('.')[0]] = avg_score
                break
            except Exception as e:
                print('Error in GPT-Eval', e)
                time.sleep(10)
    avg_score = sum(scores.values())/len(scores.values())
    try:
        cache = json.load(open(cache_file, encoding='utf-8'), strict=False)
    except json.JSONDecodeError as e:
        print(f"Warning: Cache corrupted at {e}. Creating new cache.")
        cache = {}
    if str((sentences, query)) not in cache:
        cache[str((sentences, query))] = dict()
    cache[str((sentences, query))][idx] = scores
    # Use safe file writing with proper file handle closure
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing subjective cache: {e}")
    return returnable_score_from_scores(scores)

import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use environment variable if set, otherwise use path relative to script directory
CACHE_FILE = os.environ.get('GLOBAL_CACHE_FILE', os.path.join(SCRIPT_DIR, 'global_cache.json'))


def _safe_write_cache(cache: dict, path: str = CACHE_FILE) -> None:
    """Safely write cache JSON to disk using atomic write (temp file + rename).
    If writing fails, does not interrupt main process.
    
    Note: This now only writes sources data, not LLM responses.
    LLM responses are kept in memory only to prevent cache corruption.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Use atomic write: write to temp file first, then rename
        # This prevents corruption if process is interrupted during write
        temp_path = path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        # Atomic rename (works on Windows and Unix)
        if os.path.exists(path):
            os.replace(temp_path, path)
        else:
            os.rename(temp_path, path)
    except (OSError, TypeError, ValueError) as e:
        # Clean up temp file if it exists
        temp_path = path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        # Don't let cache write errors crash the program, just print warning
        print(f"Warning: failed to write cache to {path}: {e}")


# Note: Do NOT auto-create empty cache file
# Users should preload cache using: python src/preload_cache_from_geobench.py
# if not os.path.exists(CACHE_FILE):
#     _safe_write_cache({})

from search_try import search_handler
from generative_le import generate_answer

def check_summaries_exist(sources, summaries):
    for source in sources:
        s2 = [x['summary'] for x in source['sources']]  
        if s2 == summaries:
            return source
    return None

def get_answer(query, summaries = None, n = 5, num_completions = 1, cache_idx = 0, regenerate_answer = False, write_to_cache = True, loaded_cache = None):
    """Get answer for a query using cached sources and real-time LLM generation.
    
    Args:
        query: The query string
        summaries: Optional list of summaries to use instead of cached sources
        n: Number of sources to fetch
        num_completions: Number of LLM completions to generate
        cache_idx: Index in cache to use
        regenerate_answer: Whether to regenerate the answer (deprecated - always regenerates now)
        write_to_cache: Whether to write sources to cache (responses are never cached)
        loaded_cache: Pre-loaded cache dictionary
        
    Returns:
        Dict with 'sources' and 'responses' keys. Responses are generated fresh each time.
        
    Note: LLM responses are no longer cached to prevent corruption from concurrent writes.
    Only the sources (from GEO-Bench dataset) are cached.
    """
    # print(CACHE_FILE, query)
    if loaded_cache is None:    
        try:
            cache = json.load(open(CACHE_FILE, encoding='utf-8'), strict=False)
        except json.JSONDecodeError as e:
            print(f"CRITICAL ERROR: Cache file {CACHE_FILE} is corrupted at {e}")
            print(f"The cache file contains invalid JSON and cannot be loaded.")
            print(f"Please restore from backup or regenerate: python src/preload_cache_from_geobench.py")
            raise RuntimeError(f"Cache file corrupted. Cannot proceed without valid cache.")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Cache file {CACHE_FILE} not found")
            print(f"Please preload cache first: python src/preload_cache_from_geobench.py")
            raise RuntimeError(f"Cache file not found. Please preload cache first.")
    else: cache = loaded_cache
    if summaries is None:
        if cache.get(query) is None:
            search_results = search_handler(query, source_count = n)
            if loaded_cache is None:    
                try:
                    cache = json.load(open(CACHE_FILE, encoding='utf-8'), strict=False)
                except json.JSONDecodeError as e:
                    print(f"Warning: Cache file corrupted at {e}. Using empty cache.")
                    cache = {}
            else: cache = loaded_cache
            cache[query] = [{'sources': search_results['sources'], 'responses': []}]
            if write_to_cache:
                _safe_write_cache(cache)
        else:
            search_results = cache[query][cache_idx]

        summaries = [x['summary'] for x in search_results['sources']]
        # Check if summaries is empty - if so, try to re-search
        if len(summaries) == 0:
            print(f'Warning: No sources found in cache for query "{query}". Attempting to re-search...')
            search_results = search_handler(query, source_count = n)
            if loaded_cache is None:    
                try:
                    cache = json.load(open(CACHE_FILE, encoding='utf-8'), strict=False)
                except json.JSONDecodeError as e:
                    print(f"Warning: Cache file corrupted at {e}. Using empty cache.")
                    cache = {}
            else: cache = loaded_cache
            summaries = [x['summary'] for x in search_results['sources']]
            if len(summaries) == 0:
                raise ValueError(f'No sources found for query "{query}" after search. Please check search_handler function.')
            # Update cache with new search results
            cache[query] = [{'sources': search_results['sources'], 'responses': []}]
            if write_to_cache:
                _safe_write_cache(cache)
    # Generate answers in real-time without caching LLM responses
    # This prevents cache corruption from concurrent writes and keeps the system stateless
    print('Generating answer from LLM (responses not cached)')
    answers = generate_answer(query, summaries, num_completions = num_completions)
    
    # Return a structure compatible with the existing code
    # Note: 'responses' field is populated but NOT written to disk cache
    return {
        'sources': cache[query][cache_idx]['sources'] if cache.get(query) else [],
        'responses': [answers]
    } 