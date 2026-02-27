from utils import get_answer, extract_citations_new, impression_subjective_impression, impression_wordpos_count_simple, impression_subjpos_detailed, impression_diversity_detailed, impression_uniqueness_detailed, impression_follow_detailed, impression_influence_detailed, impression_relevance_detailed, impression_subjcount_detailed, impression_pos_count_simple, impression_word_count_simple
from typing import List, Tuple, Dict, Any
import numpy as np
import json
from geo_functions import *
import sys
import time
import os
from datasets import load_dataset
import nltk
import asyncio
from tqdm import tqdm
import shutil
from datetime import datetime
from config_loader import use_local_llm, get_local_llm_model
import re

# Surrogate model selection:
# Always use evolved surrogate (strategies + trained Critic)
USE_EVOLVED_SURROGATE = True
from surrogate_model_evolved import StrategySelector, get_strategy_prompts, load_evolved_strategies
print("[INFO] Using EVOLVED strategies + trained Critic from agentic_geo")

def format_number_7decimals(num):
	"""Format number to 7 decimal places."""
	if isinstance(num, (int, float)):
		return round(float(num), 7)
	elif isinstance(num, (list, tuple, np.ndarray)):
		return [format_number_7decimals(x) for x in num]
	else:
		return num

def format_scores_7decimals(scores):
	"""Format scores (can be nested lists/arrays) to 7 decimal places."""
	if isinstance(scores, (list, tuple, np.ndarray)):
		return [format_scores_7decimals(x) if isinstance(x, (list, tuple, np.ndarray)) else format_number_7decimals(x) for x in scores]
	else:
		return format_number_7decimals(scores)

nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')) 

# ============================================================
# Evolved Strategy Rewrite Function
# ============================================================

def _extract_rewritten_content(raw: str) -> str:
	"""Best-effort extractor to return ONLY rewritten content.
	
	Some evolved strategies include an output format like:
	- "First output your plan, then the final text"
	- "First: list changes"
	- "Hard words: ..."
	
	This function strips such wrappers when present.
	"""
	if raw is None:
		return ""
	text = raw.strip()
	if not text:
		return text

	# Prefer content after a "Final Text" marker if present.
	m = re.search(r"(?is)^\s*#{1,6}\s*final\s*text\s*$", text, flags=re.MULTILINE)
	if m:
		text = text[m.end():].strip()
	else:
		m2 = re.search(r"(?is)\bfinal\s*text\b\s*[:\-]?\s*", text)
		if m2:
			# take the last occurrence to avoid "plan ... final text" earlier
			all_m = list(re.finditer(r"(?is)\bfinal\s*text\b\s*[:\-]?\s*", text))
			if all_m:
				text = text[all_m[-1].end():].strip()

	# If there's a code block, prefer the last fenced block as "final".
	fences = list(re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)\n```", text))
	if fences:
		candidate = fences[-1].group(1).strip()
		if len(candidate) > 0:
			text = candidate

	# Remove trailing "Hard words:" section if present.
	hw = re.search(r"(?is)\n\s*hard\s+words\s*:\s*", text)
	if hw:
		text = text[:hw.start()].strip()

	# Remove leading "Modifications Plan" section if still present.
	# If both plan and final exist, final extraction above should handle it.
	plan = re.search(r"(?is)^\s*#{1,6}\s*modifications?\s*plan\s*$", text, flags=re.MULTILINE)
	if plan and ("final text" not in text.lower()):
		# drop everything up to the first blank line after plan header
		after = text[plan.end():]
		blank = re.search(r"\n\s*\n", after)
		if blank:
			text = after[blank.end():].strip()

	return text

def apply_evolved_strategy(query: str, content: str, strategy_prompt: str) -> str:
	"""Apply an evolved strategy to rewrite content using LLM.
	
	This function implements the Oracle Rewrite logic for evolved strategies.
	The strategy_prompt should be the FULL_PROMPT from evolved strategies.
	
	Args:
		query: User query (for context)
		content: Original content to rewrite
		strategy_prompt: Full strategy prompt (from evolved strategies)
	
	Returns:
		Rewritten content
	"""
	# Build the rewrite prompt (matching agentic_geo's RewriteTool format)
	user_prompt = f"""## Query (for context)
{query}

## Original Source Content
{content}

## Optimization Strategy
{strategy_prompt}

## Instructions
1. Apply the optimization strategy to rewrite the source content.
2. The rewritten content should better answer the query while following the strategy.
3. Do NOT fabricate facts, citations, or statistics unless the strategy explicitly allows it.
4. Preserve the original structure (paragraphs, bullets, etc.) unless the strategy requires changes.
5. CRITICAL OUTPUT RULE: Output ONLY the rewritten content.
   - Do NOT output any plan, analysis, list of changes, or section headers like "Modifications Plan"/"Final Text".
   - Do NOT output extra sections like "Hard words:" or metadata.
   - If the strategy's Output Format conflicts with this rule (e.g. asks for a plan), IGNORE that part and still output only the rewritten content."""

	system_prompt = """You are an expert SEO/GEO content optimizer. Your task is to apply the given optimization strategy to rewrite the source content. Follow the strategy instructions precisely while maintaining the core information and meaning of the original content."""
	
	# Use the same call_gpt function as other GEO methods
	result = call_gpt(user_prompt, system_prompt=system_prompt)
	return _extract_rewritten_content(result)


def identity(summary, source=None):

	return summary

IMPRESSION_FNS = {
	'simple_wordpos' : impression_wordpos_count_simple, 
	'simple_word' : impression_word_count_simple,
	'simple_pos' : impression_pos_count_simple,
	'subjective_score' : impression_subjective_impression,
	'subjpos_detailed' : impression_subjpos_detailed,
	'diversity_detailed' : impression_diversity_detailed,
	'uniqueness_detailed' : impression_uniqueness_detailed,
	'follow_detailed' : impression_follow_detailed,
	'influence_detailed' : impression_influence_detailed,
	'relevance_detailed' : impression_relevance_detailed,
	'subjcount_detailed' : impression_subjcount_detailed,
}


GEO_METHODS = {
	'identity' : identity,
	'fluent_gpt' : fluent_optimization_gpt,
	'unique_words_gpt' : unique_words_optimization_gpt,
	'authoritative_mine' : authoritative_optimization_mine,
	'more_quotes_mine' : more_quotes_mine,
	'citing_credible_mine': citing_credible_sources_mine,
	'simple_language_mine': simple_language_mine,
	'technical_terms_mine' : technical_terms_mine,
	'stats_optimization_gpt' : stats_optimization_mine,
	'seo_optimize_mine2' : seo_optimize_mine2,
}

EXTRACTIVE = False
loaded_cache = None
LAST_UPDATE_TIME = time.time()

# Global variable for results file path
RESULTS_FILE_PATH = None

# Global variable for surrogate model
strategy_selector = None

def get_model_name() -> str:
	"""Get the current LLM model name for file naming."""
	if use_local_llm():
		model = get_local_llm_model()
	else:
		model = os.environ.get('MODEL_NAME', 'gpt-3.5-turbo-16k')
	# Clean model name for file naming (remove special characters)
	clean_model = model.replace('/', '_').replace(':', '_').replace('.', '_')
	return clean_model

def init_results_file(dataset_split: str = None) -> str:
	"""Initialize results file with model name."""
	model_name = get_model_name()
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
	os.makedirs(results_dir, exist_ok=True)
	
	# Include dataset split in filename if specified
	if dataset_split:
		file_path = os.path.join(results_dir, f'geo_results_{model_name}_{dataset_split}.json')
	else:
		file_path = os.path.join(results_dir, f'geo_results_{model_name}.json')
	
	# Initialize empty JSON file
	with open(file_path, 'w', encoding='utf-8') as f:
		json.dump({'results': [], 'metadata': {
			'model': model_name,
			'timestamp': timestamp,
			'start_time': datetime.now().isoformat()
		}}, f, indent=2, ensure_ascii=False)
	
	print(f"\n[Results] Saving results to: {file_path}\n")
	return file_path


def verify_and_fix_results_file(file_path: str) -> Tuple[bool, str]:
	"""Verify the final results file is valid JSON and fix if possible.
	
	Args:
		file_path: Path to the results file
	
	Returns:
		Tuple of (is_valid, message)
	"""
	print(f"\n[Verify] Checking results file integrity...")
	
	if not os.path.exists(file_path):
		return False, f"File does not exist: {file_path}"
	
	try:
		# Try to load the file
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()
		
		# Try to parse JSON
		data = json.loads(content)
		
		# Validate structure
		if not isinstance(data, dict):
			return False, "Root is not a dictionary"
		
		if 'results' not in data:
			return False, "Missing 'results' key"
		
		if 'metadata' not in data:
			print("[Verify] Warning: Missing 'metadata' key, adding it...")
			data['metadata'] = {
				'model': get_model_name(),
				'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
				'repaired': True
			}
		
		if not isinstance(data['results'], list):
			return False, "'results' is not a list"
		
		# Validate each result
		valid_results = []
		for i, result in enumerate(data['results']):
			try:
				# Check if result is a dict
				if not isinstance(result, dict):
					print(f"[Verify] Warning: Result {i} is not a dict, skipping...")
					continue
				
				# Check required fields
				required_fields = ['query', 'strategy_name']
				if all(field in result for field in required_fields):
					valid_results.append(result)
				else:
					print(f"[Verify] Warning: Result {i} missing required fields, skipping...")
			except Exception as e:
				print(f"[Verify] Warning: Error validating result {i}: {e}")
		
		# If we filtered out any invalid results, save the cleaned version
		if len(valid_results) != len(data['results']):
			print(f"[Verify] Cleaned {len(data['results']) - len(valid_results)} invalid results")
			data['results'] = valid_results
			data['metadata']['total_records'] = len(valid_results)
			data['metadata']['cleaned'] = True
			
			# Save cleaned version
			backup_path = file_path + '.original'
			shutil.copy2(file_path, backup_path)
			print(f"[Verify] Original saved to: {backup_path}")
			
			with open(file_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2, ensure_ascii=False)
			print(f"[Verify] Cleaned file saved")
		
		# Final validation
		with open(file_path, 'r', encoding='utf-8') as f:
			json.load(f)
		
		print(f"[Verify] ✓ File is valid JSON with {len(data['results'])} records")
		return True, f"Valid with {len(data['results'])} records"
		
	except json.JSONDecodeError as e:
		# JSON is corrupted, try to fix
		print(f"[Verify] ERROR: File is corrupted at {e}")
		
		# Try to load from backup
		backup_path = file_path + '.backup'
		if os.path.exists(backup_path):
			try:
				print(f"[Verify] Attempting recovery from backup...")
				with open(backup_path, 'r', encoding='utf-8') as f:
					backup_data = json.load(f)
				
				# Save backup as main file
				corrupted_path = file_path + '.corrupted'
				shutil.copy2(file_path, corrupted_path)
				print(f"[Verify] Corrupted file saved to: {corrupted_path}")
				
				shutil.copy2(backup_path, file_path)
				print(f"[Verify] ✓ Recovered from backup")
				return True, f"Recovered from backup with {len(backup_data.get('results', []))} records"
			except Exception as backup_error:
				return False, f"Backup recovery failed: {backup_error}"
		else:
			return False, f"No backup available for recovery"
	
	except Exception as e:
		return False, f"Unexpected error: {str(e)}"

def validate_json_serializable(obj: Any, path: str = "root") -> Tuple[bool, str]:
	"""Validate that an object can be JSON serialized.
	
	Args:
		obj: Object to validate
		path: Current path in object tree (for error reporting)
	
	Returns:
		Tuple of (is_valid, error_message)
	"""
	try:
		if obj is None or isinstance(obj, (bool, int, float, str)):
			return True, ""
		elif isinstance(obj, (list, tuple)):
			for i, item in enumerate(obj):
				is_valid, error = validate_json_serializable(item, f"{path}[{i}]")
				if not is_valid:
					return False, error
			return True, ""
		elif isinstance(obj, dict):
			for key, value in obj.items():
				if not isinstance(key, str):
					return False, f"Non-string key at {path}.{key}"
				is_valid, error = validate_json_serializable(value, f"{path}.{key}")
				if not is_valid:
					return False, error
			return True, ""
		elif isinstance(obj, np.ndarray):
			return validate_json_serializable(obj.tolist(), path)
		elif hasattr(obj, '__dict__'):
			return False, f"Non-serializable object at {path}: {type(obj)}"
		else:
			# Try to serialize it
			try:
				json.dumps(obj)
				return True, ""
			except:
				return False, f"Non-serializable value at {path}: {type(obj)}"
	except Exception as e:
		return False, f"Validation error at {path}: {str(e)}"


def sanitize_string(s: str) -> str:
	"""Sanitize string to prevent JSON encoding issues.
	
	Args:
		s: String to sanitize
	
	Returns:
		Sanitized string safe for JSON encoding
	"""
	if not isinstance(s, str):
		return s
	
	# Replace problematic characters that might cause JSON issues
	# These are already handled by json.dumps, but we do extra cleaning for safety
	replacements = {
		'\x00': '',  # Null byte
		'\x01': '',  # Start of heading
		'\x02': '',  # Start of text
		'\x03': '',  # End of text
		'\x04': '',  # End of transmission
		'\x05': '',  # Enquiry
		'\x06': '',  # Acknowledge
		'\x07': '',  # Bell
		'\x08': '',  # Backspace
		'\x0b': '',  # Vertical tab
		'\x0c': '',  # Form feed
		'\x0e': '',  # Shift out
		'\x0f': '',  # Shift in
		'\x10': '',  # Data link escape
		'\x11': '',  # Device control 1
		'\x12': '',  # Device control 2
		'\x13': '',  # Device control 3
		'\x14': '',  # Device control 4
		'\x15': '',  # Negative acknowledge
		'\x16': '',  # Synchronous idle
		'\x17': '',  # End of transmission block
		'\x18': '',  # Cancel
		'\x19': '',  # End of medium
		'\x1a': '',  # Substitute
		'\x1b': '',  # Escape
		'\x1c': '',  # File separator
		'\x1d': '',  # Group separator
		'\x1e': '',  # Record separator
		'\x1f': '',  # Unit separator
	}
	
	for old, new in replacements.items():
		s = s.replace(old, new)
	
	return s


def sanitize_for_json(obj: Any) -> Any:
	"""Convert object to JSON-serializable form.
	
	Args:
		obj: Object to sanitize
	
	Returns:
		JSON-serializable version of the object
	"""
	if obj is None or isinstance(obj, bool):
		return obj
	elif isinstance(obj, str):
		# Sanitize strings to remove control characters
		return sanitize_string(obj)
	elif isinstance(obj, int):
		return obj
	elif isinstance(obj, float):
		# Handle NaN and infinity
		if np.isnan(obj) or np.isinf(obj):
			return None
		return obj
	elif isinstance(obj, np.ndarray):
		return sanitize_for_json(obj.tolist())
	elif isinstance(obj, (list, tuple)):
		return [sanitize_for_json(item) for item in obj]
	elif isinstance(obj, dict):
		return {str(k): sanitize_for_json(v) for k, v in obj.items()}
	else:
		# Try to convert to string as last resort
		try:
			return sanitize_string(str(obj))
		except:
			return None


def save_query_result(file_path: str, query_result: Dict[str, Any]) -> None:
	"""Append a single query result to the results file with robust error handling.
	
	Args:
		file_path: Path to the results JSON file
		query_result: Dictionary containing query results
	"""
	max_retries = 3
	retry_delay = 0.5  # seconds
	
	for attempt in range(max_retries):
		try:
			# Step 1: Validate and sanitize input data
			query_result = sanitize_for_json(query_result)
			is_valid, error_msg = validate_json_serializable(query_result)
			if not is_valid:
				print(f"[Warning] Query result validation failed: {error_msg}")
				print(f"[Warning] Attempting to sanitize and continue...")
			
			# Step 2: Read existing data
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					data = json.load(f)
			except json.JSONDecodeError as e:
				print(f"[ERROR] Existing file is corrupted at {e}. Attempting recovery...")
				# Try to load backup
				backup_path = file_path + '.backup'
				if os.path.exists(backup_path):
					print(f"[Recovery] Loading from backup: {backup_path}")
					with open(backup_path, 'r', encoding='utf-8') as f:
						data = json.load(f)
					# Restore from backup
					import shutil
					shutil.copy2(backup_path, file_path)
				else:
					print(f"[ERROR] No backup found. Reinitializing file...")
					data = {'results': [], 'metadata': {
						'model': get_model_name(),
						'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
						'start_time': datetime.now().isoformat()
					}}
			
			# Step 3: Create backup before modifying
			backup_path = file_path + '.backup'
			try:
				import shutil
				if os.path.exists(file_path):
					shutil.copy2(file_path, backup_path)
			except Exception as e:
				print(f"[Warning] Failed to create backup: {e}")
			
			# Step 4: Append new result and update metadata
			data['results'].append(query_result)
			data['metadata']['last_update'] = datetime.now().isoformat()
			unique_queries = len(set(r.get('query', '') for r in data['results']))
			data['metadata']['total_records'] = len(data['results'])
			data['metadata']['total_queries'] = unique_queries
			
			# Step 5: Write to temporary file with extra safety
			temp_path = file_path + '.tmp'
			try:
				# First, try to serialize to string to catch encoding issues early
				json_string = json.dumps(data, indent=2, ensure_ascii=False)
				
				# Write to temporary file
				with open(temp_path, 'w', encoding='utf-8') as f:
					f.write(json_string)
				
			except (TypeError, ValueError, UnicodeEncodeError) as e:
				print(f"[ERROR] Failed to serialize data: {e}")
				print(f"[ERROR] Attempting to use ensure_ascii=True as fallback...")
				try:
					# Fallback: use ensure_ascii=True
					json_string = json.dumps(data, indent=2, ensure_ascii=True)
					with open(temp_path, 'w', encoding='utf-8') as f:
						f.write(json_string)
				except Exception as e2:
					print(f"[ERROR] Fallback also failed: {e2}")
					if os.path.exists(temp_path):
						os.remove(temp_path)
					if attempt < max_retries - 1:
						time.sleep(retry_delay)
						continue
					else:
						raise RuntimeError(f"Failed to serialize data: {e}")
			
			# Step 6: Validate the temporary file
			try:
				with open(temp_path, 'r', encoding='utf-8') as f:
					validated_data = json.load(f)
				
				# Verify data integrity
				if len(validated_data.get('results', [])) != len(data['results']):
					raise ValueError(f"Result count mismatch: expected {len(data['results'])}, got {len(validated_data.get('results', []))}")
				
			except json.JSONDecodeError as e:
				print(f"[ERROR] Generated invalid JSON at {e}. Retrying...")
				if os.path.exists(temp_path):
					os.remove(temp_path)
				if attempt < max_retries - 1:
					time.sleep(retry_delay)
					continue
				else:
					raise RuntimeError(f"Failed to generate valid JSON after {max_retries} attempts")
			except Exception as e:
				print(f"[ERROR] Validation failed: {e}")
				if os.path.exists(temp_path):
					os.remove(temp_path)
				if attempt < max_retries - 1:
					time.sleep(retry_delay)
					continue
				else:
					raise
			
			# Step 7: Atomic rename (this is the critical moment)
			if os.path.exists(file_path):
				os.replace(temp_path, file_path)
			else:
				os.rename(temp_path, file_path)
			
			# Step 8: Verify the final file
			try:
				with open(file_path, 'r', encoding='utf-8') as f:
					final_data = json.load(f)
					if len(final_data['results']) != len(data['results']):
						raise ValueError("Result count mismatch after write")
			except Exception as e:
				print(f"[ERROR] File verification failed: {e}")
				# Restore from backup
				if os.path.exists(backup_path):
					print(f"[Recovery] Restoring from backup...")
					import shutil
					shutil.copy2(backup_path, file_path)
				raise
			
			# Success!
			print(f"[Results] ✓ Saved strategy result (Total records: {len(data['results'])}, Unique queries: {unique_queries})")
			return  # Exit successfully
			
		except Exception as e:
			print(f"[Warning] Save attempt {attempt + 1}/{max_retries} failed: {e}")
			
			# Clean up temp file if it exists
			temp_path = file_path + '.tmp'
			if os.path.exists(temp_path):
				try:
					os.remove(temp_path)
				except:
					pass
			
			# If not last attempt, retry
			if attempt < max_retries - 1:
				print(f"[Retry] Waiting {retry_delay}s before retry...")
				time.sleep(retry_delay)
				retry_delay *= 2  # Exponential backoff
			else:
				# Last attempt failed
				print(f"[ERROR] Failed to save result after {max_retries} attempts: {e}")
				print(f"[ERROR] Query: {query_result.get('query', 'unknown')[:60]}...")
				print(f"[ERROR] Strategy: {query_result.get('strategy_name', 'unknown')}")
				# Log to error file
				try:
					error_log_path = file_path.replace('.json', '_errors.log')
					with open(error_log_path, 'a', encoding='utf-8') as f:
						f.write(f"\n{'='*80}\n")
						f.write(f"Timestamp: {datetime.now().isoformat()}\n")
						f.write(f"Error: {str(e)}\n")
						f.write(f"Query: {query_result.get('query', 'unknown')}\n")
						f.write(f"Strategy: {query_result.get('strategy_name', 'unknown')}\n")
						f.write(f"{'='*80}\n")
					print(f"[Log] Error logged to: {error_log_path}")
				except:
					pass

def improve_with_model_selection(query: str, idx: int, sources: List[str] = None, summaries: List[str] = None, 
                                  impression_fn = impression_wordpos_count_simple, return_detailed_scores = False):
	"""Process a query using surrogate model for strategy selection.
	
	New workflow:
	1. Use surrogate model to rank all strategies
	2. Select the best strategy based on model prediction
	3. Apply the selected strategy and compute GEO scores
	
	Args:
		query: The query string
		idx: Index of the source to optimize
		sources: List of source texts
		summaries: List of summary texts
		impression_fn: Function to calculate impression scores
		return_detailed_scores: If True, return detailed scores
	
	Returns:
		Dict with model predictions and GEO evaluation results
	"""
	global loaded_cache, strategy_selector
	global LAST_UPDATE_TIME
	
	# Get cache file path
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	cache_file_path = os.environ.get('GLOBAL_CACHE_FILE', os.path.join(SCRIPT_DIR, 'global_cache.json'))
	
	# Load cache if needed
	# If summaries are provided (e.g., from MSdata text_list), cache is optional
	# If summaries are None, cache is required to get sources
	cache_required = summaries is None
	
	if loaded_cache is None:
		try:
			print(f"Loading cache from {cache_file_path}...")
			loaded_cache = json.load(open(cache_file_path, encoding='utf-8'), strict=False)
			print(f"Cache loaded successfully: {len(loaded_cache)} queries")
			LAST_UPDATE_TIME = os.path.getmtime(cache_file_path)
		except json.JSONDecodeError as e:
			if cache_required:
				print(f"CRITICAL ERROR: Cache file {cache_file_path} is corrupted at {e}")
				raise RuntimeError(f"Cache file corrupted. Cannot proceed without valid cache.")
			else:
				print(f"Warning: Cache file {cache_file_path} is corrupted at {e}. Using empty cache (summaries provided, cache not required).")
				loaded_cache = {}
		except FileNotFoundError:
			if cache_required:
				print(f"CRITICAL ERROR: Cache file {cache_file_path} not found")
				raise RuntimeError(f"Cache file not found. Please preload cache first.")
			else:
				print(f"Info: Cache file {cache_file_path} not found. Using empty cache (summaries provided, cache not required).")
				loaded_cache = {}
	
	print(f'Query: {query}')
	
	# Get initial answers and scores
	answers = get_answer(query, summaries=summaries, num_completions=5, n=5, loaded_cache=loaded_cache)
	if sources is None:
		sources = [x['source'] for x in answers['sources']]
		# If sources is empty but summaries are provided, create placeholder sources
		if len(sources) == 0 and summaries is not None and len(summaries) > 0:
			sources = [f"Source {i+1}" for i in range(len(summaries))]
	if summaries is None:
		summaries = [x['summary'] for x in answers['sources']]
	
	answers = answers['responses'][-1]
	
	# Calculate initial scores
	if impression_fn == impression_subjective_impression or impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
		orig_init_scores = np.array([impression_fn(x, query, 5, idx=idx) for x in answers])
		orig_init_scores = orig_init_scores[~np.all(orig_init_scores == 0, axis=1)]
	else:
		orig_init_scores = np.array([impression_fn(extract_citations_new(x), 5) for x in answers])
	
	init_scores = orig_init_scores.mean(axis=0)
	print(f'Init Scores: {init_scores}')
	
	# Store content before
	content_before = summaries[idx] if summaries else None
	init_target_score = init_scores[idx] if len(init_scores) > idx else 0.0
	init_all_scores = init_scores.tolist()
	
	# === Model Selection Phase ===
	print('\n[Model Selection] Using surrogate model to select best strategy...')
	
	# Evolved mode: strategies loaded from archive, pass None here
	strategy_prompts = None
	
	# Rank strategies using model
	best_strategy_name, model_predictions = strategy_selector.select_best_strategy(
		query=query,
		context=content_before,
		strategy_prompts=strategy_prompts
	)
	
	# Get strategy metadata for evolved strategies
	if USE_EVOLVED_SURROGATE and hasattr(strategy_selector, 'strategy_metadata'):
		best_strategy_metadata = strategy_selector.strategy_metadata.get(best_strategy_name, {})
		best_strategy_type = best_strategy_metadata.get('strategy_type', 'unknown')
		# Add best_strategy to model_predictions for consistency
		model_predictions['best_strategy'] = best_strategy_metadata
	else:
		best_strategy_metadata = {}
		best_strategy_type = best_strategy_name
	
	# Display strategy info (show strategy_type if available, otherwise show ID)
	# Only evolved strategies: log type + ID
	print(f'[Model Selection] Best strategy: {best_strategy_type} (ID: {best_strategy_name[:8]}...)')
	
	print(f'[Model Selection] Predicted reward: {model_predictions["expected_reward"]:.4f}')
	print(f'[Model Selection] Uncertainty: {model_predictions["uncertainty"]:.4f}')
	print(f'[Model Selection] Top-3 strategies:')
	for i, ranking in enumerate(model_predictions['all_rankings'][:3]):
		strategy_id = ranking.get('strategy', '')
		# Try to get strategy type from metadata
		if USE_EVOLVED_SURROGATE and hasattr(strategy_selector, 'strategy_metadata'):
			strategy_metadata = strategy_selector.strategy_metadata.get(strategy_id, {})
			strategy_type = strategy_metadata.get('strategy_type', '')
		else:
			strategy_type = ''
		if strategy_type:
			print(f'  {i+1}. {strategy_type} (ID: {strategy_id[:8]}...): {ranking["reward"]:.4f} (±{ranking["uncertainty"]:.4f})')
		else:
			print(f'  {i+1}. {strategy_id}: {ranking["reward"]:.4f} (±{ranking["uncertainty"]:.4f})')
	
	# === Iterative Optimization ===
	# Number of iterations (apply strategy multiple times)
	num_iterations = int(os.environ.get('GEO_NUM_ITERATIONS', '2'))
	
	# Track used strategies for exclusion in later iterations
	used_strategy_ids = set()
	used_strategy_types = set()
	
	# Track strategies used in each iteration (for result saving)
	iteration_strategies = []
	
	# Display strategy info for logging (evolved only)
	best_strategy_info = model_predictions.get('best_strategy', {})
	best_strategy_type = best_strategy_info.get('strategy_type', 'unknown')
	print(f'\n[GEO Evaluation] Applying evolved strategy: {best_strategy_type} (ID: {best_strategy_name[:8]}...)')
	print(f'[GEO Evaluation] Iterations: {num_iterations}')
	
	# Iterative optimization: apply strategy multiple times
	content_after = summaries[idx] if summaries else None
	current_strategy_name = best_strategy_name
	current_model_predictions = model_predictions
	
	for iteration in range(num_iterations):
		if iteration > 0:
			print(f'\n[GEO Evaluation] Iteration {iteration + 1}/{num_iterations}...')
			print(f'[GEO Evaluation] Excluding used strategies: {len(used_strategy_ids)} IDs, {len(used_strategy_types)} types')
			
			# Evolved strategies: filter out used strategies for later iterations
			if hasattr(strategy_selector, 'strategy_metadata') and hasattr(strategy_selector, 'strategy_prompts'):
				available_strategy_ids = [
					gid for gid, metadata in strategy_selector.strategy_metadata.items()
					if gid not in used_strategy_ids
					and metadata.get('strategy_type', '') not in used_strategy_types
				]
				
				if not available_strategy_ids:
					print(f'[GEO Evaluation] Warning: No more strategies available, stopping iteration')
					break
				
				print(f'[GEO Evaluation] Selecting from {len(available_strategy_ids)} remaining strategies...')
				
				available_strategy_prompts = {
					gid: strategy_selector.strategy_prompts[gid]
					for gid in available_strategy_ids
					if gid in strategy_selector.strategy_prompts
				}
				
				if not available_strategy_prompts:
					print(f'[GEO Evaluation] Warning: No valid strategy prompts found, stopping iteration')
					break
				
				# Re-rank remaining strategies using current content
				rankings = strategy_selector.rank_strategies(
					query=query,
					context=content_after,
					strategy_prompts=available_strategy_prompts,
					exclude_ids=None
				)
				
				if rankings:
					best_id, best_reward, best_uncertainty = rankings[0]
					current_strategy_name = best_id
					
					best_strategy_metadata = strategy_selector.strategy_metadata.get(best_id, {})
					current_strategy_type = best_strategy_metadata.get('strategy_type', 'unknown')
					
					raw_reward = best_reward
					
					print(f'[Model Selection] Iteration {iteration + 1} - Best strategy: {current_strategy_type} (ID: {best_id[:8]}...)')
					print(f'[Model Selection] Predicted reward: {raw_reward:.4f}')
					
					current_model_predictions = {
						'expected_reward': raw_reward,
						'uncertainty': best_uncertainty,
						'best_strategy': best_strategy_metadata
					}
				else:
					print(f'[GEO Evaluation] Warning: No strategies ranked, stopping iteration')
					break
			else:
				print(f'[GEO Evaluation] Warning: Cannot filter strategies (missing strategy_metadata or strategy_prompts), stopping iteration')
				break
		
		# Apply the selected strategy (evolved only)
		best_strategy_info = current_model_predictions.get('best_strategy', {})
		full_prompt = best_strategy_info.get('full_prompt', '')
		if full_prompt and content_after:
			if iteration == 0:
				print(f'[GEO Evaluation] Using evolved strategy FULL_PROMPT for rewriting')
			content_after = apply_evolved_strategy(query, content_after, full_prompt)
		else:
			if iteration == 0:
				print(f'[GEO Evaluation] Warning: No FULL_PROMPT found, using identity')
			# Keep current content unchanged
			pass
		
		# Record used strategy for exclusion in next iteration (evolved only)
		used_strategy_ids.add(current_strategy_name)
		best_strategy_info = current_model_predictions.get('best_strategy', {})
		strategy_type = best_strategy_info.get('strategy_type', '')
		if strategy_type:
			used_strategy_types.add(strategy_type)
			print(f'[GEO Evaluation] Marked strategy as used: ID={current_strategy_name[:8]}..., type={strategy_type}')
		iteration_strategies.append({
			'iteration': iteration + 1,
			'strategy_id': current_strategy_name,
			'strategy_type': strategy_type,
			'predicted_reward': current_model_predictions.get('expected_reward', 0.0)
		})
	
	# Evaluate with GEO
	summaries_copy = summaries[:idx] + [content_after] + summaries[idx+1:]
	answers = get_answer(query, summaries=summaries_copy, num_completions=5, n=5, loaded_cache=loaded_cache)
	answers = answers['responses'][-1]
	
	# Calculate detailed scores (word, pos, overall)
	# overall is simple_wordpos
	final_overall_scores = [impression_wordpos_count_simple(extract_citations_new(x), 5) for x in answers]
	final_word_scores = [impression_word_count_simple(extract_citations_new(x), 5) for x in answers]
	final_pos_scores = [impression_pos_count_simple(extract_citations_new(x), 5) for x in answers]
	
	final_overall_mean = np.array(final_overall_scores).mean(axis=0)
	final_word_mean = np.array(final_word_scores).mean(axis=0)
	final_pos_mean = np.array(final_pos_scores).mean(axis=0)
	
	print(f'[GEO Evaluation] Final scores (Overall): {final_overall_mean}')
	print(f'[GEO Evaluation] Final scores (Word): {final_word_mean}')
	print(f'[GEO Evaluation] Final scores (Pos): {final_pos_mean}')
	
	# Maintain compatibility with existing code that uses final_scores_mean
	final_scores_mean = final_overall_mean
	
	# Calculate improvement
	improvement = (final_scores_mean - init_scores)
	final_target_score = final_scores_mean[idx] if len(final_scores_mean) > idx else 0.0
	final_all_scores = final_scores_mean.tolist()
	
	# Calculate actual GEO reward (for comparison with model prediction)
	actual_reward = improvement[idx]  # Improvement on target source
	print(f'[GEO Evaluation] Actual reward: {actual_reward:.4f}')
	print(f'[Comparison] Model predicted: {model_predictions["expected_reward"]:.4f}, Actual: {actual_reward:.4f}')
	
	# Prepare result
	if return_detailed_scores:
		# Use overall scores for detailed per-completion results
		final_scores_per_completion = [score.tolist() if isinstance(score, np.ndarray) else score for score in final_overall_scores]
		
		# Prepare iteration info
		iteration_info = {
			'num_iterations': len(iteration_strategies),
			'strategies_used': iteration_strategies
		}
		
		return {
			'selected_strategy': best_strategy_name,  # First iteration strategy (for backward compatibility)
			'iteration_info': iteration_info,  # All iterations info
			'model_predictions': {
				'expected_reward': format_number_7decimals(model_predictions['expected_reward']),
				'uncertainty': format_number_7decimals(model_predictions['uncertainty']),
				'all_rankings': [
					{
						'strategy': r['strategy'],
						'reward': format_number_7decimals(r['reward']),
						'uncertainty': format_number_7decimals(r['uncertainty'])
					}
					for r in model_predictions['all_rankings']
				]
			},
			'content_before': content_before,
			'content_after': content_after,  # Final content after all iterations
			'content_after_list': summaries_copy,  # Updated summaries list in original order
			'init_scores': format_scores_7decimals(init_scores.tolist()),
			'final_scores': format_scores_7decimals(final_scores_mean.tolist()),
			'improvement': format_scores_7decimals(improvement.tolist()),
			'init_target_source_score': format_number_7decimals(init_target_score),
			'init_all_source_scores': format_scores_7decimals(init_all_scores),
			'target_source_score': format_number_7decimals(final_target_score),
			'target_source_word_score': format_number_7decimals(final_word_mean[idx]),
			'target_source_pos_score': format_number_7decimals(final_pos_mean[idx]),
			'target_source_overall_score': format_number_7decimals(final_overall_mean[idx]),
			'all_source_scores': format_scores_7decimals(final_all_scores),
			'final_scores_per_completion': format_scores_7decimals(final_scores_per_completion),
			'actual_reward': format_number_7decimals(actual_reward)
		}
	else:
		return {
			'selected_strategy': best_strategy_name,
			'improvement': improvement[idx],
			'success': improvement[idx] > 0
		}


def improve(query : str, idx : int, sources : List[str] = None, summaries : List[str] = None, impression_fn = impression_wordpos_count_simple, returnFullData = False, static_cache=os.environ.get('STATIC_CACHE', 'True')=='True', return_detailed_scores = False): 
	"""Process a query through GEO optimization pipeline.
	
	Note: static_cache now defaults to True to use pre-loaded cache from GEO-Bench.
	This prevents unnecessary Google searches and improves performance.
	
	Args:
		query: The query string
		idx: Index of the source to optimize
		sources: List of source texts
		summaries: List of summary texts
		impression_fn: Function to calculate impression scores
		returnFullData: If True, return full score data
		static_cache: Whether to use static cache
		return_detailed_scores: If True, return detailed scores for all methods
	
	Returns:
		If return_detailed_scores=True: dict with init_scores and method_scores
		Otherwise: improvements matrix and boolean array
	"""
	global loaded_cache
	global LAST_UPDATE_TIME
	# Get cache file path (same logic as utils.py)
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	cache_file_path = os.environ.get('GLOBAL_CACHE_FILE', os.path.join(SCRIPT_DIR, 'global_cache.json'))
	
	# Always load cache for efficient operation (unless explicitly disabled)
	if loaded_cache is None:
		try:
			print(f"Loading cache from {cache_file_path}...")
			loaded_cache = json.load(open(cache_file_path, encoding='utf-8'), strict=False)
			print(f"Cache loaded successfully: {len(loaded_cache)} queries")
			LAST_UPDATE_TIME = os.path.getmtime(cache_file_path)
		except json.JSONDecodeError as e:
			print(f"CRITICAL ERROR: Cache file {cache_file_path} is corrupted at {e}")
			print(f"Please fix the JSON file or restore from backup.")
			raise RuntimeError(f"Cache file corrupted. Cannot proceed without valid cache.")
		except FileNotFoundError:
			print(f"CRITICAL ERROR: Cache file {cache_file_path} not found")
			print(f"Please run: python src/preload_cache_from_geobench.py")
			raise RuntimeError(f"Cache file not found. Please preload cache first.")
	# idx indicates the website to boost
	print('query is', query)
	answers = get_answer(query, summaries = summaries, num_completions = 5, n = 5, loaded_cache = loaded_cache)
	if sources is None:
		sources = [x['source'] for x in answers['sources']]
	if summaries is None:
		summaries = [x['summary'] for x in answers['sources']]

	answers = answers['responses'][-1]

	if impression_fn == impression_subjective_impression or  impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
		orig_init_scores = np.array([impression_fn(x, query, 5, idx = idx) for x in answers])
		orig_init_scores = orig_init_scores[~np.all(orig_init_scores == 0, axis=1)]
	else:
		orig_init_scores = np.array([impression_fn(extract_citations_new(x), 5) for x in answers])
	
	init_scores = orig_init_scores.mean(axis=0)
	print('Init Scores: ', init_scores)
	improvements = []
	all_final_scores = []
	method_scores_dict = {}  # Store scores for each method
	strategy_results = []  # Store strategy results with content before/after

	# Store content before (original content)
	content_before = summaries[idx] if summaries else None
	
	# Store initial scores for target source (idx) - the two components
	init_target_score = init_scores[idx] if len(init_scores) > idx else 0.0
	init_all_scores = init_scores.tolist()  # All 5 source scores

	for meth_name in GEO_METHODS:
		# Get content after applying the strategy
		content_after = GEO_METHODS[meth_name](summaries[idx]) if summaries else None
		
		summaries_copy = summaries[:idx] + [content_after] + summaries[idx+1:] 
		answers = get_answer(query, summaries = summaries_copy, num_completions = 5, n = 5, loaded_cache = loaded_cache)
		answers = answers['responses'][-1]
		if impression_fn == impression_subjective_impression or impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
			final_scores = np.array([impression_fn(x, query, 5, idx = idx) for x in answers])
			final_scores = final_scores[~np.all(final_scores == 0, axis=1)]
		else:
			final_scores = [impression_fn(extract_citations_new(x), 5) for x in answers]
		all_final_scores.append(np.array(final_scores))
		final_scores_mean = np.array(final_scores).mean(axis=0)
		print(f"{meth_name}: {final_scores_mean}")
		
		# Store scores for this method
		method_scores_dict[meth_name] = final_scores_mean.tolist()
		
		improvement = (final_scores_mean - init_scores)
		improvements.append(improvement)
		
		# Store strategy result with content before/after
		if return_detailed_scores:
			# Get target source score (idx) and all source scores
			final_target_score = final_scores_mean[idx] if len(final_scores_mean) > idx else 0.0
			final_all_scores = final_scores_mean.tolist()  # All 5 source scores
			
			# Store individual completion scores for detailed analysis
			final_scores_per_completion = [score.tolist() if isinstance(score, np.ndarray) else score for score in final_scores]
			
			strategy_results.append({
				'strategy_name': meth_name,
				'content_before': content_before,
				'content_after': content_after,
				'improvement': format_scores_7decimals(improvement.tolist()),
				'final_scores': format_scores_7decimals(final_scores_mean.tolist()),
				# Detailed scores: target source and all sources
				'target_source_score': format_number_7decimals(final_target_score),
				'all_source_scores': format_scores_7decimals(final_all_scores),
				'init_target_source_score': format_number_7decimals(init_target_score),
				'init_all_source_scores': format_scores_7decimals(init_all_scores),
				# Per-completion scores for detailed analysis
				'final_scores_per_completion': format_scores_7decimals(final_scores_per_completion)
			})
	
	improvements = np.vstack(improvements)

	if return_detailed_scores:
		# Return detailed scores for saving, including content before/after for each strategy
		# Format all scores to 7 decimal places
		formatted_method_scores = {k: format_scores_7decimals(v) for k, v in method_scores_dict.items()}
		return {
			'init_scores': format_scores_7decimals(init_scores.tolist()),
			'method_scores': formatted_method_scores,
			'improvements': format_scores_7decimals(improvements.tolist()),
			'content_before': content_before,
			'strategy_results': strategy_results
		}
	elif returnFullData:
		return orig_init_scores, all_final_scores
	else:
		return improvements, improvements[:, idx] > 0


async def process_single_query_async(item_tuple, results_file_path, loaded_cache=None):
	"""Async process a single query with error handling and immediate result saving."""
	i, k = item_tuple
	query = k['query']
	
	# Check if we have text_list from parquet file (direct sources)
	text_list = k.get('text_list', None)
	has_direct_sources = text_list is not None and len(text_list) > 0
	
	# Check if we have sources from jsonl file (test set has sources field)
	jsonl_sources = k.get('sources', None)
	has_jsonl_sources = jsonl_sources is not None and len(jsonl_sources) > 0
	
	# If no direct sources, check jsonl sources, then cache
	if not has_direct_sources and not has_jsonl_sources:
		if loaded_cache is not None:
			cache_entry = loaded_cache.get(query)
			if cache_entry is None or not cache_entry:
				print(f"Skipping query {i+1}: '{query[:60]}...' - not found in cache and no text_list/sources")
				return (i, query, None, "SKIPPED: Query not found in cache and no text_list/sources")
			
			# Check if sources exist and are not empty
			sources = cache_entry[0].get('sources', []) if isinstance(cache_entry, list) and len(cache_entry) > 0 else []
			if not sources or len(sources) == 0:
				print(f"Skipping query {i+1}: '{query[:60]}...' - no sources in cache")
				return (i, query, None, "SKIPPED: No sources in cache")
		else:
			print(f"Skipping query {i+1}: '{query[:60]}...' - no text_list/sources and no cache")
			return (i, query, None, "SKIPPED: No text_list/sources and no cache")
	
	try:
		print(f"Processing query {i+1}: {query}")
		
		# Prepare summaries from different sources
		summaries = None
		if has_direct_sources:
			# Convert text_list (numpy array or list of strings) to summaries list
			if isinstance(text_list, (list, tuple)) or hasattr(text_list, '__iter__'):
				summaries = [str(text) for text in text_list]
			else:
				summaries = [str(text_list)]
			print(f"  Using {len(summaries)} sources from text_list")
		elif has_jsonl_sources:
			# Extract cleaned_text from jsonl sources
			summaries = []
			for source in jsonl_sources:
				if isinstance(source, dict):
					# Use cleaned_text if available, otherwise raw_text
					text = source.get('cleaned_text') or source.get('raw_text', '')
					if text:
						summaries.append(str(text))
				elif isinstance(source, str):
					summaries.append(source)
			print(f"  Using {len(summaries)} sources from jsonl sources field")
		
		# Use asyncio.to_thread to run synchronous improve function with model selection
		result = await asyncio.to_thread(
			improve_with_model_selection, 
			query, 
			idx=int(k['sugg_idx']), 
			summaries=summaries,  # Pass summaries if available
			impression_fn=impression_wordpos_count_simple,
			return_detailed_scores=True  # Get detailed scores for saving
		)
		print(f"Query {i+1} completed")
		
		# Save result with model predictions (new format)
		suggest_idx = int(k['sugg_idx'])
		query_result = {
			'query': query,
			'suggest_idx': suggest_idx,
			'selected_strategy': result['selected_strategy'],
			'content_before': result['content_before'],
			'content_after': result['content_after'],
			'content_after_list': result.get('content_after_list', []),  # All contents in original order after rewriting
			'model_predictions': result['model_predictions'],  # Model's ranking and scores
			'geo_evaluation': {
				'init_scores': result['init_scores'],
				'final_scores': result['final_scores'],
				'improvement': result['improvement'],
				'init_target_source_score': result['init_target_source_score'],
				'init_all_source_scores': result['init_all_source_scores'],
				# Three detailed scores for target source: word, pos, overall
				'target_source_word_score': result.get('target_source_word_score', 0.0),
				'target_source_pos_score': result.get('target_source_pos_score', 0.0),
				'target_source_overall_score': result.get('target_source_overall_score', 0.0),
				'target_source_score': result['target_source_score'],  # Same as overall for backward compatibility
				'all_source_scores': result['all_source_scores'],
				'final_scores_per_completion': result['final_scores_per_completion'],
				'actual_reward': result['actual_reward']  # Actual GEO improvement
			},
			'timestamp': datetime.now().isoformat()
		}
		# Save immediately after processing
		await asyncio.to_thread(save_query_result, results_file_path, query_result)
		
		return (i, query, result, None)
	except Exception as e:
		print(f"Error processing query {i+1}: {e}")
		# Save error result too
		error_result = {
			'query': query,
			'suggest_idx': int(k.get('sugg_idx', -1)) if 'sugg_idx' in k else -1,
			'strategy_name': 'error',
			'content_before': '',
			'content_after': '',
			'res': {'error': str(e)},
			'timestamp': datetime.now().isoformat()
		}
		await asyncio.to_thread(save_query_result, results_file_path, error_result)
		return (i, query, None, str(e))


async def run_sequential_async(dataset, dataset_split, results_file_path, loaded_cache=None):
	"""Sequential execution using async (original method, but async)."""
	print("\n=== Running in SEQUENTIAL mode ===\n")
	results = []
	for i, k in enumerate(dataset[dataset_split]):
		result = await process_single_query_async((i, k), results_file_path, loaded_cache)
		results.append(result)
	return results


async def run_concurrent_async(dataset, dataset_split, results_file_path, max_concurrent=4, loaded_cache=None):
	"""Concurrent execution using asyncio."""
	print(f"\n=== Running in ASYNC CONCURRENT mode (max_concurrent={max_concurrent}) ===\n")
	
	# Prepare all tasks
	tasks = list(enumerate(dataset[dataset_split]))
	
	# Use semaphore to limit concurrent tasks
	semaphore = asyncio.Semaphore(max_concurrent)
	
	async def process_with_semaphore(task):
		async with semaphore:
			return await process_single_query_async(task, results_file_path, loaded_cache)
	
	# Create all coroutines
	coroutines = [process_with_semaphore(task) for task in tasks]
	
	# Execute concurrently with progress bar
	results = []
	with tqdm(total=len(tasks), desc="Processing queries") as pbar:
		for coro in asyncio.as_completed(coroutines):
			result = await coro
			results.append(result)
			pbar.update(1)
	
	# Sort results by index to maintain order
	results.sort(key=lambda x: x[0])
	return results


async def main_async():
	"""Main async entry point."""
	global strategy_selector
	
	# Set environment variables for offline mode (no HuggingFace internet access needed)
	os.environ['HF_HUB_OFFLINE'] = '1'
	os.environ['TRANSFORMERS_OFFLINE'] = '1'
	
	# Get dataset type from environment variable (default: 'msdata')
	# Options: 'msdata', 'ecommerce', or 'geobench'
	dataset_type = os.environ.get('DATASET_TYPE', 'msdata').lower()
	if dataset_type not in ['msdata', 'ecommerce', 'geobench']:
		print(f"[Warning] Unknown dataset_type '{dataset_type}', using 'msdata'")
		dataset_type = 'msdata'
	print(f"[INFO] Dataset type: {dataset_type}")
	
	# Load surrogate model at startup
	print("\n" + "="*60)
	print("LOADING SURROGATE MODEL FOR STRATEGY SELECTION")
	print("="*60)
	
	# Always initialize evolved surrogate (default behavior)
	print("[INFO] Initializing EVOLVED strategy selector")
	
	proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	
	# Resolve base model path for surrogate:
	# 1. Use EVOLVED_BASE_MODEL if set (absolute or relative to project root)
	# 2. Otherwise default to <project_root>/base_model
	# 3. If still not found, fall back to legacy Qwen locations (for older environments)
	base_model_path = os.environ.get('EVOLVED_BASE_MODEL', None)
	if base_model_path and not os.path.isabs(base_model_path):
		base_model_path = os.path.join(proj_root, base_model_path)
	
	if not base_model_path:
		base_model_path = os.path.join(proj_root, "base_model")
	
	if not os.path.exists(base_model_path):
		original_qwen_paths = [
			'/autodl-fs/data/modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct',
			'/autodl-fs/data/modelscope_cache/Qwen/Qwen2___5-1___5B-Instruct',
			'/home/zhangziwei/yuanjq/LLMs/Qwen/Qwen2.5-1.5B-Instruct',
			'Qwen/Qwen2.5-1.5B-Instruct'
		]
		for qwen_path in original_qwen_paths:
			if os.path.exists(qwen_path):
				base_model_path = qwen_path
				print(f"[INFO] Using base model from legacy path: {base_model_path}")
				break
	
	if not base_model_path or not os.path.exists(base_model_path):
		raise ValueError(
			f"Base model not found. Expected at EVOLVED_BASE_MODEL or {os.path.join(proj_root, 'base_model')}."
		)
	
	pretrained_backbone_path = os.environ.get('EVOLVED_PRETRAINED_BACKBONE', None)
	if pretrained_backbone_path and not os.path.isabs(pretrained_backbone_path):
		pretrained_backbone_path = os.path.join(proj_root, pretrained_backbone_path)
	
	if not pretrained_backbone_path:
		surrogate_stage1_paths = [
			f'surrogate_{dataset_type}_full/stage1/checkpoint-610/exported_bin/pytorch_model.bin',
			f'surrogate_{dataset_type}_full/stage1/checkpoint-610',
		]
		for ckpt_path in surrogate_stage1_paths:
			full_ckpt_path = os.path.join(proj_root, ckpt_path)
			if os.path.exists(full_ckpt_path):
				pretrained_backbone_path = full_ckpt_path
				print(f"[INFO] Found pretrained backbone checkpoint: {pretrained_backbone_path}")
				break
	
	if not pretrained_backbone_path:
		fallback_datasets = ['msdata', 'ecommerce', 'geobench']
		for fb_dataset in fallback_datasets:
			if fb_dataset == dataset_type:
				continue
			ckpt_path = os.path.join(proj_root, f'surrogate_{fb_dataset}_full/stage1/checkpoint-610/exported_bin/pytorch_model.bin')
			if os.path.exists(ckpt_path):
				pretrained_backbone_path = ckpt_path
				print(f"[WARN] Using fallback pretrained backbone from {fb_dataset}: {pretrained_backbone_path}")
				break
	
	if pretrained_backbone_path:
		print(f"[INFO] Pretrained backbone: {pretrained_backbone_path}")
	else:
		print(f"[WARN] No pretrained backbone checkpoint found, will use base model weights only")
	
	default_value_head = 'evolved/critic/value_head.bin'
	value_head_path = os.environ.get('EVOLVED_VALUE_HEAD', default_value_head)
	if not os.path.isabs(value_head_path):
		value_head_path = os.path.join(proj_root, value_head_path)
	
	default_lora = 'evolved/critic/lora_adapter'
	lora_adapter_path = os.environ.get('EVOLVED_LORA_ADAPTER', None)
	if lora_adapter_path and not os.path.isabs(lora_adapter_path):
		lora_adapter_path = os.path.join(proj_root, lora_adapter_path)
	if lora_adapter_path is None:
		default_lora_path = os.path.join(proj_root, default_lora)
		if os.path.exists(default_lora_path):
			lora_adapter_path = default_lora_path
			print(f"[INFO] Found default LoRA adapter at: {lora_adapter_path}")
	
	default_strategies = 'evolved/archive/strategies.json'
	strategies_path = os.environ.get('EVOLVED_STRATEGIES', default_strategies)
	if not os.path.isabs(strategies_path):
		strategies_path = os.path.join(proj_root, strategies_path)
	
	critic_reward_scale = float(os.environ.get('CRITIC_REWARD_SCALE', '10.0'))
	
	print(f"[INFO] ========== Evolved Surrogate Model Configuration ==========")
	print(f"[INFO] Base model (for structure): {base_model_path}")
	print(f"[INFO] Pretrained backbone (for weights): {pretrained_backbone_path}")
	print(f"[INFO] Value head: {value_head_path}")
	print(f"[INFO] LoRA adapter: {lora_adapter_path}")
	print(f"[INFO] Strategies: {strategies_path}")
	print(f"[INFO] Critic reward scale: {critic_reward_scale}")
	print(f"[INFO] ==========================================================")
	
	tokenizer_path = os.environ.get('EVOLVED_TOKENIZER_PATH', None)
	if tokenizer_path and not os.path.isabs(tokenizer_path):
		tokenizer_path = os.path.join(proj_root, tokenizer_path)
	if not tokenizer_path:
		tokenizer_path = base_model_path
	print(f"[INFO] Tokenizer: {tokenizer_path}")
	
	critic_device = os.environ.get('EVOLVED_CRITIC_DEVICE', None)
	if critic_device:
		print(f"[INFO] Critic device: {critic_device}")
	
	strategy_selector = StrategySelector(
		base_model_path=base_model_path,
		value_head_path=value_head_path,
		strategies_path=strategies_path,
		lora_adapter_path=lora_adapter_path,
		pretrained_backbone_path=pretrained_backbone_path,
		tokenizer_path=tokenizer_path,
		device=critic_device,
		use_chat_template=True,
		cutoff_len=2048,
		critic_reward_scale=critic_reward_scale
	)
	
	print("="*60 + "\n")
	
	# Get dataset split from environment variable (default: 'test')
	# Options: 'train', 'test', 'val'
	dataset_split = os.environ.get('DATASET_SPLIT', 'test').lower()
	if dataset_split not in ['train', 'test', 'val']:
		print(f"Warning: Invalid DATASET_SPLIT '{dataset_split}', using 'test'")
		dataset_split = 'test'
	
	# Initialize results file at the start (include dataset split in filename)
	results_file_path = init_results_file(dataset_split)
	
	# Define ParquetDataset class for parquet format
	class ParquetDataset:
		def __init__(self, data_dict):
			self.data = data_dict
			self.column_names = list(data_dict.keys())
		
		def __len__(self):
			return len(self.data['query'])
		
		def __getitem__(self, idx):
			return {key: self.data[key][idx] for key in self.column_names}
		
		def __iter__(self):
			for i in range(len(self)):
				yield self[i]
		
		def shuffle(self, seed=None):
			# Shuffle is already done on DataFrame, so just return self
			# This method exists for compatibility with datasets library API
			return self
	
	# Define MSDATADataset class for MSdata format
	class MSDATADataset:
		def __init__(self, data_dict):
			self.data = data_dict
			self.column_names = list(data_dict.keys())
		
		def __len__(self):
			return len(self.data['query'])
		
		def __getitem__(self, idx):
			return {key: self.data[key][idx] for key in self.column_names}
		
		def __iter__(self):
			for i in range(len(self)):
				yield self[i]
		
		def shuffle(self, seed=None):
			# Already shuffled if needed
			return self
		
		def select(self, indices):
			# Create a new dataset with selected indices (for sampling)
			sampled_dict = {
				'query': [self.data['query'][i] for i in indices],
				'sugg_idx': [self.data['sugg_idx'][i] for i in indices],
				'text_list': [self.data['text_list'][i] for i in indices]
			}
			return MSDATADataset(sampled_dict)
	
	# Load dataset based on dataset type
	if dataset_type == 'msdata':
		# Load MSdata JSON format
		msdata_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MSdata", f"{dataset_split}.json")
		
		if not os.path.exists(msdata_path):
			raise FileNotFoundError(f"MSdata file not found: {msdata_path}")
		
		print(f"Loading MSdata dataset from: {msdata_path}")
		print(f"Using dataset split: {dataset_split}")
		
		with open(msdata_path, 'r', encoding='utf-8') as f:
			data_list = json.load(f)
		
		print(f"Loaded {len(data_list)} entries from MSdata")
		
		# Shuffle training set if using train split
		if dataset_split == 'train':
			import random
			shuffle_seed = int(os.environ.get('SHUFFLE_SEED', 42))
			print(f"Shuffling training dataset with seed={shuffle_seed}...")
			random.seed(shuffle_seed)
			random.shuffle(data_list)
			print(f"Dataset shuffled successfully!")
		
		# Convert MSdata format to expected format
		# MSdata has: query_id, query, suggest_id, text_list
		# Expected: query, sugg_idx, text_list
		dataset_dict = {
			'query': [item['query'] for item in data_list],
			'sugg_idx': [item['suggest_id'] for item in data_list],
			'text_list': [item['text_list'] for item in data_list]
		}
		
		dataset = {dataset_split: MSDATADataset(dataset_dict)}
		print(f"Converted MSdata to dataset format: {len(dataset[dataset_split])} entries")
		use_parquet = False
		use_jsonl = False
		use_msdata = True
	else:
		# Load GEO-Bench dataset (original code)
		use_msdata = False
		# Try to load parquet file first, fallback to jsonl
		parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GEO-Bench", "GEO-Bench", f"{dataset_split}.parquet")
		jsonl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GEO-Bench", "geo-bench-hf", f"{dataset_split}.jsonl")
		
		# Check which file exists
		use_parquet = os.path.exists(parquet_path)
		use_jsonl = os.path.exists(jsonl_path)
		
		if use_parquet:
			print(f"Loading dataset from parquet file: {parquet_path}")
			print(f"Using dataset split: {dataset_split}")
			# Load parquet file using pandas, then convert to dataset format
			import pandas as pd
			# Note: numpy is already imported at the top of the file
			df = pd.read_parquet(parquet_path)
			print(f"Loaded {len(df)} rows from parquet file")
			
			# Shuffle training set if using train split (before converting to dataset)
			if dataset_split == 'train':
				shuffle_seed = int(os.environ.get('SHUFFLE_SEED', 42))
				print(f"Shuffling training dataset with seed={shuffle_seed}...")
				df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
				print(f"Dataset shuffled successfully!")
			
			# Convert parquet format to expected format
			# Parquet has: query_id, query, target_id, text_list
			# Expected: query, sugg_idx (and sources from cache)
			dataset_dict = {
				'query': df['query'].tolist(),
				'sugg_idx': df['target_id'].tolist(),
				'text_list': df['text_list'].tolist()  # Store text_list for later use
			}
			
			dataset = {dataset_split: ParquetDataset(dataset_dict)}
			print(f"Converted parquet to dataset format: {len(dataset[dataset_split])} entries")
			
		elif use_jsonl:
			print(f"Loading dataset from jsonl file: {jsonl_path}")
			print(f"Using dataset split: {dataset_split}")
			dataset = load_dataset("json", data_files={dataset_split: jsonl_path})
			
			# Shuffle training set if using train split (only for jsonl, parquet is already shuffled)
			if dataset_split == 'train':
				# Get random seed from environment variable or use default
				shuffle_seed = int(os.environ.get('SHUFFLE_SEED', 42))
				print(f"Shuffling training dataset with seed={shuffle_seed}...")
				dataset[dataset_split] = dataset[dataset_split].shuffle(seed=shuffle_seed)
				print(f"Dataset shuffled successfully!")
		else:
			raise FileNotFoundError(f"Neither parquet nor jsonl file found for {dataset_split}. "
			                       f"Checked: {parquet_path} and {jsonl_path}")

	# Random sampling: sample 200 random samples for testing
	MAX_SAMPLES = 900
	original_size = len(dataset[dataset_split])
	if original_size > MAX_SAMPLES:
		print(f"\n{'='*60}")
		print(f"Random Sampling: Selecting {MAX_SAMPLES} samples from {original_size} total samples")
		print(f"{'='*60}")
		
		# Get random seed from environment variable or use default
		sample_seed = int(os.environ.get('SAMPLE_SEED', 42))
		np.random.seed(sample_seed)
		
		if use_parquet:
			# For ParquetDataset, sample indices and create new dataset
			indices = np.random.choice(original_size, size=MAX_SAMPLES, replace=False)
			indices = sorted(indices)  # Keep original order for reproducibility
			
			# Create sampled dataset dict
			sampled_dict = {
				'query': [dataset[dataset_split].data['query'][i] for i in indices],
				'sugg_idx': [dataset[dataset_split].data['sugg_idx'][i] for i in indices],
				'text_list': [dataset[dataset_split].data['text_list'][i] for i in indices]
			}
			dataset[dataset_split] = ParquetDataset(sampled_dict)
			print(f"Sampled {MAX_SAMPLES} samples (seed={sample_seed})")
		elif use_msdata:
			# For MSDATADataset, sample indices and create new dataset
			indices = np.random.choice(original_size, size=MAX_SAMPLES, replace=False)
			indices = sorted(indices.tolist())  # Convert to sorted list
			dataset[dataset_split] = dataset[dataset_split].select(indices)
			print(f"Sampled {MAX_SAMPLES} samples (seed={sample_seed})")
		else:
			# For jsonl (datasets library), use select
			indices = np.random.choice(original_size, size=MAX_SAMPLES, replace=False)
			indices = sorted(indices.tolist())  # Convert to sorted list
			dataset[dataset_split] = dataset[dataset_split].select(indices)
			print(f"Sampled {MAX_SAMPLES} samples (seed={sample_seed})")
	else:
		print(f"\nDataset size ({original_size}) is less than or equal to {MAX_SAMPLES}, using all samples")

	print(f'len(dataset["{dataset_split}"]) is', len(dataset[dataset_split]))
	if hasattr(dataset[dataset_split], 'column_names'):
		print(f'The dataset contains the following keys:', dataset[dataset_split].column_names)
	else:
		# For datasets library format
		print(f'The dataset contains the following keys:', list(dataset[dataset_split].features.keys()))
	
	# Load cache once to check for sources before processing
	SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
	cache_file_path = os.environ.get('GLOBAL_CACHE_FILE', os.path.join(SCRIPT_DIR, 'global_cache.json'))
	try:
		loaded_cache = json.load(open(cache_file_path, encoding='utf-8'), strict=False)
		print(f"Cache loaded for source checking: {len(loaded_cache)} queries")
	except FileNotFoundError:
		print(f"Warning: Cache file {cache_file_path} not found. Will attempt to process all queries.")
		loaded_cache = {}
	except json.JSONDecodeError as e:
		print(f"Warning: Cache file corrupted at {e}. Will attempt to process all queries.")
		loaded_cache = {}
	
	# Choose execution mode based on environment variable
	use_concurrent = os.environ.get('USE_CONCURRENT', 'True').lower() == 'true'
	max_concurrent = int(os.environ.get('MAX_WORKERS', '10'))  
	
	if use_concurrent:
		results = await run_concurrent_async(dataset, dataset_split, results_file_path, max_concurrent=max_concurrent, loaded_cache=loaded_cache)
	else:
		results = await run_sequential_async(dataset, dataset_split, results_file_path, loaded_cache=loaded_cache)
	
	# Print summary
	print("\n" + "="*60)
	print("PROCESSING SUMMARY")
	print("="*60)
	successful = sum(1 for r in results if r[3] is None)
	failed = sum(1 for r in results if r[3] is not None and not r[3].startswith("SKIPPED"))
	skipped = sum(1 for r in results if r[3] is not None and r[3].startswith("SKIPPED"))
	
	# Count total strategy results saved
	try:
		with open(results_file_path, 'r', encoding='utf-8') as f:
			saved_data = json.load(f)
			total_records = len(saved_data.get('results', []))
			unique_queries = len(set(r.get('query', '') for r in saved_data.get('results', [])))
	except:
		total_records = 0
		unique_queries = 0
	
	print(f"Total queries processed: {len(results)}")
	print(f"Successful queries: {successful}")
	print(f"Failed queries: {failed}")
	print(f"Skipped queries (no sources in cache): {skipped}")
	print(f"Total strategy results saved: {total_records}")
	print(f"Unique queries in results: {unique_queries}")
	print(f"Average strategies per query: {total_records / unique_queries if unique_queries > 0 else 0:.7f}")
	
	if skipped > 0:
		print("\nSkipped queries (no sources in cache):")
		skipped_queries = [(i, q, e) for i, q, r, e in results if e and e.startswith("SKIPPED")]
		for i, query, error in skipped_queries[:10]:  # Show first 10
			print(f"  - Query {i+1}: {query[:60]}... | Reason: {error}")
		if len(skipped_queries) > 10:
			print(f"  ... and {len(skipped_queries) - 10} more skipped queries")
	
	if failed > 0:
		print("\nFailed queries:")
		for i, query, result, error in results:
			if error is not None and not error.startswith("SKIPPED"):
				print(f"  - Query {i+1}: {query[:50]}... | Error: {error}")
	
	print(f"\n[Results] All results saved to: {results_file_path}")
	print("="*60)


if __name__ == '__main__':
	# Run the async main function
	asyncio.run(main_async())
