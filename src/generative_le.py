from openai import OpenAI, AsyncOpenAI
import time
import os
import pickle
import uuid
import asyncio
from config_loader import get_effective_api_config, use_async, use_local_llm, get_local_llm_model

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or Japanese should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""

# Initialize OpenAI clients (v1+ SDK)
api_config = get_effective_api_config()
sync_client = OpenAI(
    api_key=api_config['api_key'],
    base_url=api_config['base_url'],
)
async_client = AsyncOpenAI(
    api_key=api_config['api_key'],
    base_url=api_config['base_url'],
)

# Directory to store usage logs (kept tidy under src/)
USAGE_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response_usages_16k")
os.makedirs(USAGE_LOG_DIR, exist_ok=True)

async def generate_answer_async(query, sources, num_completions, temperature = 0.5, verbose = False, model = 'gpt-3.5-turbo-16k'):
    """Async version of generate_answer using AsyncOpenAI client."""
    # Check if sources is empty
    if not sources or len(sources) == 0:
        raise ValueError(f'Cannot generate answer: sources list is empty for query "{query}". Please ensure sources are provided.')

    # Filter out empty or None sources
    sources = [s for s in sources if s and len(str(s).strip()) > 0]
    if len(sources) == 0:
        raise ValueError(f'Cannot generate answer: all sources are empty for query "{query}".')

    # Override model if using local LLM
    if use_local_llm():
        model = get_local_llm_model()
        print(f'Using local LLM model: {model}')

    source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
    prompt = query_prompt.format(query = query, source_text = source_text)

    while True:
        try:
            print('Running LLM Model (Async)')
            response = await async_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                top_p=1,
                n=num_completions,
            )
            print('Response Done')
            break
        except Exception as e:
            print(f'Error in calling LLM API: {e}')
            await asyncio.sleep(15)
            continue

    # Save compact usage log with readable filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"{timestamp}_{model}_{uuid.uuid4().hex[:8]}_usage.pkl"
    log_path = os.path.join(USAGE_LOG_DIR, log_name)
    with open(log_path, "wb") as f:
        pickle.dump(response.usage, f)

    return [x.message.content + "\n" for x in response.choices]


def generate_answer(query, sources, num_completions, temperature = 0.5, verbose = False, model = 'gpt-3.5-turbo-16k'):
    """Synchronous wrapper that calls async or sync version based on config."""
    if use_async():
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create new loop in thread to avoid nested loop error
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    generate_answer_async(query, sources, num_completions, temperature, verbose, model)
                )
                return future.result()
        except RuntimeError:
            # No running loop - we can create one safely
            return asyncio.run(
                generate_answer_async(query, sources, num_completions, temperature, verbose, model)
            )
    else:
        # Synchronous version (original logic)
        # Check if sources is empty
        if not sources or len(sources) == 0:
            raise ValueError(f'Cannot generate answer: sources list is empty for query "{query}". Please ensure sources are provided.')

        # Filter out empty or None sources
        sources = [s for s in sources if s and len(str(s).strip()) > 0]
        if len(sources) == 0:
            raise ValueError(f'Cannot generate answer: all sources are empty for query "{query}".')

        # Override model if using local LLM
        if use_local_llm():
            model = get_local_llm_model()
            print(f'Using local LLM model: {model}')

        source_text = '\n\n'.join(['### Source '+str(idx+1)+':\n'+source + '\n\n\n' for idx, source in enumerate(sources)])
        prompt = query_prompt.format(query = query, source_text = source_text)

        while True:
            try:
                print('Running LLM Model (Sync)')
                response = sync_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    top_p=1,
                    n=num_completions,
                )
                print('Response Done')
                break
            except Exception as e:
                print(f'Error in calling LLM API: {e}')
                time.sleep(15)
                continue

        # Save compact usage log with readable filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_name = f"{timestamp}_{model}_{uuid.uuid4().hex[:8]}_usage.pkl"
        log_path = os.path.join(USAGE_LOG_DIR, log_name)
        with open(log_path, "wb") as f:
            pickle.dump(response.usage, f)

        return [x.message.content + "\n" for x in response.choices]
