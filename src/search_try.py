import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
# from readabilipy import simple_json_from_html_string
# import trafilatura
import nltk
from openai import OpenAI, AsyncOpenAI
import os
import pickle
import uuid
import time
import asyncio
from config_loader import get_effective_api_config, use_async, use_local_llm, get_local_llm_model

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

# Directory for search-time cleaning usage logs
SEARCH_USAGE_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response_usages")
os.makedirs(SEARCH_USAGE_LOG_DIR, exist_ok=True)


async def clean_source_gpt35_async(source: str) -> str:
    """Async version of clean_source_gpt35 using AsyncOpenAI client."""
    model = "gpt-3.5-turbo"
    if use_local_llm():
        model = get_local_llm_model()
        print(f'Using local LLM model for cleaning: {model}')

    for idx in range(8):
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Clean and refine the extracted text from a website. Remove any unwanted "
                            "content such as headers, sidebars, and navigation menus. Retain only the "
                            "main content of the page and ensure that the text is well-formatted and "
                            "free of HTML tags, special characters, and any other irrelevant "
                            "information. Refined text should contain the main intended readable text. "
                            "Apply markdown formatting when outputting the answer.\n\nHere is the "
                            "website:\n```html_text\n"
                            f"{source.strip()}```"
                        ),
                    },
                ],
                temperature=0,
                max_tokens=1800,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            break
        except Exception as e:
            print(f'Error while cleaning text with LLM: {e}')
            source = source[:-int(800 * (1 + idx / 2))]
            await asyncio.sleep(3 + idx**2)

    # Store the usage in a tidy subdirectory with readable filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"{timestamp}_{model}_{uuid.uuid4().hex[:8]}_search_usage.pkl"
    log_path = os.path.join(SEARCH_USAGE_LOG_DIR, log_name)
    with open(log_path, "wb") as f:
        pickle.dump(response.usage, f)

    tex = response.choices[0].message.content.strip()
    new_lines = [""]
    for line in tex.split('\n\n'):
        new_lines[-1] +=line+'\n'
        if len(nltk.sent_tokenize(line))!=1:
            new_lines.append("")
    new_lines = [x.strip() for x in new_lines]
    return "\n\n".join(new_lines)


def clean_source_gpt35(source: str) -> str:
    """Synchronous wrapper that calls async or sync version based on config."""
    if use_async():
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create new loop in thread to avoid nested loop error
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, clean_source_gpt35_async(source))
                return future.result()
        except RuntimeError:
            # No running loop - we can create one safely
            return asyncio.run(clean_source_gpt35_async(source))
    else:
        # Synchronous version (original logic)
        model = "gpt-3.5-turbo"
        if use_local_llm():
            model = get_local_llm_model()
            print(f'Using local LLM model for cleaning: {model}')

        for idx in range(8):
            try:
                response = sync_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Clean and refine the extracted text from a website. Remove any unwanted "
                                "content such as headers, sidebars, and navigation menus. Retain only the "
                                "main content of the page and ensure that the text is well-formatted and "
                                "free of HTML tags, special characters, and any other irrelevant "
                                "information. Refined text should contain the main intended readable text. "
                                "Apply markdown formatting when outputting the answer.\n\nHere is the "
                                "website:\n```html_text\n"
                                f"{source.strip()}```"
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=1800,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                break
            except Exception as e:
                print(f'Error while cleaning text with LLM: {e}')
                source = source[:-int(800 * (1 + idx / 2))]
                time.sleep(3 + idx**2)

        # Store the usage in a tidy subdirectory with readable filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_name = f"{timestamp}_{model}_{uuid.uuid4().hex[:8]}_search_usage.pkl"
        log_path = os.path.join(SEARCH_USAGE_LOG_DIR, log_name)
        with open(log_path, "wb") as f:
            pickle.dump(response.usage, f)

        tex = response.choices[0].message.content.strip()
        new_lines = [""]
        for line in tex.split('\n\n'):
            new_lines[-1] +=line+'\n'
            if len(nltk.sent_tokenize(line))!=1:
                new_lines.append("")
        new_lines = [x.strip() for x in new_lines]
        return "\n\n".join(new_lines)

def clean_source_text(text: str) -> str:
    return (
        text.strip()
        .replace("\n\n\n", "\n\n")
        .replace("\n\n", " ")
        .replace("  ", " ")
        .replace("\t", "")
        .replace("\n", "")
    )

import time
from pdb import set_trace as bp


def summarize_text_identity(source, query) -> str:
    return source[:8000]


def search_handler(req, source_count = 8):
    query = req

    # GET LINKS
    for _ in range(5):
        try:
            response = requests.get(f"https://www.google.com/search?q={query}")
            print(response.url)
            print(response.text)
            break
        except Exception as e:
            print(f'Error while fetching from Google {e}')
            time.sleep(5)
            continue

    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    print("\n" + "="*60)
    print("Debug: Check Google page content")
    print("="*60)

    page_title = soup.find('title')
    if page_title:
        print(f"Page title: {page_title.get_text()}")

    if 'verify' in html.lower() or 'captcha' in html.lower() or 'unusual traffic' in html.lower():
        print("Warning: Possible captcha/verification page detected!")
        print("    Google may have detected automated requests")

    if 'enablejs' in html.lower() or 'javascript' in html.lower() or 'noscript' in html.lower():
        print("Warning: Page requires JavaScript to display content!")
        print("    requests library cannot execute JavaScript")

    print(f"HTML content length: {len(html)} chars")
    if len(html) < 10000:
        print("Warning: HTML too short, may be lightweight or error page")

    result_containers = soup.find_all(['div', 'h3'], class_=lambda x: x and ('result' in x.lower() or 'g' in x.lower()))
    print(f"Possible result containers found: {len(result_containers)}")

    if 'enablejs' in html or '/httpservice/retry/enablejs' in html:
        print("Found: Page contains 'enablejs' retry link")
        print("    Confirms Google requires JavaScript for search results")

    print("\n--- Analyzing HTML content ---")

    script_tags = soup.find_all('script')
    print(f"Script tags count: {len(script_tags)}")

    script_content_length = sum(len(str(script)) for script in script_tags)
    print(f"Script content length: {script_content_length}")
    print(f"Script content ratio: {script_content_length/len(html)*100:.1f}%")

    import json
    import re
    json_data_found = False
    url_patterns_found = []

    for idx, script in enumerate(script_tags):
        script_text = script.string or ""
        script_str = str(script)

        if 'sourceMappingURL' in script_text or 'sourcemap' in script_text.lower():
            continue

        url_pattern = r'https?://[^\s"\'<>]+'
        urls_in_script = re.findall(url_pattern, script_text)

        relevant_urls = [url for url in urls_in_script
                        if not any(x in url for x in ['google.com', 'gstatic.com', 'googleapis.com', 'doubleclick.net'])]

        if relevant_urls:
            json_data_found = True
            url_patterns_found.extend(relevant_urls[:5])
            print(f"Script #{idx+1} found possible search result URLs:")
            for url in relevant_urls[:3]:
                print(f"   - {url[:80]}...")

        if '{"' in script_text or "['" in script_text or 'var ' in script_text:
            if any(keyword in script_text.lower() for keyword in ['result', 'search', 'link', 'href', 'title']):
                print(f"Script #{idx+1} may contain search result data structure")
                lines = script_text.split('\n')
                for line in lines[:10]:
                    if any(keyword in line.lower() for keyword in ['result', 'search', 'link']):
                        print(f"    Sample line: {line[:100]}...")
                        break

    if not json_data_found and not url_patterns_found:
        print("Script tags: No search result URLs found")
        print("    Results are fully loaded via JavaScript")

    div_tags = soup.find_all('div')
    print(f"Div tags count: {len(div_tags)}")

    google_result_classes = ['g', 'tF2Cxc', 'yuRUbf', 'LC20lb']
    for class_name in google_result_classes:
        found = soup.find_all('div', class_=class_name)
        if found:
            print(f"Found Google result container: class='{class_name}' -> {len(found)}")

    print(f"\nHTTP status: {response.status_code}")
    print(f"Response URL: {response.url}")
    print("="*60 + "\n")

    link_tags = soup.find_all('a')
    links = []

    print("\n" + "="*60)
    print("Debug: Analyze Google search result link formats")
    print("="*60)
    print(f"Total <a> tags found: {len(link_tags)}\n")

    url_q_count = 0
    url_url_count = 0
    http_count = 0
    https_count = 0
    other_count = 0
    none_count = 0

    print("First 30 link href values:")
    print("-"*60)
    for idx, link in enumerate(link_tags[:30]):
        href = link.get('href')
        if href is None:
            none_count += 1
            print(f"{idx+1:2d}. [None]")
        elif href.startswith('/url?q='):
            url_q_count += 1
            print(f"{idx+1:2d}. [Match] /url?q=... -> {href[:80]}...")
        elif href.startswith('/url?url='):
            url_url_count += 1
            print(f"{idx+1:2d}. [Other] /url?url=... -> {href[:80]}...")
        elif href.startswith('http://'):
            http_count += 1
            print(f"{idx+1:2d}. [Other] http://... -> {href[:80]}...")
        elif href.startswith('https://'):
            https_count += 1
            print(f"{idx+1:2d}. [Other] https://... -> {href[:80]}...")
        else:
            other_count += 1
            print(f"{idx+1:2d}. [Other] {href[:80]}...")

    print("-"*60)
    print(f"\nStats (first 30 links):")
    print(f"  /url?q= format: {url_q_count}")
    print(f"  /url?url= format: {url_url_count}")
    print(f"  http:// format: {http_count}")
    print(f"  https:// format: {https_count}")
    print(f"  Other format: {other_count}")
    print(f"  No href: {none_count}")
    print("="*60 + "\n")

    for link in link_tags:
        href = link.get('href')

        if href and href.startswith('/url?q='):
            cleaned_href = href.replace('/url?q=', '').split('&')[0]

            if cleaned_href not in links:
                links.append(cleaned_href)
                print(cleaned_href)

    print(f"\n[Debug] Extracted {len(links)} links\n")

    exclude_list = ["google", "facebook", "twitter", "instagram", "youtube", "tiktok","quora"]
    filtered_links = []
    links = set(list(links))
    for link in links:
        try:
            if urlparse(link).hostname.split('.')[1] not in exclude_list:
                filtered_links.append(link)
        except: ...
    filtered_links = [link for idx, link in enumerate(links) if urlparse(link).hostname.split('.')[1] not in exclude_list and links.index(link) == idx]

    final_links = filtered_links#[:source_count]

    # SCRAPE TEXT FROM LINKS
    sources = []

    for link in final_links:
        print(f'Will be loading link {link}')
        try:
            for _ in range(5):
                downloaded = trafilatura.fetch_url(link)
                source_text = trafilatura.extract(downloaded)
                if source_text is not None:
                    break

                print(f'Error fetching link {link}')
                time.sleep(4)
            if source_text is None:
                continue
            response = requests.get(link, timeout=15)
        except Exception as e:
            continue
        print('Link Loaded')
        html = response.text
        try:
            html = simple_json_from_html_string(html)
            html_text = html['content']
        except:
            try:
                from readabilipy.extractors import extract_title
                {
                    "title": extract_title(html),
                    "content": str(html)
                }
            except:
                continue
        if len(html_text) < 400:
            continue
        print(len(html_text))

        soup = BeautifulSoup(html_text, 'html.parser')

        if source_text:
            source_text = clean_source_text(source_text)
            print('Going to call openai')
            raw_source = source_text
            source_text = clean_source_gpt35(source_text[:8000])
            summary_text = summarize_text_identity(source_text, query)
            sources.append({'url': link, 'text': f'Title: {html["title"]}\nSummary:' + summary_text, 'raw_source' : raw_source, 'source' : source_text, 'summary' : summary_text})
            print('Openai Called')
        if len(sources) == source_count:
            break
    return {'sources': sources}

if __name__ == '__main__':
    import sys
    search_handler('What is Generative Engine Optimization?')
    import json
    print(json.dumps(search_handler(sys.argv[1]), indent = 2))
