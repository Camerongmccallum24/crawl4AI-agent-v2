"""
insert_docs.py
--------------
Command-line utility to crawl any URL using Crawl4AI, detect content type (sitemap, .txt, or regular page),
use the appropriate crawl method, chunk the resulting Markdown into <1000 character blocks by header hierarchy,
and insert all chunks into Supabase with metadata.

Usage:
    python insert_docs.py <URL> [--table-name ...] [--embedding-model ...]
"""
import argparse
import sys
import re
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
import requests
from utils import add_documents_to_supabase

def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    """Hierarchically splits markdown by #, ##, ### headers, then by characters, to ensure all chunks < max_len."""
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    # First try to split by headers
    chunks = []
    
    # Look for any level of headers
    header_patterns = [r'^# .+$', r'^## .+$', r'^### .+$', r'^#### .+$']
    
    for pattern in header_patterns:
        header_chunks = split_by_header(markdown, pattern)
        if len(header_chunks) > 1:  # If we found headers, use them
            chunks = header_chunks
            break
    
    # If no headers found, split by paragraphs or sentences
    if not chunks:
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', markdown)
        if len(paragraphs) > 1:
            chunks = paragraphs
        else:
            # Split by sentences
            sentences = re.split(r'[.!?]+', markdown)
            chunks = [s.strip() for s in sentences if s.strip()]
    
    # If still no chunks, just split by character count
    if not chunks:
        chunks = [markdown]

    final_chunks = []

    for c in chunks:
        if len(c) > max_len:
            # Split large chunks by character count
            for i in range(0, len(c), max_len):
                chunk = c[i:i+max_len].strip()
                if chunk:
                    final_chunks.append(chunk)
        else:
            if c.strip():
                final_chunks.append(c.strip())

    return [c for c in final_chunks if c and len(c) > 50]  # Filter out very short chunks

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    return url.endswith('.txt')

async def crawl_recursive_internal_links(start_urls, max_depth=3, max_concurrent=10) -> List[Dict[str,Any]]:
    """Recursive crawl using logic from 5-crawl_recursive_internal_links.py. Returns list of dicts with url and markdown."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            print(f"Depth {depth}: Processing {len(current_urls)} URLs")
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            print(f"URLs to crawl: {len(urls_to_crawl)}")
            if not urls_to_crawl:
                print("No more URLs to crawl")
                break

            # Process URLs in batches to control concurrency
            batch_size = max_concurrent
            for i in range(0, len(urls_to_crawl), batch_size):
                batch_urls = urls_to_crawl[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}: {len(batch_urls)} URLs")
                results = await crawler.arun_many(urls=batch_urls, config=run_config, dispatcher=dispatcher)
                
                next_level_urls = set()

                for result in results:
                    norm_url = normalize_url(result.url)
                    visited.add(norm_url)

                    if result.success and result.markdown:
                        print(f"Successfully crawled: {result.url} ({len(result.markdown)} chars)")
                        results_all.append({'url': result.url, 'markdown': result.markdown})
                        # Collect internal links for next level
                        if result.links and "internal" in result.links:
                            internal_links = result.links["internal"]
                            print(f"Found {len(internal_links)} internal links")
                            for link in internal_links:
                                if isinstance(link, dict) and "href" in link:
                                    next_url = normalize_url(link["href"])
                                    if next_url not in visited and next_url.startswith("https://docs.pydantic.dev"):
                                        next_level_urls.add(next_url)
                    else:
                        print(f"Failed to crawl: {result.url} - {result.error_message}")

                print(f"Next level URLs: {len(next_level_urls)}")
                current_urls = next_level_urls

    return results_all

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str,Any]]:
    """Batch crawl using logic from 3-crawl_sitemap_in_parallel.py."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Process URLs in batches to control concurrency
        batch_size = max_concurrent
        all_results = []
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            results = await crawler.arun_many(urls=batch_urls, config=crawl_config, dispatcher=dispatcher)
            all_results.extend([{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown])
        
        return all_results

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def main():
    parser = argparse.ArgumentParser(description="Insert crawled docs into Supabase")
    parser.add_argument("url", help="URL to crawl (regular, .txt, or sitemap)")
    parser.add_argument("--table-name", default="documents", help="Supabase table name")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--max-depth", type=int, default=3, help="Recursion depth for regular URLs")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max parallel browser sessions")
    parser.add_argument("--batch-size", type=int, default=100, help="Supabase insert batch size")
    args = parser.parse_args()

    # Detect URL type
    url = args.url
    if is_txt(url):
        print(f"Detected .txt/markdown file: {url}")
        crawl_results = asyncio.run(crawl_markdown_file(url))
    elif is_sitemap(url):
        print(f"Detected sitemap: {url}")
        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            print("No URLs found in sitemap.")
            sys.exit(1)
        crawl_results = asyncio.run(crawl_batch(sitemap_urls, max_concurrent=args.max_concurrent))
    else:
        print(f"Detected regular URL: {url}")
        crawl_results = asyncio.run(crawl_recursive_internal_links([url], max_depth=args.max_depth, max_concurrent=args.max_concurrent))

    # Chunk and collect metadata
    print(f"Processing {len(crawl_results)} crawled documents...")
    ids, documents, metadatas = [], [], []
    chunk_idx = 0
    for doc in crawl_results:
        url = doc['url']
        md = doc['markdown']
        print(f"Processing document: {url} ({len(md)} chars)")
        chunks = smart_chunk_markdown(md, max_len=args.chunk_size)
        print(f"  Created {len(chunks)} chunks")
        for chunk in chunks:
            ids.append(f"chunk-{chunk_idx}")
            documents.append(chunk)
            meta = extract_section_info(chunk)
            meta["chunk_index"] = chunk_idx
            meta["source"] = url
            metadatas.append(meta)
            chunk_idx += 1

    if not documents:
        print("No documents found to insert.")
        sys.exit(1)

    print(f"Inserting {len(documents)} chunks into Supabase table '{args.table_name}'...")

    add_documents_to_supabase(
        table_name=args.table_name,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        batch_size=args.batch_size,
        embedding_model_name=args.embedding_model
    )

    print(f"Successfully added {len(documents)} chunks to Supabase table '{args.table_name}'.")

if __name__ == "__main__":
    main()
