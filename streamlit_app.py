from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
import re
import sys
from typing import List, Dict, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
import requests

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from rag_agent import agent, RAGDeps
from utils import add_documents_to_supabase

# Import crawling functions from insert_docs.py
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

load_dotenv()

async def get_agent_deps():
    return RAGDeps(
        table_name="documents",
        embedding_model="all-MiniLM-L6-v2"
    )

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
    # Simplified Windows-compatible browser config
    browser_config = BrowserConfig(
        headless=True, 
        verbose=False,
        # Minimal Windows-specific settings
        extra_args=[
            '--no-sandbox',
            '--disable-dev-shm-usage'
        ]
    )
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    # Use a simpler dispatcher for Windows
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

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(max_depth):
                urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
                if not urls_to_crawl:
                    break

                # Process URLs in batches to control concurrency
                batch_size = max_concurrent
                for i in range(0, len(urls_to_crawl), batch_size):
                    batch_urls = urls_to_crawl[i:i+batch_size]
                    results = await crawler.arun_many(urls=batch_urls, config=run_config, dispatcher=dispatcher)
                    
                    next_level_urls = set()

                    for result in results:
                        norm_url = normalize_url(result.url)
                        visited.add(norm_url)

                        if result.success and result.markdown:
                            results_all.append({'url': result.url, 'markdown': result.markdown})
                            # Collect internal links for next level
                            if result.links and "internal" in result.links:
                                internal_links = result.links["internal"]
                                for link in internal_links:
                                    if isinstance(link, dict) and "href" in link:
                                        next_url = normalize_url(link["href"])
                                        if next_url not in visited and next_url.startswith("https://docs.pydantic.dev"):
                                            next_level_urls.add(next_url)

                    current_urls = next_level_urls
    except Exception as e:
        st.error(f"Browser error: {str(e)}")
        # Fallback to single page crawl
        return await crawl_single_page(start_urls[0])

    return results_all

async def crawl_single_page(url: str) -> List[Dict[str,Any]]:
    """Fallback single page crawl for Windows compatibility."""
    # Simplified browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            '--no-sandbox',
            '--disable-dev-shm-usage'
        ]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success and result.markdown:
                return [{'url': url, 'markdown': result.markdown}]
            else:
                st.error(f"Failed to crawl {url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                return []
    except Exception as e:
        st.error(f"Single page crawl failed: {str(e)}")
        return []

async def crawl_simple_page(url: str) -> List[Dict[str,Any]]:
    """Ultra-simple crawl with minimal configuration."""
    try:
        # Use the most basic configuration possible
        browser_config = BrowserConfig(headless=True)
        crawl_config = CrawlerRunConfig()
        
        st.info(f"🔄 Starting simple crawl for: {url}")
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            st.info("✅ Crawler created successfully")
            result = await crawler.arun(url=url, config=crawl_config)
            st.info(f"✅ Crawl completed. Success: {result.success}")
            
            if result.success and result.markdown:
                content_length = len(result.markdown)
                st.success(f"✅ Simple crawl successful! Retrieved {content_length} characters")
                st.info(f"📝 First 100 chars: {result.markdown[:100]}...")
                return [{'url': url, 'markdown': result.markdown}]
            else:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                st.error(f"Simple crawl failed for {url}: {error_msg}")
                st.info(f"Result object: {result}")
                return []
    except Exception as e:
        st.error(f"Simple crawl exception: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return []

async def crawl_markdown_file(url: str) -> List[Dict[str,Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    # Simplified browser config
    browser_config = BrowserConfig(
        headless=True,
        extra_args=[
            '--no-sandbox',
            '--disable-dev-shm-usage'
        ]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success and result.markdown:
                return [{'url': url, 'markdown': result.markdown}]
            else:
                st.error(f"Markdown file crawl failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                return []
    except Exception as e:
        st.error(f"Markdown file crawl failed: {str(e)}")
        return []

def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            st.error(f"Error parsing sitemap XML: {e}")

    return urls

async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str,Any]]:
    """Batch crawl using logic from 3-crawl_sitemap_in_parallel.py."""
    # Simplified browser config
    browser_config = BrowserConfig(
        headless=True, 
        verbose=False,
        extra_args=[
            '--no-sandbox',
            '--disable-dev-shm-usage'
        ]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs in batches to control concurrency
            batch_size = max_concurrent
            all_results = []
            
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i+batch_size]
                results = await crawler.arun_many(urls=batch_urls, config=crawl_config, dispatcher=dispatcher)
                all_results.extend([{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown])
            
            return all_results
    except Exception as e:
        st.error(f"Batch crawl failed: {str(e)}")
        return []

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extracts headers and stats from a chunk."""
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

async def crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 1000) -> Dict[str, Any]:
    """Main crawling function that handles different URL types and returns results."""
    try:
        # Detect URL type
        if is_txt(url):
            st.info(f"Detected .txt/markdown file: {url}")
            crawl_results = await crawl_markdown_file(url)
        elif is_sitemap(url):
            st.info(f"Detected sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return {"success": False, "error": "No URLs found in sitemap."}
            crawl_results = await crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
        else:
            st.info(f"Detected regular URL: {url}")
            # For now, use simple single page crawl since it's working
            crawl_results = await crawl_simple_page(url)
            # TODO: Re-enable recursive crawling once we confirm it works
            # crawl_results = await crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent)

        if not crawl_results:
            return {"success": False, "error": "No content found to crawl."}

        # Chunk and collect metadata
        ids, documents, metadatas = [], [], []
        chunk_idx = 0
        
        for doc in crawl_results:
            doc_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, max_len=chunk_size)
            
            for chunk in chunks:
                ids.append(f"chunk-{chunk_idx}")
                documents.append(chunk)
                meta = extract_section_info(chunk)
                meta["chunk_index"] = chunk_idx
                meta["source"] = doc_url
                metadatas.append(meta)
                chunk_idx += 1

        if not documents:
            return {"success": False, "error": "No documents found to insert."}

        # Insert into Supabase
        add_documents_to_supabase(
            table_name="documents",
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            batch_size=100,
            embedding_model_name="all-MiniLM-L6-v2"
        )

        return {
            "success": True,
            "documents_crawled": len(crawl_results),
            "chunks_created": len(documents),
            "urls": [doc['url'] for doc in crawl_results]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)             

async def run_agent_with_streaming(user_input):
    async with agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
    ) as result:
        async for message in result.stream_text(delta=True):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())

async def main():
    st.set_page_config(
        page_title="Crawl4AI RAG Agent",
        page_icon="🕷️",
        layout="wide"
    )
    
    st.title("🕷️ Crawl4AI RAG Agent")
    st.markdown("Crawl websites and chat with your knowledge base!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_deps" not in st.session_state:
        st.session_state.agent_deps = await get_agent_deps()

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🤖 Chat with AI", "🕷️ Crawl Websites"])
    
    with tab1:
        st.header("Chat with Your Knowledge Base")
        
        # Display all messages from the conversation so far
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)

        # Chat input for the user
        user_input = st.chat_input("What do you want to know about your crawled content?")

        if user_input:
            # Display user prompt in the UI
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display the assistant's partial response while streaming
            with st.chat_message("assistant"):
                # Create a placeholder for the streaming text
                message_placeholder = st.empty()
                full_response = ""
                
                # Properly consume the async generator with async for
                generator = run_agent_with_streaming(user_input)
                async for message in generator:
                    full_response += message
                    message_placeholder.markdown(full_response + "▌")
                
                # Final response without the cursor
                message_placeholder.markdown(full_response)

    with tab2:
        st.header("Crawl Websites")
        st.markdown("Enter a URL to crawl and add to your knowledge base.")
        
        # URL input
        url = st.text_input(
            "Enter URL to crawl:",
            placeholder="https://docs.pydantic.dev/2.11/",
            help="Supports regular websites, .txt files, and sitemaps"
        )
        
        # Crawling options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_depth = st.slider(
                "Max Depth",
                min_value=1,
                max_value=5,
                value=1,  # Start with 1 for Windows compatibility
                help="How deep to crawl internal links"
            )
        
        with col2:
            max_concurrent = st.slider(
                "Max Concurrent",
                min_value=1,
                max_value=10,  # Reduced for Windows
                value=3,       # Start with 3 for Windows compatibility
                help="Number of parallel browser sessions"
            )
        
        with col3:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=1000,
                help="Maximum characters per chunk"
            )
        
        # Add Windows compatibility info
        st.info("💡 **Windows Tip**: Start with depth=1 and concurrent=3 for better compatibility. Increase gradually if successful.")
        
        # Test button for debugging
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test Single Page", help="Test crawling a single page to debug issues"):
                if url:
                    with st.spinner("Testing single page crawl..."):
                        try:
                            result = await crawl_simple_page(url)
                            if result:
                                st.success(f"✅ Test successful! Crawled: {result[0]['url']}")
                                st.info(f"📝 Content length: {len(result[0]['markdown'])} characters")
                            else:
                                st.error("❌ Test failed - no content retrieved")
                        except Exception as e:
                            st.error(f"❌ Test error: {str(e)}")
                else:
                    st.warning("Please enter a URL first.")
        
        with col2:
            # Crawl button
            if st.button("🕷️ Start Crawling", type="primary", use_container_width=True):
                if url:
                    with st.spinner("Crawling in progress..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Run the crawling
                            result = await crawl_url(url, max_depth, max_concurrent, chunk_size)
                            
                            if result["success"]:
                                st.success(f"✅ Crawling completed successfully!")
                                st.info(f"📄 Crawled {result['documents_crawled']} documents")
                                st.info(f"📝 Created {result['chunks_created']} chunks")
                                
                                # Show crawled URLs
                                with st.expander("📋 View Crawled URLs"):
                                    for i, crawled_url in enumerate(result['urls'], 1):
                                        st.write(f"{i}. {crawled_url}")
                                
                                # Clear the progress indicators
                                progress_bar.empty()
                                status_text.empty()
                            else:
                                st.error(f"❌ Crawling failed: {result['error']}")
                                progress_bar.empty()
                                status_text.empty()
                                
                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
                            progress_bar.empty()
                            status_text.empty()
                else:
                    st.warning("Please enter a URL to crawl.")

if __name__ == "__main__":
    asyncio.run(main())
