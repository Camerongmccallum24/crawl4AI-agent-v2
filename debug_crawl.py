"""
Detailed debugging script for crawling issues on Windows
"""
import asyncio
import sys
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def debug_crawl():
    """Detailed debugging of crawling process."""
    print("🔍 Starting detailed crawl debugging...")
    print("=" * 60)
    
    url = "https://docs.pydantic.dev/2.11/"
    print(f"Target URL: {url}")
    
    # Test 1: Most basic configuration
    print("\n1️⃣ Testing most basic configuration...")
    try:
        browser_config = BrowserConfig(headless=True)
        print(f"Browser config created: {browser_config}")
        
        crawl_config = CrawlerRunConfig()
        print(f"Crawl config created: {crawl_config}")
        
        print("Creating AsyncWebCrawler...")
        async with AsyncWebCrawler(config=browser_config) as crawler:
            print("✅ Crawler created successfully")
            print("Attempting to crawl...")
            
            result = await crawler.arun(url=url, config=crawl_config)
            print(f"✅ Crawl completed")
            print(f"Result success: {result.success}")
            print(f"Result has markdown: {hasattr(result, 'markdown')}")
            
            if hasattr(result, 'markdown'):
                print(f"Markdown length: {len(result.markdown) if result.markdown else 0}")
                if result.markdown:
                    print(f"First 200 chars: {result.markdown[:200]}...")
                else:
                    print("❌ Markdown is empty or None")
            
            if hasattr(result, 'error_message'):
                print(f"Error message: {result.error_message}")
            
            return result
            
    except Exception as e:
        print(f"❌ Exception in basic crawl: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_with_requests():
    """Test if we can at least fetch the page with requests."""
    print("\n2️⃣ Testing with requests library...")
    try:
        import requests
        response = requests.get("https://docs.pydantic.dev/2.11/")
        print(f"Status code: {response.status_code}")
        print(f"Content length: {len(response.text)}")
        print(f"First 200 chars: {response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Requests failed: {str(e)}")
        return False

async def main():
    print("🧪 Detailed Crawl4AI Debugging on Windows")
    print("=" * 60)
    
    # Test basic web connectivity
    web_ok = await test_with_requests()
    print(f"Web connectivity: {'✅ OK' if web_ok else '❌ Failed'}")
    
    # Test crawling
    result = await debug_crawl()
    
    if result and result.success and result.markdown:
        print("\n✅ SUCCESS: Crawling is working!")
        print(f"Content length: {len(result.markdown)} characters")
    else:
        print("\n❌ FAILED: Crawling is not working properly")
        if result:
            print(f"Success: {result.success}")
            print(f"Has markdown: {hasattr(result, 'markdown')}")
            if hasattr(result, 'error_message'):
                print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
