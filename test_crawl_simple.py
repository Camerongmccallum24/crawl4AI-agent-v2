"""
Simple test script to debug crawling issues on Windows
"""
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def test_simple_crawl():
    """Test the most basic crawling functionality."""
    print("Testing simple crawl...")
    
    # Test 1: Basic configuration
    try:
        browser_config = BrowserConfig(headless=True)
        crawl_config = CrawlerRunConfig()
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url="https://docs.pydantic.dev/2.11/", config=crawl_config)
            if result.success:
                print(f"✅ Basic crawl successful! Content length: {len(result.markdown)}")
                return True
            else:
                print(f"❌ Basic crawl failed: {result.error_message}")
                return False
    except Exception as e:
        print(f"❌ Basic crawl exception: {str(e)}")
        return False

async def test_windows_config():
    """Test Windows-specific configuration."""
    print("Testing Windows configuration...")
    
    try:
        browser_config = BrowserConfig(
            headless=True,
            extra_args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        crawl_config = CrawlerRunConfig()
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url="https://docs.pydantic.dev/2.11/", config=crawl_config)
            if result.success:
                print(f"✅ Windows config successful! Content length: {len(result.markdown)}")
                return True
            else:
                print(f"❌ Windows config failed: {result.error_message}")
                return False
    except Exception as e:
        print(f"❌ Windows config exception: {str(e)}")
        return False

async def main():
    print("🧪 Testing Crawl4AI on Windows...")
    print("=" * 50)
    
    # Test 1: Basic crawl
    success1 = await test_simple_crawl()
    print()
    
    # Test 2: Windows config
    success2 = await test_windows_config()
    print()
    
    if success1 or success2:
        print("✅ At least one test passed! Crawling should work.")
    else:
        print("❌ All tests failed. There may be a deeper issue with Playwright installation.")
        print("💡 Try running: playwright install")

if __name__ == "__main__":
    asyncio.run(main())
