import os
import asyncio
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from src.search_crawler import ContentCrawler



def test_content_crawler(query: str, max_results: int = 3, filter_type: str = None):
    """
    Test the ContentCrawler class by running a search and saving the raw and fit markdown files
    
    Args:
        query: Search query
        max_results: Maximum number of search results to process
        filter_type: Type of content filter to apply ("bm25", "prune", or None)
    """
    print(f"Testing ContentCrawler with query: {query}")
    
    # Create output directory if it doesn't exist
    output_dir = "crawler_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the crawler
    crawler = ContentCrawler(
        query=query,
        max_results=max_results,
        filter=filter_type,
        bm25_threshold=1.2,
        prune_threshold=0.5,
    )
    
    # Check if we got results
    if not crawler.colleced_data:
        print("No results collected")
        return
    
    # Save the results
    for i, result in enumerate(crawler.colleced_data):
        # Create a sanitized filename from the URL
        url_filename = result['url'].replace('https://', '').replace('http://', '')
        url_filename = ''.join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in url_filename)
        url_filename = url_filename[:50]  # Limit filename length
        
        # Save raw markdown
        if result['raw_markdown']:
            raw_path = os.path.join(output_dir, f"{i+1}_{url_filename}_raw.md")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(result['raw_markdown'])
            print(f"Saved raw markdown to: {raw_path}")
        
        # Save fit markdown
        if result['fit_markdown']:
            fit_path = os.path.join(output_dir, f"{i+1}_{url_filename}_fit.md")
            with open(fit_path, "w", encoding="utf-8") as f:
                f.write(result['fit_markdown'])
            print(f"Saved fit markdown to: {fit_path}")

if __name__ == "__main__":
    # Test ContentCrawler
    test_content_crawler(
        query="Interstellar Ending Explained",
        max_results=6,
        filter_type="prune"  # Try "bm25", "prune", or None
    )
    