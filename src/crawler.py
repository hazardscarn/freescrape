import asyncio
from typing import Optional, List, Dict, Any
from crawl4ai import AsyncWebCrawler, CacheMode

class ContentCrawler:
    """Content crawler using crawl4ai"""
    def __init__(
        self,
        max_length: Optional[int] = 1000,
        thread_safe: bool = True
    ):
        self.max_length = max_length
        self.thread_safe = thread_safe

    async def _crawl_single_url(self, url: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Crawl a single URL and return its content
        """
        try:
            async with AsyncWebCrawler(thread_safe=self.thread_safe) as crawler:
                result = await crawler.arun(url=url, cache_mode=CacheMode.BYPASS)
                
                if not result or not result.markdown:
                    return {
                        "url": url,
                        "success": False,
                        "content": "",
                        "error": "No content found"
                    }
                
                # Process content length
                content = result.markdown
                length = max_length or self.max_length
                if length:
                    content = content[:length]
                
                return {
                    "url": url,
                    "success": True,
                    "content": content,
                    "error": None
                }
                
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "content": "",
                "error": str(e)
            }

    async def crawl_urls(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs concurrently and return their content
        
        Args:
            urls: List of dictionaries containing url, title, and tool information
            
        Returns:
            List of dictionaries with url, title, tool, and content information
        """
        crawl_tasks = []
        for url_info in urls:
            task = self._crawl_single_url(url_info['url'])
            crawl_tasks.append(task)
        
        # Crawl all URLs concurrently
        crawl_results = await asyncio.gather(*crawl_tasks)
        
        # Combine results with original URL information
        processed_results = []
        for url_info, crawl_result in zip(urls, crawl_results):
            result = {
                "url": url_info['url'],
                "title": url_info.get('title', ''),
                "tool": url_info.get('tool', ''),
                "content": crawl_result['content'] if crawl_result['success'] else '',
                "error": crawl_result['error']
            }
            processed_results.append(result)
            
        return processed_results

    def process_urls(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for crawling multiple URLs
        """
        return asyncio.run(self.crawl_urls(urls))