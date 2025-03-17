import asyncio
from typing import Optional, List, Dict, Any
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig,CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter,PruningContentFilter
from src.google_search import GoogleSearch

class ContentCrawler:
    """Content crawler using crawl4ai"""
    def __init__(
        self,
        query: str,
        max_results: int,
        max_length: Optional[int] = 1000,
        thread_safe: bool = True,
        filter: str = None,
        bm25_threshold: float = 1.2,
        prune_threshold: float = 0.5,
        prune_threshold_type: str = "fixed",
        prune_min_word_threshold: int = 10
    ):
        self.max_length = max_length
        self.thread_safe = thread_safe
        self.query = query
        self.max_results = max_results
        self.search_results=None
        self.filter=filter
        self.bm25_threshold=bm25_threshold
        self.prune_threshold=prune_threshold
        self.prune_threshold_type=prune_threshold_type
        self.prune_min_word_threshold=prune_min_word_threshold

        self.colleced_data=None


        ##Get URL from google search
        self.search_urls()

        ##Set the content filter
        if self.filter=="bm25":
            print("BM25 filter Applied")
            content_filter = BM25ContentFilter(
                        user_query=self.query,
                        bm25_threshold=self.bm25_threshold
                            )
        elif self.filter=="prune":
            print("Prune filter Applied")
            content_filter = PruningContentFilter(
                threshold=self.prune_threshold,
                threshold_type=self.prune_threshold_type,  # or "dynamic"
                min_word_threshold=self.prune_min_word_threshold
            )
        else:
            content_filter = None
            
        ##Create markdown generator
        self.markdown_generator = DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={"ignore_links": True})

        ##Set the crawler configuration
        self.config = CrawlerRunConfig(
            markdown_generator=self.markdown_generator,
            exclude_external_links=True,
            process_iframes=True,
            word_count_threshold=10,
            remove_overlay_elements=True,
            excluded_tags=["nav", "footer", "header"],
            check_robots_txt=True, # Ensures compliance with robots.txt rules for ethical and legal web crawling.
            

        )

        ## Collect the data from google search
        self.colleced_data=self.process_urls()    

        

    def search_urls(self) -> List[Dict[str, str]]:
        """
        Perform Google search and return structured results
        """
        print("Searching for query: ",self.query)
        print("Max results: ",self.max_results)
        gs=GoogleSearch(fixed_max_results=self.max_results)
        self.search_results=gs.search(query=self.query)

    async def _crawl_single_url(self, url: str) -> Dict[str, Any]:
        """
        Crawl a single URL and return its content
        """
        try:
            async with AsyncWebCrawler(thread_safe=self.thread_safe) as crawler:
                result = await crawler.arun(url=url,  config=self.config)
                
                if not result or not result.success:
                    return {
                        "url": url,
                        "success": False,
                        "error": "No content found",
                        "raw_markdown" : "",
                        "fit_markdown" : ""
                    }
                
                # Process content 
                raw_markdown=result.markdown.raw_markdown
                fit_markdown=result.markdown.fit_markdown

                
                return {
                    "url": url,
                    "success": True,
                    "error": None,
                    "raw_markdown" : raw_markdown,
                    "fit_markdown" : fit_markdown
                }
                
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "raw_markdown" : "",
                "fit_markdown" : ""
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
                "error": crawl_result['error'],
                "raw_markdown" : crawl_result['raw_markdown'] if crawl_result['success'] else '',
                "fit_markdown" : crawl_result['fit_markdown'] if crawl_result['success'] else ''
            }
            processed_results.append(result)
            
        return processed_results

    def process_urls(self) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for crawling multiple URLs
        """
        if self.search_results:
            return asyncio.run(self.crawl_urls(self.search_results))
        else:
            return []
