import asyncio
from typing import Optional, List, Dict, Any, Type, Union, Literal
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LLMConfig, BrowserConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import os
import json
from src.google_search import GoogleSearch
from pydantic import BaseModel

# Define supported provider types
ProviderType = Literal[
    "gemini/gemini-1.5-flash",  # Default
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-exp",
    "gemini/gemini-2.0-flash-lite-preview-02-05",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o1-mini",
    "openai/o1-preview",
    "openai/o3-mini",
    "openai/o3-mini-high",
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20240620",
    "ollama/llama3",
    "groq/llama3-70b-8192",
    "groq/llama3-8b-8192",
    "deepseek/deepseek-chat"
]

class StructuredExtraction:
    """
    Content crawler using crawl4ai for structured data extraction
    
    Extracts structured data based on a provided schema (Pydantic model or dict)
    and specific instructions for the LLM extraction process.
    
    Supports multiple LLM providers including OpenAI, Gemini, Claude, Llama, and more.
    """
    def __init__(
        self,
        query: str=None,
        url_passed: str=None,
        max_results: int=5,
        max_length: Optional[int] = 1000,
        thread_safe: bool = True,
        schema: Union[Dict[str, Any], Type[BaseModel]] = None,
        instruction: str = None,
        headless: bool = True,
        chunk_token_threshold: int = 1000,
        overlap_rate: float = 0.0,
        apply_chunking: bool = True,
        input_format: str = "markdown",
        temperature: float = 0.0,
        max_tokens: int = 800,
        provider: str = "gemini/gemini-1.5-flash",  # Default is Gemini 1.5 Flash
        extract_first_item: bool = False  # Flag to extract first item if result is a list
    ):
        # Validate required inputs
        if schema is None:
            raise ValueError("Schema is required for structured extraction (Pydantic model or dict)")
        
        if instruction is None:
            raise ValueError("Instruction is required for guiding the extraction process")
            
        # If neither query nor URL is provided, we can't proceed
        if query is None and url_passed is None:
            raise ValueError("Either a search query or a specific URL must be provided")

        # Validate provider is specified
        if not provider:
            raise ValueError("LLM provider must be specified (e.g., 'gemini/gemini-1.5-flash')")

        self.max_length = max_length
        self.thread_safe = thread_safe
        self.query = query
        self.url_passed = url_passed
        self.max_results = max_results
        self.search_results = None
        self.instruction = instruction
        self.headless = headless
        self.provider = provider
        self.extract_first_item = extract_first_item

        # Process schema (handle both Pydantic model and dict)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # If it's a Pydantic model class, get its schema
            self.schema = schema.model_json_schema()
            print(f"Using Pydantic model schema: {schema.__name__}")
        else:
            # Assume it's a dictionary
            self.schema = schema
            print("Using dictionary schema")

        self.collected_data = None

        if (query is not None and url_passed is None):
            ##Get URL from google search
            self.search_urls()
        else:
            self.search_results = [{"url": self.url_passed,
                                  "title": "None",
                                  "description": "Passed URL"}]

        # Check for appropriate API key based on provider
        api_key = self._get_api_key_for_provider(provider)
        if not api_key:
            raise ValueError(f"API key not found for provider {provider}")

        # Create LLM Config with explicit provider
        llm_config = LLMConfig(
            provider=self.provider,
            api_token=api_key
        )

        # Define the LLM extraction strategy
        self.llm_strategy = LLMExtractionStrategy(
            llmConfig=llm_config,  # Use the explicitly created config
            schema=self.schema,
            extraction_type="schema",
            instruction=self.instruction,
            chunk_token_threshold=chunk_token_threshold,
            overlap_rate=overlap_rate,
            apply_chunking=apply_chunking,
            input_format=input_format,
            extra_args={"temperature": temperature, "max_tokens": max_tokens}
        )

        # Monkey Patch the LLM strategy to fix the bug in crawl4ai
        if hasattr(self.llm_strategy, 'llm_config') and self.llm_strategy.llm_config is None:
            self.llm_strategy.llm_config = llm_config

        # Set the crawler configuration
        self.crawl_config = CrawlerRunConfig(
            extraction_strategy=self.llm_strategy,
            cache_mode=CacheMode.BYPASS
        )
        
        # Create a browser config if needed
        browser_cfg = BrowserConfig(headless=self.headless)

        # Collect the data from search results
        self.collected_data = self.process_urls()
        
    def _get_api_key_for_provider(self, provider: str) -> str:
        """
        Get the appropriate API key based on the provider
        """
        provider_prefix = provider.split('/')[0].lower()
        
        # Map provider prefixes to environment variable names
        env_var_map = {
            'gemini': 'GOOGLE_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'groq': 'GROQ_API_KEY',
            'ollama': None,  # Ollama typically doesn't need an API key
            'deepseek': 'DEEPSEEK_API_KEY'
        }
        
        # Get the appropriate environment variable name
        env_var_name = env_var_map.get(provider_prefix)
        
        # If provider doesn't need an API key (like ollama)
        if env_var_name is None:
            return "not_required"
            
        # Get the API key from environment variables
        api_key = os.getenv(env_var_name)
        
        if not api_key:
            print(f"Warning: {env_var_name} environment variable is not set for provider {provider}")
            
        return api_key

    def search_urls(self) -> List[Dict[str, str]]:
        """
        Perform Google search and return structured results
        """
        print("Searching for query: ", self.query)
        print("Max results: ", self.max_results)
        gs = GoogleSearch(fixed_max_results=self.max_results)
        self.search_results = gs.search(query=self.query)

    async def _crawl_single_url(self, url: str) -> Dict[str, Any]:
        """
        Crawl a single URL and return its content
        """
        try:
            async with AsyncWebCrawler(thread_safe=self.thread_safe) as crawler:
                result = await crawler.arun(url=url, config=self.crawl_config)
                
                if result.success:
                    # Extract JSON content
                    try:
                        data = json.loads(result.extracted_content)
                        
                        # Handle case where data is a list but we want the first item
                        if self.extract_first_item and isinstance(data, list) and len(data) > 0:
                            data = data[0]
                            
                        print("Extracted items:", data)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        print(f"Raw content: {result.extracted_content[:500]}...")  # Print first 500 chars
                        return {"data": None, "error": f"Failed to parse JSON: {str(e)}"}

                    # Show usage stats
                    self.llm_strategy.show_usage()  # prints token usage
                    
                    # Process content 
                    raw_markdown = result.markdown.raw_markdown
                    fit_markdown = result.markdown.fit_markdown
                    
                    return {"data": data, "error": None}
                else:
                    print("Error:", result.error_message)
                    return {"data": None, "error": result.error_message}
                
        except Exception as e:
            print("Error:", str(e))
            return {"data": None, "error": str(e)}

    async def crawl_urls(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Crawl multiple URLs concurrently and return their content
        
        Args:
            urls: List of dictionaries containing url, title, and tool information
            
        Returns:
            List of extracted information
        """
        crawl_tasks = []
        for url_info in urls:
            try:
                task = self._crawl_single_url(url_info['url'])
                crawl_tasks.append(task)
            except Exception as e:
                print("Error:", str(e))

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
                "data": crawl_result['data']
            }
            processed_results.append(result)
            
        return processed_results

    def process_urls(self) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for crawling multiple URLs
        """
        if self.search_results:  # Check if search_results exists and is not empty
            return asyncio.run(self.crawl_urls(self.search_results))
        else:
            return []
            
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Return the collected data
        """
        return self.collected_data
    
    def get_usage_stats(self) -> None:
        """
        Display usage statistics for the LLM strategy
        """
        if self.llm_strategy:
            self.llm_strategy.show_usage()
        else:
            print("No LLM strategy usage data available")
            
    @staticmethod
    def list_supported_providers() -> List[str]:
        """
        Returns a list of all supported LLM providers
        """
        return [
            "gemini/gemini-1.5-flash",  # Default
            "gemini/gemini-pro",
            "gemini/gemini-1.5-pro",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-2.0-flash-exp",
            "gemini/gemini-2.0-flash-lite-preview-02-05",
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/o1-mini",
            "openai/o1-preview",
            "openai/o3-mini",
            "openai/o3-mini-high",
            "anthropic/claude-3-haiku-20240307",
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-sonnet-20240229",
            "anthropic/claude-3-5-sonnet-20240620",
            "ollama/llama3",
            "groq/llama3-70b-8192",
            "groq/llama3-8b-8192",
            "deepseek/deepseek-chat"
        ]