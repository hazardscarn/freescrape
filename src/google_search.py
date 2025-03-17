from typing import TypedDict, List, Optional, Dict, Any
from googlesearch import search as google_search

class SearchResult(TypedDict):
    """Structure for a single search result"""
    title: str
    url: str
    description: str

class GoogleSearch:
    """Direct implementation of Google Search without agent wrapper"""
    def __init__(
        self,
        fixed_max_results: Optional[int] = None,
        fixed_language: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: Optional[int] = 10,
    ):
        self.fixed_max_results = fixed_max_results
        self.fixed_language = fixed_language
        self.proxy = proxy
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5, language: str = "en") -> List[Dict[str, Any]]:
        """
        Perform Google search and return structured results
        """
        max_results = self.fixed_max_results or max_results
        language = self.fixed_language or language
        
        try:
            # Perform search using googlesearch-python
            results = list(google_search(
                query, 
                num_results=max_results, 
                lang=language, 
                proxy=self.proxy,
                advanced=True,
                unique=True
            ))
            
            # Structure the results
            structured_results = []
            for result in results:
                structured_results.append({
                    "title": result.title if hasattr(result, 'title') else "",
                    "url": result.url if hasattr(result, 'url') else "",
                    "description": result.description if hasattr(result, 'description') else ""
                })
            
            return structured_results
            
        except Exception as e:
            print(f"Error in Google search: {str(e)}")
            return []