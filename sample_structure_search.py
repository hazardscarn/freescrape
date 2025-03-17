import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Type, TypeVar, Generic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.structured_extraction import StructuredExtraction

# Load environment variables
load_dotenv()

# Generic type for pydantic models
T = TypeVar('T', bound=BaseModel)

class Extractor(Generic[T]):
    """
    Generic extractor class that works with any Pydantic model
    """
    def __init__(self, model_class: Type[T]):
        """
        Initialize the extractor with a specific Pydantic model class
        
        Args:
            model_class: The Pydantic model class to use for extraction
        """
        self.model_class = model_class
        self.model_name = model_class.__name__
    
    def extract_from_url(
        self, 
        url: str, 
        instruction: Optional[str] = None,
        provider: str = "gemini/gemini-1.5-flash",
        save_json: bool = False,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract data from a specific URL
        
        Args:
            url: The URL to extract data from
            instruction: Custom instruction for the extraction
                         (if None, a default will be generated)
            provider: LLM provider to use
            save_json: Whether to save results as JSON
            output_file: Custom filename for saved JSON (if None, auto-generated)
            
        Returns:
            List of extraction results
        """
        if instruction is None:
            instruction = f"Extract {self.model_name} information from this webpage. Include all available fields."
        
        print(f"\n=== Extracting {self.model_name} with {provider.split('/')[0].title()} ===")
        
        # Check if we need a specific API key
        if provider.startswith("openai/") and not os.getenv("OPENAI_API_KEY"):
            print(f"Error: OPENAI_API_KEY not set. Cannot use {provider}.")
            return []
        
        if provider.startswith("groq/") and not os.getenv("GROQ_API_KEY"):
            print(f"Error: GROQ_API_KEY not set. Cannot use {provider}.")
            return []
            
        if provider.startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
            print(f"Error: ANTHROPIC_API_KEY not set. Cannot use {provider}.")
            return []
            
        if provider.startswith("gemini/") and not os.getenv("GOOGLE_API_KEY"):
            print(f"Error: GOOGLE_API_KEY not set. Cannot use {provider}.")
            return []
        
        extractor = StructuredExtraction(
            url_passed=url,
            schema=self.model_class,
            instruction=instruction,
            provider=provider
        )
        
        results = extractor.get_results()
        self._print_results(results)
        
        # Save results as JSON if requested
        if save_json:
            self.save_results_as_json(results, output_file)
            
        return results
    
    def search_and_extract(
        self, 
        query: str, 
        max_results: int = 2, 
        instruction: Optional[str] = None,
        provider: str = "gemini/gemini-1.5-flash",
        save_json: bool = False,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for content and extract data
        
        Args:
            query: Search query
            max_results: Maximum number of search results to process
            instruction: Custom instruction for the extraction
                         (if None, a default will be generated)
            provider: LLM provider to use
            save_json: Whether to save results as JSON
            output_file: Custom filename for saved JSON (if None, auto-generated)
            
        Returns:
            List of extraction results
        """
        if instruction is None:
            instruction = f"Extract {self.model_name} information from this webpage. Include all available fields."
        
        print(f"\n=== Searching and Extracting {self.model_name} with {provider.split('/')[0].title()}: {query} ===")
        
        # Check if we have GOOGLE_API_KEY for search
        if not os.getenv("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY not set. Cannot perform search.")
            return []
        
        extractor = StructuredExtraction(
            query=query,
            max_results=max_results,
            schema=self.model_class,
            instruction=instruction,
            provider=provider
        )
        
        results = extractor.get_results()
        self._print_results(results)
        
        # Save results as JSON if requested
        if save_json:
            self.save_results_as_json(results, output_file)
            
        return results
    
    def save_results_as_json(
        self, 
        results: List[Dict[str, Any]], 
        output_file: Optional[str] = None
    ) -> str:
        """
        Save extraction results as a JSON file
        
        Args:
            results: Extraction results to save
            output_file: Custom filename (if None, auto-generated)
            
        Returns:
            Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        output_dir = "extracted_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate default filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.model_name.lower()}_{timestamp}.json"
        
        # Ensure the filename has .json extension
        if not output_file.lower().endswith('.json'):
            output_file += '.json'
            
        # Combine path
        output_path = os.path.join(output_dir, output_file)
        
        # Format the data for better readability
        # Extract just the data for a cleaner JSON
        formatted_data = []
        for result in results:
            item = {
                "url": result["url"],
                "title": result["title"],
            }
            
            # Include error message if there is one
            if result["error"]:
                item["error"] = result["error"]
            else:
                # Include the extracted data
                item["data"] = result["data"]
                
            formatted_data.append(item)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nResults saved to: {output_path}")
        return output_path
        
    def _print_results(self, results: List[Dict[str, Any]]) -> None:
        """Print extraction results that handles both dict and list data"""
        for i, result in enumerate(results):
            print(f"\nResult from {result['url']}:")
            if result['error']:
                print(f"Error: {result['error']}")
                continue
                
            # Get the data (might be dict or list)
            data = result['data']
            
            if data is None:
                print("No data extracted")
                continue
            
            # Get model fields for printing
            model_fields = self.model_class.model_fields
            
            # Print each field from the model if it exists in the data
            for field_name, field_info in model_fields.items():
                value = self._safe_get(data, field_name)
                
                # Skip if the value is the default N/A
                if value == 'N/A':
                    continue
                    
                # Handle special field types
                if field_name == 'description' and isinstance(value, str) and len(value) > 100:
                    print(f"- Description: {value[:100]}...")
                    continue
                
                # Handle list fields (like features)
                if isinstance(value, list):
                    print(f"- {field_name.title()} ({len(value)}):")
                    # Show first 3 items
                    for idx, item in enumerate(value[:3]):
                        if isinstance(item, dict) and 'name' in item:
                            print(f"  {idx+1}. {item['name']}")
                        else:
                            print(f"  {idx+1}. {item}")
                    continue
                    
                # For regular fields
                print(f"- {field_name.title()}: {value}")
    
    @staticmethod
    def _safe_get(data: Union[Dict[str, Any], List, None], key: str, default: Any = 'N/A') -> Any:
        """
        Safely get a value from data which could be a dict, list, or None
        
        Args:
            data: The data object (dict, list, or None)
            key: The key to retrieve (for dict) or index (for list, if key is numeric)
            default: Default value to return if key not found
            
        Returns:
            The value or default
        """
        if data is None:
            return default
            
        # If data is a dictionary, use get method
        if isinstance(data, dict):
            return data.get(key, default)
            
        # If data is a list and key is first item in the list and it's a dict
        if isinstance(data, list) and len(data) > 0:
            # If first item is a dict, try to get the key from it
            if isinstance(data[0], dict):
                return data[0].get(key, default)
            # If the entire list is what we want, return it for certain keys
            if key in ['features', 'items', 'options', 'benefits']:
                return data
                
        # Default case
        return default
                    
def list_providers():
    """List all supported providers"""
    print("\n=== Supported LLM Providers ===")
    providers = StructuredExtraction.list_supported_providers()
    
    # Group providers by platform
    grouped = {}
    for provider in providers:
        platform, model = provider.split('/')
        if platform not in grouped:
            grouped[platform] = []
        grouped[platform].append(model)
    
    # Print grouped providers
    for platform, models in grouped.items():
        print(f"\n{platform.title()}:")
        for model in models:
            print(f"  - {model}")

# Example Pydantic models for different extraction purposes
class Address(BaseModel):
    """Model for product information"""
    name: str = Field(description="The Name of the Business")
    address: str = Field(description="The address of the business")
    hours: Optional[str] = Field(None, description="The working or open hours of the business")
    description: Optional[str] = Field(None, description="Brief description about business")
    function: Optional[List[str]] = Field(None, description="What is the business function or purpose? What does it do?")
    industry: Optional[float] = Field(None, description="What type of industry the business belongs to")
    size: Optional[str] = Field(None, description="Approximate size of the business in square feet")
    

class ElectricPlan(BaseModel):
    """Model for electric plan information"""
    name: str = Field(description="The name of the electric plan")
    provider: Optional[str] = Field(None, description="The company/provider offering the plan")
    rate: Optional[str] = Field(None, description="The electricity rate (e.g., cents per kWh)")
    term_length: Optional[str] = Field(None, description="Length of the contract term")
    benefits: Optional[List[str]] = Field(None, description="Key benefits or features of the plan")
    renewable: Optional[bool] = Field(None, description="Whether the plan includes renewable energy")
    early_termination_fee: Optional[str] = Field(None, description="Fee for early termination of contract")
    base_charge: Optional[str] = Field(None, description="Base or monthly charge")


class StockTracker(BaseModel):
    """Model identifying the big volatile stocks due to US Tariffs"""
    name: str = Field(description="The name of the stock")
    ticker: Optional[str] = Field(None, description="The ticker of the stock")
    sector: Optional[str] = Field(None, description="The sector of the stock")
    tariff_impact: Optional[str] = Field(None, description="The impact of US tariffs on the stock")
    reasons: Optional[List[str]] = Field(None, description="What are the reasons for the stock being affected")


class Article(BaseModel):
    """Model for news article information"""
    title: str = Field(description="The article headline or title")
    author: Optional[str] = Field(None, description="Article author name")
    publication_date: Optional[str] = Field(None, description="When the article was published")
    summary: Optional[str] = Field(None, description="A concise summary of the article content")
    key_points: Optional[List[str]] = Field(None, description="Main points or takeaways from the article")
    topics: Optional[List[str]] = Field(None, description="Topics or categories the article belongs to")

# Example usage
if __name__ == "__main__":
    # List all supported providers
    list_providers()
    
    # Create extractors for different models
    address_extractor = Extractor(Address)
    plan_extractor = Extractor(ElectricPlan)
    article_extractor = Extractor(Article)
    stock_tracker = Extractor(StockTracker)
    


    stock_tracker.search_and_extract(
    "Identify top 5 big US stocks going to be affected by US Tariffs from wallstreetbets subreddit", 
    max_results=5,
    instruction="Extract all the required details about the top 5 stocks that are going to be affected by US Tariffs",
    save_json=True,  # This will save the results as JSON with an auto-generated filename
    output_file="stock_tracker.json")


    # # Extract address info and save as JSON
    # address_extractor.search_and_extract(
    #     "6977 Vanguard Dr, Mississauga, ON L5S 1B2", 
    #     max_results=4,
    #     instruction="Extract all the required details about this address",
    #     save_json=True,  # This will save the results as JSON with an auto-generated filename
    #     output_file="sample_address_info.json"

    # )
    
    # Example with custom filename
    # plan_extractor.search_and_extract(
    #     "Electric Plans for Home in Houston zip code 77024",
    #     max_results=3,
    #     save_json=True,
    #     output_file="houston_electric_plans.json"
    # )
    
    # You can also save results from a specific URL
    # article_extractor.extract_from_url(
    #     "https://www.nytimes.com/section/technology",
    #     provider="openai/gpt-4o-mini",
    #     save_json=True,
    #     output_file="tech_news.json"
    # )