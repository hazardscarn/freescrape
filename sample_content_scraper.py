import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig,CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import os 
from crawl4ai.content_filter_strategy import BM25ContentFilter,PruningContentFilter


bm25_filter = BM25ContentFilter(
user_query="machine learning",
bm25_threshold=1.2
    )

prune_filter = PruningContentFilter(
    threshold=0.5,
    threshold_type="fixed",  # or "dynamic"
    min_word_threshold=10
)
async def main():




    md_generator = DefaultMarkdownGenerator(
        content_filter=bm25_filter,
        options={"ignore_links": True}
    )

    # md_generator = DefaultMarkdownGenerator(
    #     options={
    #         "ignore_links": True,
    #         "escape_html": True,
    #         "ignore_images": True,
    #         "body_width": 80
    #     }
    # )

    # config = CrawlerRunConfig(
    #     markdown_generator=md_generator
    # )

    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=prune_filter,
            options={"ignore_links": True},
            
            ),
        excluded_tags=["nav", "footer", "header"],
        exclude_external_links=True,
        check_robots_txt=True, # Ensures compliance with robots.txt rules for ethical and legal web crawling.
        
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://www.studiobinder.com/blog/interstellar-explained-meaning-plot-summary/", config=config)

        if result.success:
            print("Raw Markdown Output:\n")
            print(result.markdown)  # The unfiltered markdown from the page
            with open(os.path.join("raw_markdown.md"), "w", encoding="utf-8") as f:
                f.write(result.markdown.raw_markdown)
            with open(os.path.join("fit_markdown.md"), "w", encoding="utf-8") as f:
                f.write(result.markdown.fit_markdown)
        else:
            print("Crawl failed:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main())