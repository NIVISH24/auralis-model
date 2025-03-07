import sys
import asyncio
import json
import os
import logging

# On Windows, set the event loop to the Selector event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from twisted.internet import asyncioreactor, error
try:
    asyncioreactor.install()
except error.ReactorAlreadyInstalledError:
    pass

from duckduckgo_search import DDGS
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

# Configure logging for debugging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spider definition to scrape the title from each URL.
class ContentSpider(scrapy.Spider):
    name = "content_spider"
    custom_settings = {
        "FEEDS": {"results.json": {"format": "json", "overwrite": True}},
        "LOG_ENABLED": True,
        # Force the spider to close after 30 seconds (if not finished).
        "CLOSESPIDER_TIMEOUT": 30,
        # Set a download timeout so pages don't hang indefinitely.
        "DOWNLOAD_TIMEOUT": 10,
        "RETRY_TIMES": 0,
        # Disable offsite filtering so all URLs are allowed.
        "OFFSITE_ENABLED": False,
        # Use a common browser user agent.
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
    }

    def __init__(self, start_urls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls

    def start_requests(self):
        for url in self.start_urls:
            self.logger.info("Requesting: %s", url)
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        self.logger.info("Received response from: %s", response.url)
        title = response.css("title::text").get()
        yield {"url": response.url, "title": title if title else "No title found"}

# Asynchronous function to run the Scrapy spider.
async def run_spider_async(links):
    configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
    runner = CrawlerRunner(settings=ContentSpider.custom_settings)
    # Pass the start_urls via the spider constructor.
    deferred = runner.crawl(ContentSpider, start_urls=links)
    loop = asyncio.get_running_loop()
    await deferred.asFuture(loop=loop)

# Function to save results to a text file.
def save_to_txt(results, filename="scraped_results.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"URL: {result.get('url')}\n")
            f.write(f"Title: {result.get('title', '')}\n")
            f.write("-" * 80 + "\n")
    return filename

async def main():
    # Ask the user for a search query.
    query = input("Enter your search query: ")

    # Fetch search results from DuckDuckGo.
    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=3)
    except Exception as e:
        logger.error("Error fetching search results: %s", e)
        return

    # Extract URLs from the search results.
    links = [result["href"] for result in search_results if "href" in result]
    if not links:
        logger.error("No links found for the given query.")
        return

    logger.info("Fetched links: %s", links)

    # Run the spider asynchronously.
    await run_spider_async(links)
    # Wait briefly to ensure the JSON file is written.
    await asyncio.sleep(1)

    if not os.path.exists("results.json"):
        logger.error("results.json file not found.")
        return
    try:
        with open("results.json", "r", encoding="utf-8") as f:
            scraped_data = json.load(f)
    except Exception as e:
        logger.error("Error reading results: %s", e)
        return

    # Save the scraped data to a text file.
    filename = save_to_txt(scraped_data)
    logger.info("Scraping completed and saved in %s", filename)
    logger.info("Scraped data: %s", scraped_data)

if __name__ == "__main__":
    asyncio.run(main())
