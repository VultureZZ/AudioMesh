"""
Article scraping service for extracting content from URLs.
"""
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

# Optional import for newspaper3k (fallback scraper)
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    Article = None

logger = logging.getLogger(__name__)


class ArticleScraper:
    """Service for scraping article content from URLs."""

    def __init__(self, timeout: int = 30):
        """
        Initialize article scraper.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def validate_url(self, url: str) -> bool:
        """
        Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def scrape_with_beautifulsoup(self, url: str) -> Optional[str]:
        """
        Scrape article using BeautifulSoup.

        Args:
            url: URL to scrape

        Returns:
            Extracted article text or None if failed
        """
        try:
            logger.info(f"Scraping with BeautifulSoup: {url}")
            response = self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Try to find main article content
            # Common article selectors
            article_selectors = [
                "article",
                'div[role="article"]',
                ".article-content",
                ".post-content",
                ".entry-content",
                ".content",
                "main",
            ]

            article_text = None
            for selector in article_selectors:
                elements = soup.select(selector)
                if elements:
                    # Get the largest element (likely the main content)
                    largest = max(elements, key=lambda e: len(e.get_text()))
                    article_text = largest.get_text(separator="\n", strip=True)
                    if len(article_text) > 200:  # Minimum content length
                        logger.info(f"Found content using selector: {selector}")
                        break

            # Fallback: get all paragraph text
            if not article_text or len(article_text) < 200:
                paragraphs = soup.find_all("p")
                article_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                logger.info("Using fallback: extracted all paragraphs")

            if article_text and len(article_text) > 100:
                # Clean up whitespace
                lines = [line.strip() for line in article_text.split("\n") if line.strip()]
                return "\n".join(lines)

            return None

        except Exception as e:
            logger.warning(f"BeautifulSoup scraping failed: {e}")
            return None

    def scrape_with_newspaper(self, url: str) -> Optional[str]:
        """
        Scrape article using newspaper3k library.

        Args:
            url: URL to scrape

        Returns:
            Extracted article text or None if failed
        """
        if not NEWSPAPER_AVAILABLE:
            logger.debug("Newspaper3k not available, skipping")
            return None

        try:
            logger.info(f"Scraping with newspaper3k: {url}")
            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text) > 100:
                logger.info(f"Successfully extracted {len(article.text)} characters")
                return article.text

            return None

        except Exception as e:
            logger.warning(f"Newspaper scraping failed: {e}")
            return None

    def scrape_article(self, url: str) -> str:
        """
        Scrape article content from URL.

        Uses multiple methods with fallback:
        1. BeautifulSoup (primary)
        2. newspaper3k (fallback)

        Args:
            url: URL of the article to scrape

        Returns:
            Extracted article text

        Raises:
            ValueError: If URL is invalid or scraping fails
        """
        if not self.validate_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        logger.info(f"Scraping article from URL: {url}")

        # Try BeautifulSoup first
        content = self.scrape_with_beautifulsoup(url)

        # Fallback to newspaper3k
        if not content or len(content) < 200:
            logger.info("BeautifulSoup result insufficient, trying newspaper3k")
            content = self.scrape_with_newspaper(url)

        if not content or len(content) < 100:
            raise ValueError(
                f"Failed to extract sufficient content from URL: {url}. "
                "The article may be too short or the page structure is not supported."
            )

        logger.info(f"Successfully scraped {len(content)} characters from {url}")
        return content

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


# Global article scraper instance
article_scraper = ArticleScraper()
