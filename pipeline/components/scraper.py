"""Crawls a documentation site and extracts clean text per page."""

import logging
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_STRIP_TAGS = {"nav", "footer", "script", "style", "header", "aside"}


def _clean_html(soup: BeautifulSoup) -> str:
    """Removes noise tags and returns stripped text content."""
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _same_origin(base_url: str, candidate: str) -> bool:
    """Returns True if candidate shares the same scheme and host as base_url."""
    base = urlparse(base_url)
    cand = urlparse(candidate)
    return base.scheme == cand.scheme and base.netloc == cand.netloc


def _collect_links(soup: BeautifulSoup, current_url: str, base_url: str) -> list[str]:
    """Extracts internal anchor hrefs that fall under base_url."""
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href: str = anchor["href"]
        absolute = urljoin(current_url, href)
        absolute = absolute.split("#")[0]
        if absolute.startswith(base_url) and _same_origin(base_url, absolute):
            links.append(absolute)
    return links


def scrape_docs(
    base_url: str = "https://www.kubeflow.org/docs/",
    max_pages: int = 50,
) -> list[dict]:
    """Crawls base_url recursively and returns clean text for each page.

    Follows internal links up to max_pages. Strips nav, footer, script, and
    style elements. Enforces a 0.5 s delay between requests.
    """
    visited: set[str] = set()
    queue: list[str] = [base_url]
    results: list[dict] = []

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url
            content = _clean_html(soup)

            if content:
                results.append({"url": url, "title": title, "content": content})
                logger.info("Scraped [%d/%d]: %s", len(results), max_pages, url)

            for link in _collect_links(soup, url, base_url):
                if link not in visited and link not in queue:
                    queue.append(link)

            time.sleep(0.5)

    logger.info("Scraping complete. Total pages collected: %d", len(results))
    return results
