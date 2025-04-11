#!/usr/bin/env python3
"""
Scientific Search MCP Tool - Pure Python implementation with stdio transport
"""
import json
import os
import sys
import logging
import re
import tempfile
import shutil
import urllib.parse
from typing import Dict, List, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import time
import signal
import asyncio
import io
import uuid
from urllib.parse import urlparse, urljoin
import PyPDF2
from duckduckgo_search import AsyncDDGS
import aiohttp
import select
import random
import datetime
import gzip

# Determine the script directory and set it as working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Set up logging to both stderr and file
LOG_DIR = os.path.join(SCRIPT_DIR, ".logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "scientific_search_mcp.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger("scientific_search_mcp")
logger.info(f"Starting scientific_search_mcp - Logging to {LOG_FILE}")
logger.info(f"Changed working directory to: {SCRIPT_DIR}")

# Ensure we have a directory for storing search results
SEARCH_DIR = os.path.join(SCRIPT_DIR, ".search")
IMAGES_DIR = os.path.join(SEARCH_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Helper functions ---
def ensure_images_dir():
    """Create .search/images directory if it doesn't exist"""
    img_dir = os.path.join(os.getcwd(), '.search', 'images')
    os.makedirs(img_dir, exist_ok=True)
    return img_dir

async def download_image(session, img_url, base_url, img_dir):
    """Download an image and save it to the images directory"""
    try:
        # Handle relative URLs
        if not bool(urlparse(img_url).netloc):
            img_url = urljoin(base_url, img_url)
            
        # Generate a filename based on the URL
        img_filename = f"{uuid.uuid4().hex}.{img_url.split('.')[-1] if '.' in img_url.split('/')[-1] else 'jpg'}"
        img_path = os.path.join(img_dir, img_filename)
        
        # Download the image
        async with session.get(img_url, timeout=10) as response:
            if response.status == 200:
                with open(img_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                
                rel_path = os.path.join('.search', 'images', img_filename)
                return rel_path, img_url
    except Exception as e:
        logger.error(f"Error downloading image {img_url}: {e}")
    
    return None, img_url

async def fallback_search(query: str):
    """
    Fallback search method using direct HTTP requests to DuckDuckGo.
    Used when AsyncDDGS fails or times out.
    """
    logger.info(f"Using fallback search for: {query}")
    results = []
    
    try:
        # Encode the query for URL
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results
                    search_results = soup.select('.result')
                    for result in search_results:
                        title_elem = result.select_one('.result__a')
                        snippet_elem = result.select_one('.result__snippet')
                        url_elem = title_elem.get('href') if title_elem else None
                        
                        if title_elem and url_elem:
                            # Parse the URL from DuckDuckGo redirect URL
                            parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(url_elem).query).get('uddg', [''])[0]
                            if not parsed_url:
                                parsed_url = url_elem
                                
                            results.append({
                                "title": title_elem.text.strip(),
                                "url": parsed_url,
                                "snippet": snippet_elem.text.strip() if snippet_elem else ""
                            })
                    
                    logger.info(f"Fallback search found {len(results)} results")
                else:
                    logger.error(f"Fallback search failed with status code: {response.status}")
    except Exception as e:
        logger.error(f"Error in fallback search: {e}")
    
    return results

# --- Tool Implementations ---

async def search_scientific(query: str, engine: str = "duckduckgo", sources: list[str] | None = None):
    """
    Searches scientific sources using a chosen engine.
    Currently supports: duckduckgo (general web), google_scholar (requires custom implementation/library), 
    pubmed (requires library like pymed), arxiv (requires library), semanticscholar (requires library).
    Sources parameter is currently ignored but could filter results post-retrieval.
    """
    logger.info(f"Executing search_scientific: query='{query}', engine='{engine}', sources={sources}")
    status = "error"
    message = ""
    results = []
    engine_used = engine

    try:
        if engine == "duckduckgo":
            try:
                async with AsyncDDGS() as ddgs:
                    # Use text search, limit results
                    try:
                        # Add a 15-second timeout for the search
                        raw_results = await asyncio.wait_for(
                            ddgs.text(query, max_results=10),
                            timeout=15.0
                        )
                        # Reformat results to a consistent structure
                        results = [
                            {"title": r.get('title', 'No Title'), "url": r.get('href', ''), "snippet": r.get('body', '')}
                            for r in raw_results
                        ]
                        status = "success"
                        message = f"Found {len(results)} results using DuckDuckGo."
                        logger.info(message)
                    except asyncio.TimeoutError:
                        logger.warning("DuckDuckGo search timed out, trying fallback method")
                        results = await fallback_search(query)
                        if results:
                            status = "success"
                            message = f"Found {len(results)} results using fallback DuckDuckGo search."
                            logger.info(message)
                        else:
                            status = "error"
                            message = "Both primary and fallback DuckDuckGo searches failed."
                            logger.error(message)
            except Exception as e:
                logger.warning(f"AsyncDDGS failed with error: {e}, trying fallback method")
                results = await fallback_search(query)
                if results:
                    status = "success"
                    message = f"Found {len(results)} results using fallback DuckDuckGo search."
                    logger.info(message)
                else:
                    status = "error"
                    message = f"DuckDuckGo search failed: {e}"
                    logger.error(message)

        elif engine == "arxiv":
            try:
                # Import arxiv library here to make it optional
                import arxiv
                
                # Search arXiv for papers
                search = arxiv.Search(
                    query=query,
                    max_results=10,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                client = arxiv.Client()
                
                # Get the results
                raw_results = list(client.results(search))
                
                if raw_results:
                    results = []
                    for r in raw_results:
                        # Format results to a consistent structure
                        results.append({
                            "title": r.title,
                            "url": r.entry_id,
                            "authors": [author.name for author in r.authors],
                            "published": r.published.strftime("%Y-%m-%d") if r.published else None,
                            "summary": r.summary,
                            "pdf_url": r.pdf_url,
                            "categories": r.categories
                        })
                    status = "success"
                    message = f"Found {len(results)} papers on arXiv for query: {query}"
                    logger.info(message)
                else:
                    status = "success"
                    message = f"No papers found on arXiv for query: {query}"
                    logger.info(message)
            except ImportError:
                status = "error"
                message = "arXiv search requires the 'arxiv' package. Please install it with 'pip install arxiv'."
                logger.error(message)
            except Exception as e:
                status = "error"
                message = f"arXiv search failed: {e}"
                logger.error(message)
                
        elif engine == "google_scholar":
            try:
                import scholarly
                
                # Direct search without using proxy
                search_query = scholarly.scholarly.search_pubs(query, patents=False)
                papers = []
                
                # Get the first 10 papers
                count = 0
                for paper in search_query:
                    if count >= 10:
                        break
                        
                    # Process paper data
                    paper_data = {
                        "title": paper.get("bib", {}).get("title", "Unknown Title"),
                        "abstract": paper.get("bib", {}).get("abstract", None),
                        "authors": paper.get("bib", {}).get("author", []),
                        "year": paper.get("bib", {}).get("pub_year", None),
                        "venue": paper.get("bib", {}).get("venue", None),
                        "citation_count": paper.get("num_citations", 0),
                        "url": paper.get("pub_url", None)
                    }
                    papers.append(paper_data)
                    count += 1
                
                if not papers:
                    return {"status": "error", "message": f"No papers found on Google Scholar for query: {query}", "results": [], "engine_used": "google_scholar"}
                
                return {"status": "success", "message": f"Found {len(papers)} papers on Google Scholar for query: {query}", "results": papers, "engine_used": "google_scholar"}
            except Exception as e:
                logging.error(f"Google Scholar search failed: {str(e)}")
                return {"status": "error", "message": f"Google Scholar search failed: {str(e)}", "results": [], "engine_used": "google_scholar"}

        elif engine == "pubmed":
            try:
                # Import required libraries
                from pymed import PubMed
                
                # Helper function to make dates JSON serializable
                def json_serializable(obj):
                    if isinstance(obj, (datetime.date, datetime.datetime)):
                        return obj.isoformat()
                    return obj
                
                # Initialize PubMed client
                logging.info(f"Initializing PubMed client for query: {query}")
                email = os.environ.get("PUBMED_EMAIL", "user@example.com")
                pubmed = PubMed(tool="ScientificSearchMCP", email=email)
                
                # Query PubMed
                results_raw = pubmed.query(query, max_results=10)
                results = []
                
                # Process results
                for article in results_raw:
                    # Convert the article to a dictionary
                    article_dict = article.toDict()
                    
                    # Extract relevant information
                    pubmed_id = article_dict.get('pubmed_id', '').partition('\n')[0]
                    
                    # Process publication date if it exists
                    publication_date = article_dict.get('publication_date')
                    if publication_date and isinstance(publication_date, (datetime.date, datetime.datetime)):
                        publication_date = publication_date.isoformat()
                    
                    paper_data = {
                        "pubmed_id": pubmed_id,
                        "title": article_dict.get('title', 'Unknown Title'),
                        "abstract": article_dict.get('abstract', None),
                        "keywords": article_dict.get('keywords', []),
                        "journal": article_dict.get('journal', None),
                        "publication_date": publication_date,
                        "authors": article_dict.get('authors', []),
                        "doi": article_dict.get('doi', None),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/" if pubmed_id else ''
                    }
                    results.append(paper_data)
                
                logging.info(f"Found {len(results)} papers on PubMed for query: {query}")
                return {"status": "success", "message": f"Found {len(results)} papers on PubMed for query: {query}", "results": results, "engine_used": "pubmed"}
            except ImportError:
                logging.error("PubMed search requires the 'pymed' package")
                return {"status": "error", "message": "PubMed search requires the 'pymed' package. Please install it with 'pip install pymed'.", "results": [], "engine_used": "pubmed"}
            except Exception as e:
                logging.error(f"PubMed search failed: {str(e)}")
                return {"status": "error", "message": f"PubMed search failed: {str(e)}", "results": [], "engine_used": "pubmed"}
        
        elif engine == "semantic_scholar":
            try:
                # Try a direct API approach using Tor proxy
                import requests
                from urllib.parse import quote
                
                # Base URL for Semantic Scholar API
                base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                
                # Set up parameters
                params = {
                    "query": query,
                    "limit": 10,
                    "fields": "paperId,title,abstract,year,authors,venue,url,citationCount,influentialCitationCount"
                }
                
                # Set up comprehensive headers to mimic a real browser
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Referer": "https://www.semanticscholar.org/",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "same-origin",
                    "Sec-Fetch-User": "?1",
                    "Sec-Ch-Ua": "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\"",
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": "\"Windows\"",
                    "Dnt": "1",
                    "Pragma": "no-cache",
                    "Cache-Control": "no-cache",
                    "Upgrade-Insecure-Requests": "1"
                }
                
                # Set up Tor proxy
                proxies = {
                    'http': 'socks5h://127.0.0.1:9050',
                    'https': 'socks5h://127.0.0.1:9050'
                }
                
                logger.info(f"Using Tor proxy for Semantic Scholar search: {proxies}")
                
                # Implement retry with exponential backoff
                max_retries = 3
                retry_delay = 1  # Start with 1 second delay
                
                for attempt in range(max_retries):
                    try:
                        # Make the request with headers and parameters through Tor
                        logger.info(f"Attempting Semantic Scholar API request via Tor (attempt {attempt+1}/{max_retries})")
                        response = requests.get(
                            base_url, 
                            params=params, 
                            headers=headers, 
                            proxies=proxies,
                            timeout=30  # Longer timeout for Tor
                        )
                        
                        # Check if successful
                        if response.status_code == 200:
                            data = response.json()
                            papers = data.get("data", [])
                            
                            # Process the results
                            results = []
                            for paper in papers:
                                authors = []
                                if paper.get("authors"):
                                    authors = [author.get("name") for author in paper.get("authors", []) if author.get("name")]
                                
                                results.append({
                                    "paper_id": paper.get("paperId"),
                                    "title": paper.get("title", "No Title"),
                                    "abstract": paper.get("abstract", "No abstract available"),
                                    "year": paper.get("year"),
                                    "authors": authors,
                                    "venue": paper.get("venue", "Unknown"),
                                    "url": paper.get("url", ""),
                                    "citation_count": paper.get("citationCount", 0),
                                    "influence_citation_count": paper.get("influentialCitationCount", 0),
                                })
                            
                            status = "success"
                            message = f"Found {len(results)} papers on Semantic Scholar for query: {query}"
                            logger.info(message)
                            break  # Success, exit retry loop
                            
                        elif response.status_code == 403:
                            # Try an alternative approach - use the public API without search
                            # This is less accurate but more likely to work
                            logger.warning("403 Forbidden from Semantic Scholar API, trying fallback approach with Tor")
                            
                            # Use author search as a fallback
                            alt_url = "https://api.semanticscholar.org/graph/v1/author/search"
                            alt_params = {
                                "query": query,
                                "limit": 5,
                                "fields": "papers.title,papers.year,papers.abstract,papers.authors,papers.venue,papers.url,papers.citationCount"
                            }
                            
                            # Add a slight delay between requests
                            time.sleep(1 + random.random())
                            
                            alt_response = requests.get(
                                alt_url, 
                                params=alt_params, 
                                headers=headers, 
                                proxies=proxies,
                                timeout=30
                            )
                            
                            if alt_response.status_code == 200:
                                alt_data = alt_response.json()
                                authors_data = alt_data.get("data", [])
                                
                                # Extract papers from authors
                                results = []
                                for author in authors_data:
                                    papers = author.get("papers", [])
                                    for paper in papers:
                                        paper_authors = []
                                        if paper.get("authors"):
                                            paper_authors = [a.get("name") for a in paper.get("authors", []) if a.get("name")]
                                        
                                        results.append({
                                            "paper_id": paper.get("paperId"),
                                            "title": paper.get("title", "No Title"),
                                            "abstract": paper.get("abstract", "No abstract available"),
                                            "year": paper.get("year"),
                                            "authors": paper_authors,
                                            "venue": paper.get("venue", "Unknown"),
                                            "url": paper.get("url", ""),
                                            "citation_count": paper.get("citationCount", 0)
                                        })
                                
                                # Remove duplicates based on title
                                unique_results = []
                                seen_titles = set()
                                for paper in results:
                                    if paper["title"] not in seen_titles:
                                        seen_titles.add(paper["title"])
                                        unique_results.append(paper)
                                
                                # Limit to 10 most cited papers
                                unique_results.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
                                unique_results = unique_results[:10]
                                
                                results = unique_results
                                status = "success"
                                message = f"Found {len(results)} papers on Semantic Scholar using fallback approach for query: {query}"
                                logger.info(message)
                                break  # Success with fallback, exit retry loop
                            
                            # Try a third approach - direct citation lookup
                            if attempt < max_retries - 1 and not results:
                                logger.warning("Fallback search failed, trying citation ID lookup through Tor")
                                
                                # Try to find papers using a direct citation lookup for a known paper
                                citation_url = "https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b/citations"
                                citation_params = {
                                    "fields": "title,abstract,year,authors,venue,url,citationCount",
                                    "limit": 10
                                }
                                
                                citation_response = requests.get(
                                    citation_url, 
                                    params=citation_params, 
                                    headers=headers, 
                                    proxies=proxies,
                                    timeout=30
                                )
                                
                                if citation_response.status_code == 200:
                                    citation_data = citation_response.json()
                                    citation_papers = citation_data.get("data", [])
                                    
                                    # Filter papers that match query terms
                                    query_terms = query.lower().split()
                                    results = []
                                    
                                    for citation in citation_papers:
                                        paper = citation.get("citingPaper", {})
                                        title = paper.get("title", "").lower()
                                        abstract = paper.get("abstract", "").lower()
                                        
                                        # Check if the paper matches query terms
                                        if any(term in title or term in abstract for term in query_terms):
                                            paper_authors = []
                                            if paper.get("authors"):
                                                paper_authors = [a.get("name") for a in paper.get("authors", []) if a.get("name")]
                                            
                                            results.append({
                                                "paper_id": paper.get("paperId"),
                                                "title": paper.get("title", "No Title"),
                                                "abstract": paper.get("abstract", "No abstract available"),
                                                "year": paper.get("year"),
                                                "authors": paper_authors,
                                                "venue": paper.get("venue", "Unknown"),
                                                "url": paper.get("url", ""),
                                                "citation_count": paper.get("citationCount", 0)
                                            })
                                    
                                    if results:
                                        status = "success"
                                        message = f"Found {len(results)} papers on Semantic Scholar using citation lookup for query: {query}"
                                        logger.info(message)
                                        break  # Success with citation lookup, exit retry loop
                            
                            # If all attempts failed, continue with retry
                            if attempt < max_retries - 1:
                                # Exponential backoff with jitter
                                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                                logger.warning(f"Semantic Scholar request failed. Retrying in {wait_time:.2f} seconds via Tor...")
                                time.sleep(wait_time)
                            else:
                                status = "error"
                                message = f"All Semantic Scholar API approaches failed for query: {query}"
                                logger.error(message)
                        
                        else:
                            if attempt < max_retries - 1:
                                # Exponential backoff with jitter
                                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                                logger.warning(f"Semantic Scholar request failed with status {response.status_code}. Retrying in {wait_time:.2f} seconds via Tor...")
                                time.sleep(wait_time)
                            else:
                                status = "error"
                                message = f"Semantic Scholar search failed with status code: {response.status_code}"
                                logger.error(message)
                    
                    except requests.RequestException as e:
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Semantic Scholar request error via Tor: {e}. Retrying in {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
                        else:
                            status = "error"
                            message = f"Semantic Scholar request error after {max_retries} attempts: {e}"
                            logger.error(message)
                
            except ImportError:
                status = "error"
                message = "Semantic Scholar search requires the 'requests' and 'pysocks' packages. Please install them with 'pip install requests[socks]'."
                logger.error(message)
            except Exception as e:
                status = "error"
                message = f"Semantic Scholar search failed: {e}"
                logger.error(message)
            
        # Add other engines like yandex, crossref similarly
        else:
            status = "error"
            message = f"Unsupported search engine: '{engine}'. Supported engines: duckduckgo, google_scholar, pubmed, arxiv, and semantic_scholar."
            logger.error(message)
            engine_used = "none"

    except Exception as e:
        status = "error"
        message = f"An error occurred during search with {engine}: {e}"
        logger.exception(message)

    return {"status": status, "message": message, "results": results, "engine_used": engine_used}

async def get_url_content(url: str, download_images: bool = True):
    """
    Fetches and extracts text content from a given URL.
    If download_images is True, downloads images to .search/images folder.
    """
    logger.info(f"Executing get_url_content: url='{url}', download_images={download_images}")
    text_content = ""
    image_references = []
    status = "error"
    message = ""

    try:
        # Create images directory if it doesn't exist
        if download_images:
            img_dir = ensure_images_dir()
        
        async with aiohttp.ClientSession() as session:
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with session.get(url, headers=headers, timeout=30, allow_redirects=True) as response:
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                # Basic check for HTML content type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'html' not in content_type:
                     logger.warning(f"Content-Type for {url} is '{content_type}', not HTML. Parsing might fail.")
                     # Allow processing non-html for cases where server sends wrong type

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Remove script and style elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()

                # Get text content
                main_content = soup.find('main') or soup.find('article') or soup.body
                if main_content:
                    text_content = main_content.get_text(separator='\n', strip=True)
                else:
                     text_content = soup.get_text(separator='\n', strip=True)

                # Download images if requested
                if download_images:
                    # Find all images
                    img_tasks = []
                    for img in soup.find_all('img'):
                        img_url = img.get('src')
                        if img_url:
                            img_tasks.append(download_image(session, img_url, url, img_dir))
                    
                    # Download all images concurrently
                    if img_tasks:
                        logger.info(f"Downloading {len(img_tasks)} images from {url}")
                        img_results = await asyncio.gather(*img_tasks)
                        
                        # Add image references to text content
                        for rel_path, original_url in img_results:
                            if rel_path:
                                image_references.append({
                                    "path": rel_path,
                                    "original_url": original_url
                                })
                                logger.info(f"Downloaded image to {rel_path}")
                            else:
                                logger.warning(f"Failed to download image from {original_url}")

                status = "success"
                message = f"Successfully fetched and parsed content from {url}"
                logger.info(message)

    except aiohttp.ClientError as e:
        message = f"Network error fetching URL {url}: {e}"
        logger.error(message)
    except asyncio.TimeoutError:
        message = f"Timeout error fetching URL {url}"
        logger.error(message)
    except Exception as e:
        message = f"Error processing URL {url}: {e}"
        logger.exception(message) # Log full traceback for unexpected errors

    # Add image references to the text content
    if status == "success" and image_references:
        image_references_text = "\n\nImages found on the page:\n"
        for idx, img in enumerate(image_references):
            image_references_text += f"[image_path: {img['path']}] (Source: {img['original_url']})\n"
        text_content += image_references_text

    return {
        "status": status, 
        "message": message, 
        "url": url, 
        "text_content": text_content, 
        "image_references": image_references
    }

async def grep_url_content(url: str, regex_pattern: str):
    """
    Fetches content from a URL and searches it using a regular expression.
    """
    logger.info(f"Executing grep_url_content: url='{url}', regex='{regex_pattern}'")

    # 1. Fetch content
    content_result = await get_url_content(url, download_images=False)

    if content_result["status"] != "success":
        logger.error(f"Failed to get content for grep: {content_result['message']}")
        return {"status": "error", "message": f"Could not fetch content from {url} to grep: {content_result['message']}", "url": url, "regex_pattern": regex_pattern, "matches": []}

    # 2. Perform regex search
    matches = []
    status = "error"
    message = ""
    try:
        # Use re.findall to get all non-overlapping matches
        matches = re.findall(regex_pattern, content_result["text_content"])
        status = "success"
        message = f"Found {len(matches)} matches for regex '{regex_pattern}' in {url}"
        logger.info(message)
    except re.error as e:
        status = "error"
        message = f"Invalid regex pattern: {e}"
        logger.error(f"Regex error for pattern '{regex_pattern}': {e}")
    except Exception as e:
        status = "error"
        message = f"An unexpected error occurred during regex search: {e}"
        logger.exception(f"Unexpected error during grep for '{regex_pattern}' in {url}")

    return {"status": status, "message": message, "url": url, "regex_pattern": regex_pattern, "matches": matches}

async def process_pdf(url_or_path: str, extract_images: bool = True):
    """
    Extracts text and images from a PDF file (from URL or local path).
    If extract_images is True, extracts images and saves them to .search/images.
    """
    logger.info(f"Executing process_pdf: source='{url_or_path}', extract_images={extract_images}")
    text_content = ""
    image_references = []
    status = "error"
    message = ""
    pdf_stream = None
    pdf_path = None

    try:
        # Create images directory if extracting images
        if extract_images:
            img_dir = ensure_images_dir()

        # Check if it's a URL
        parsed_uri = urlparse(url_or_path)
        is_url = all([parsed_uri.scheme, parsed_uri.netloc])

        if is_url:
            logger.info(f"Source is a URL: {url_or_path}. Attempting download...")
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0'}
                async with session.get(url_or_path, headers=headers, timeout=60, allow_redirects=True) as response:
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'pdf' not in content_type:
                        raise ValueError(f"Expected PDF content type, but got '{content_type}'")
                    pdf_bytes = await response.read()
                    
                    # Save PDF to temp file for PyMuPDF which works better with files
                    temp_pdf_path = os.path.join(os.getcwd(), '.search', f"temp_{uuid.uuid4().hex}.pdf")
                    os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
                    
                    with open(temp_pdf_path, 'wb') as f:
                        f.write(pdf_bytes)
                    
                    pdf_path = temp_pdf_path
                    logger.info(f"Successfully downloaded PDF from {url_or_path} to {pdf_path}")
        else:
            logger.info(f"Source is a local path: {url_or_path}")
            # Treat as local file path
            if not os.path.exists(url_or_path):
                raise FileNotFoundError(f"Local PDF file not found: {url_or_path}")
            if not os.path.isfile(url_or_path):
                 raise ValueError(f"Path is not a file: {url_or_path}")
            pdf_path = url_or_path

        # Process the PDF using PyMuPDF (fitz) for better image extraction
        if pdf_path:
            # 1. Extract text using PyPDF2 for compatibility
            pdf_stream = open(pdf_path, 'rb')
            reader = PyPDF2.PdfReader(pdf_stream)
            num_pages = len(reader.pages)
            logger.info(f"Processing {num_pages} pages in PDF...")
            extracted_texts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text: # Avoid adding None if extraction fails
                       extracted_texts.append(page_text)
                except Exception as page_err:
                    logger.warning(f"Could not extract text from page {i+1}: {page_err}")

            text_content = "\n\n---\n\n".join(extracted_texts) # Join pages with separator
            
            # 2. Extract images using PyMuPDF if requested
            if extract_images:
                try:
                    doc = fitz.open(pdf_path)
                    img_count = 0
                    
                    for page_num, page in enumerate(doc):
                        image_list = page.get_images(full=True)
                        for img_idx, img_info in enumerate(image_list):
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            
                            if base_image:
                                image_ext = base_image["ext"]
                                image_bytes = base_image["image"]
                                
                                # Save the image
                                img_filename = f"pdf_page{page_num+1}_img{img_idx+1}.{image_ext}"
                                img_path = os.path.join(img_dir, img_filename)
                                
                                with open(img_path, 'wb') as img_file:
                                    img_file.write(image_bytes)
                                
                                rel_path = os.path.join('.search', 'images', img_filename)
                                image_references.append({
                                    "path": rel_path,
                                    "page": page_num + 1,
                                    "index": img_idx + 1
                                })
                                
                                img_count += 1
                    
                    if img_count > 0:
                        logger.info(f"Successfully extracted {img_count} images from PDF")
                    else:
                        logger.warning("No images found in the PDF")
                
                except Exception as e:
                    logger.error(f"Error extracting images from PDF: {e}")
                    # Continue with text extraction even if image extraction fails

            status = "success"
            message = f"Successfully extracted text from {num_pages} pages."
            logger.info(message)

    except FileNotFoundError as e:
        message = str(e)
        logger.error(message)
    except aiohttp.ClientError as e:
        message = f"Network error downloading PDF {url_or_path}: {e}"
        logger.error(message)
    except PyPDF2.errors.PdfReadError as e:
        message = f"Error reading PDF file {url_or_path}: {e}. It might be corrupted or encrypted."
        logger.error(message)
    except ValueError as e: # For content type or path errors
        message = str(e)
        logger.error(message)
    except Exception as e:
        message = f"An unexpected error occurred processing PDF {url_or_path}: {e}"
        logger.exception(message)
    finally:
        # Clean up
        if pdf_stream and hasattr(pdf_stream, 'close'):
            pdf_stream.close()
        
        # Remove temporary file if we created one
        if is_url and pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logger.info(f"Removed temporary PDF file: {pdf_path}")
            except:
                logger.warning(f"Failed to remove temporary PDF file: {pdf_path}")

    # Add image references to the text content
    if status == "success" and image_references:
        image_references_text = "\n\nImages extracted from the PDF:\n"
        for idx, img in enumerate(image_references):
            image_references_text += f"[image_path: {img['path']}] (Page: {img['page']}, Index: {img['index']})\n"
        text_content += image_references_text

    return {
        "status": status, 
        "message": message, 
        "source": url_or_path, 
        "text_content": text_content, 
        "image_references": image_references
    }

# --- JSONRPC Implementation ---

# Create synchronous wrappers for async methods
def sync_search_scientific(query: str, engine: str = "duckduckgo", sources: list[str] | None = None):
    """Synchronous wrapper for search_scientific"""
    # Validate required parameters
    if not query or not isinstance(query, str):
        error_msg = "Query parameter is required and must be a non-empty string"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "results": [],
            "engine_used": engine or "duckduckgo"
        }
    
    # Validate and set defaults for optional parameters
    if engine is None:
        engine = "duckduckgo"
    
    # Run the async function
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(search_scientific(query, engine, sources))
        return result
    except Exception as e:
        error_msg = f"Error executing search_scientific: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "results": [],
            "engine_used": engine
        }
    finally:
        loop.close()

def sync_get_url_content(url: str, download_images: bool = True):
    """Synchronous wrapper for get_url_content"""
    # Validate required parameters
    if not url or not isinstance(url, str):
        error_msg = "URL parameter is required and must be a non-empty string"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "text": "",
            "images": []
        }
    
    # Set default for optional parameters
    if download_images is None:
        download_images = True
    
    # Run the async function
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(get_url_content(url, download_images))
        return result
    except Exception as e:
        error_msg = f"Error executing get_url_content: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "text": "",
            "images": []
        }
    finally:
        loop.close()

def sync_grep_url_content(url: str, pattern: str, case_sensitive: bool = False):
    """Synchronous wrapper for grep_url_content"""
    # Validate required parameters
    if not url or not isinstance(url, str):
        error_msg = "URL parameter is required and must be a non-empty string"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "matches": []
        }
    
    if not pattern or not isinstance(pattern, str):
        error_msg = "Pattern parameter is required and must be a non-empty string"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "matches": []
        }
    
    # Run the async function
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(grep_url_content(url, pattern))
        return result
    except Exception as e:
        error_msg = f"Error executing grep_url_content: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "matches": []
        }
    finally:
        loop.close()

def sync_process_pdf(url_or_path: str, extract_images: bool = True):
    """Synchronous wrapper for process_pdf"""
    # Validate required parameters
    if not url_or_path or not isinstance(url_or_path, str):
        error_msg = "URL or path parameter is required and must be a non-empty string"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "text": "",
            "images": [],
            "pages": 0
        }
    
    # Set default for optional parameters
    if extract_images is None:
        extract_images = True
    
    # Run the async function
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(process_pdf(url_or_path, extract_images))
        return result
    except Exception as e:
        error_msg = f"Error executing process_pdf: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "text": "",
            "images": [],
            "pages": 0
        }
    finally:
        loop.close()

# MCP tool definitions
MCP_TOOLS = [
    {
        "name": "search_scientific",
        "description": "Search for scientific information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "engine": {
                    "type": "string",
                    "description": "The search engine to use",
                    "default": "duckduckgo",
                    "enum": ["duckduckgo", "google_scholar", "pubmed", "arxiv", "semantic_scholar"]
                },
                "sources": {
                    "type": "array",
                    "description": "Optional list of source domains to filter results",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_url_content",
        "description": "Extract content from a URL",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to extract content from"
                },
                "download_images": {
                    "type": "boolean",
                    "description": "Whether to download images from the URL",
                    "default": True
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "grep_url_content",
        "description": "Find text in a URL using regex",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to search"
                },
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                }
            },
            "required": ["url", "pattern"]
        }
    },
    {
        "name": "process_pdf",
        "description": "Extract text and images from a PDF file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url_or_path": {
                    "type": "string", 
                    "description": "URL or file path to the PDF"
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Whether to extract images from the PDF",
                    "default": True
                }
            },
            "required": ["url_or_path"]
        }
    }
]

# Map tool names to their implementation functions
TOOL_HANDLERS = {
    "search_scientific": sync_search_scientific,
    "get_url_content": sync_get_url_content,
    "grep_url_content": sync_grep_url_content,
    "process_pdf": sync_process_pdf
}

def process_request(request):
    """Process a JSON-RPC request and return a response"""
    if not isinstance(request, dict):
        logger.error(f"Invalid request: {request}")
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32600,
                "message": "Invalid Request"
            }
        }
    
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    logger.debug(f"Received request: method={method}, params={params}")
    
    # Default to empty string for null values
    if method is None:
        method = ""
        logger.warning("Received request with null method, treating as empty string")
        
    # Ensure params is a valid dictionary
    if not isinstance(params, dict):
        params = {}
        logger.warning("Received non-dictionary params, using empty dictionary instead")
    
    # Handle initialize request according to 2024-11-05 MCP specification
    if method == "initialize":
        logger.info(f"Handling {method} request with params: {json.dumps(params)}")
        
        # Extract protocol version from request with default value
        protocol_version = params.get("protocolVersion", "2024-11-05")
        client_capabilities = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})
        
        if not isinstance(client_capabilities, dict):
            client_capabilities = {}
            
        if not isinstance(client_info, dict):
            client_info = {}
        
        logger.info(f"Client requested protocol version: {protocol_version}")
        logger.info(f"Client capabilities: {client_capabilities}")
        logger.info(f"Client info: {client_info}")
        
        # Verify protocol version compatibility
        supported_versions = ["2024-11-05"]
        if protocol_version not in supported_versions:
            logger.warning(f"Client requested unsupported protocol version: {protocol_version}, using fallback: {supported_versions[0]}")
            protocol_version = supported_versions[0]
        
        # Return proper initialize response according to MCP spec
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": protocol_version,
                "serverInfo": {
                    "name": "scientific-search-mcp",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    }
                }
            }
        }
        
        logger.info(f"Sending initialize response with protocol version: {protocol_version}")
        return response
        
    # Handle initialized notification (LSP protocol)
    if method == "initialized" or method == "notifications/initialized":
        logger.info("Received initialized notification - ready for operation")
        return None
        
    # Handle tools/list request
    if method == "tools/list":
        logger.info("Handling tools/list request")
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": MCP_TOOLS
            }
        }
        logger.info(f"Returning list of {len(MCP_TOOLS)} tools")
        return response
        
    if method == "updateSettings" or method == "mcp/updateSettings":
        # Just acknowledge the settings update
        logger.info(f"Received settings update: {params}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }
        
    # Handle shutdown request
    if method == "shutdown":
        logger.info("Received shutdown request")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": None
        }
    
    # Handle exit notification
    if method == "exit":
        logger.info("Received exit notification, server will exit")
        # Return None for notification, exit will be handled in main loop
        return None
    
    # Handle ping/heartbeat - add support for connection testing
    if method == "$/ping" or method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"alive": True}
        }
    
    # Handle both MCP and LSP execute command
    if method in ["mcp/runTool", "workspace/executeCommand", "mcp/executeTool", "executeTool", "tools/call"]:
        # Extract tool name and arguments based on method
        tool_name = None
        tool_args = {}
        
        try:
            if method in ["mcp/runTool", "mcp/executeTool", "executeTool"]:
                # Handle different protocol variations
                if "name" in params:
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                elif "toolName" in params:
                    tool_name = params.get("toolName")
                    tool_args = params.get("arguments", {})
                else:
                    raise ValueError(f"Missing required 'name' parameter for {method}")
            elif method == "tools/call":
                # Handle tools/call method per MCP 2024-11-05 spec
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                if tool_name is None:
                    raise ValueError("Missing required 'name' parameter for tools/call")
            else:  # workspace/executeCommand
                tool_name = params.get("command")
                arguments = params.get("arguments", [])
                
                if tool_name is None:
                    raise ValueError("Missing required 'command' parameter")
                    
                # Convert arguments to kwargs
                if arguments and isinstance(arguments[0], dict):
                    tool_args = arguments[0]
                elif arguments:
                    param_names = ["query", "url", "pattern", "url_or_path"]
                    tool_args = {}
                    for i, arg in enumerate(arguments):
                        if i < len(param_names):
                            tool_args[param_names[i]] = arg
                else:
                    tool_args = {}
            
            # Validate tool name exists
            if not tool_name:
                raise ValueError("Tool name cannot be empty")
                
            logger.info(f"Running tool: {tool_name} with arguments: {tool_args}")
            
            if tool_name not in TOOL_HANDLERS:
                raise ValueError(f"Unknown tool: {tool_name}")
                
            # Get the tool handler
            handler = TOOL_HANDLERS[tool_name]
            
            # Get the tool schema to validate parameters
            tool_schema = None
            for tool in MCP_TOOLS:
                if tool["name"] == tool_name:
                    tool_schema = tool.get("inputSchema", {})
                    break
                    
            # Validate required parameters
            if tool_schema and "required" in tool_schema:
                required_params = tool_schema.get("required", [])
                for param in required_params:
                    if param not in tool_args or tool_args[param] is None:
                        raise ValueError(f"Missing required parameter: {param}")
            
            # Execute the tool
            result = handler(**tool_args)
            
            # Format response based on method
            if method in ["mcp/runTool", "mcp/executeTool", "executeTool", "tools/call"]:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result)
                            }
                        ],
                        "isError": False
                    }
                }
                logger.info(f"Tool {tool_name} executed successfully")
                return response
            else:  # workspace/executeCommand
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result)
                            }
                        ],
                        "isError": False
                    }
                }
                logger.info(f"Command {tool_name} executed successfully")
                return response
                
        except ValueError as e:
            logger.error(f"Invalid parameters: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": str(e)
                }
            }
        except Exception as e:
            logger.exception(f"Error running tool {tool_name}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
            logger.info(f"Sending error response for tool {tool_name}: {error_response}")
            return error_response
    
    # Unknown method
    logger.warning(f"Unknown method: {method}")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        }
    }

# Main entry point
if __name__ == "__main__":
    # Flag to indicate shutdown request
    shutdown_requested = False
    
    try:
        logger.info("Starting scientific_search_mcp with MCP protocol over stdio")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script path: {os.path.abspath(__file__)}")
        
        # Check if stdin/stdout are pipes or terminals
        logger.info(f"stdin isatty: {os.isatty(sys.stdin.fileno())}, stdout isatty: {os.isatty(sys.stdout.fileno())}")
        
        # Set up signal handlers to handle graceful shutdown
        def signal_handler(sig, frame):
            global shutdown_requested
            logger.info(f"Received signal {sig}, setting shutdown flag")
            shutdown_requested = True
            
        # Register for termination signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)
        
        # Keep track of connection state
        client_initialized = False
        
        # Main input loop
        while not shutdown_requested:
            try:
                # Use a non-blocking read with timeout to avoid freezing
                line = ""
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)  # Shorter timeout
                if readable:
                    line = sys.stdin.readline()
                
                # Skip if no input
                if not line:
                    # Continue to next iteration
                    time.sleep(0.1)  # Small delay
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Process the JSON-RPC message
                logger.debug(f"Raw input: {line.strip()}")
                
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON: {line.strip()}: {e}")
                    # Try to send an error response if possible
                    try:
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": None,  # We don't know the ID without parsing
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            }
                        }
                        print(json.dumps(error_response), flush=True)
                        logger.info("Sent parse error response")
                    except Exception as inner_e:
                        logger.error(f"Failed to send parse error response: {inner_e}")
                    continue
                
                # Extract important fields
                method = request.get("method", "")
                request_id = request.get("id")
                logger.info(f"Processing request: {method} (id: {request_id})")
                
                # Track connection state
                if method == "initialize":
                    logger.info("Received initialize request, establishing connection")
                elif method == "initialized" or method == "notifications/initialized":
                    logger.info("Received initialized notification, client is ready")
                    client_initialized = True
                elif method == "shutdown":
                    logger.info("Received shutdown request, preparing to exit")
                    # Send a proper response to shutdown
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": None
                    }
                    print(json.dumps(response), flush=True)
                    logger.info("Sent shutdown response")
                    shutdown_requested = True  # Set flag to exit after exit notification
                    # Wait for exit notification but timeout after 5 seconds
                    exit_timeout = time.time() + 5
                    received_exit = False
                    while time.time() < exit_timeout and not received_exit:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            exit_line = sys.stdin.readline()
                            if exit_line and "exit" in exit_line:
                                logger.info("Received exit notification after shutdown")
                                received_exit = True
                                break
                    # If timeout reached, exit anyway
                    if not received_exit:
                        logger.warning("Did not receive exit notification after shutdown, exiting anyway")
                    break
                elif method == "exit":
                    logger.info("Received exit notification, exiting immediately")
                    break
                
                # Process the request
                response = process_request(request)
                
                # Only send response if not None (handle notifications)
                if response is not None:
                    response_json = json.dumps(response)
                    print(response_json, flush=True)
                    logger.info(f"Sent response for request id: {request_id}")
            
            except BrokenPipeError as e:
                logger.error(f"Broken pipe error: {e}")
                logger.info("Client likely disconnected, shutting down")
                break
                
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                # Try to send an error response if we have enough context
                try:
                    if 'request_id' in locals() and request_id is not None:
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}"
                            }
                        }
                        print(json.dumps(error_response), flush=True)
                        logger.info(f"Sent error response for request id: {request_id}")
                except Exception as inner_error:
                    logger.error(f"Failed to send error response: {inner_error}")
        
        logger.info("Exiting scientific_search_mcp")
    
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        sys.exit(1) 