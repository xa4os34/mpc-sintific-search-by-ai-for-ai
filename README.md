# Scientific Search MCP for Cursor

> [!WARNING]
> Vibecoded by AI.

This is a Model Control Protocol (MCP) implementation for Cursor that provides scientific search capabilities through various academic search engines.

## Features

- Integration with Cursor using the MCP protocol
- Scientific search across multiple engines:
  - DuckDuckGo (general web search)
  - Google Scholar (academic papers search)
  - PubMed (biomedical literature search)
  - arXiv (physics, mathematics, computer science, etc.)
  - Semantic Scholar (AI-powered research paper search via Tor proxy)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cursor-scientific-search.git
cd cursor-scientific-search
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

For specific search engines, you may need to install additional dependencies:

```bash
# For arXiv search
pip install arxiv

# For Google Scholar
pip install scholarly

# For PubMed
pip install pymed

# For Semantic Scholar (requires Tor)
pip install requests[socks]
```

### Tor Setup for Semantic Scholar

The Semantic Scholar search uses Tor to bypass rate limits and IP blocks. To use this feature:

1. Install Tor:
   - **Linux**: `sudo apt install tor` (Debian/Ubuntu) or `sudo pacman -S tor` (Arch)
   - **macOS**: `brew install tor`
   - **Windows**: Download and install the [Tor Browser](https://www.torproject.org/download/) which includes the Tor service

2. Start the Tor service:
   - **Linux**: `sudo systemctl start tor` or `sudo service tor start`
   - **macOS**: `brew services start tor`
   - **Windows**: The Tor Browser includes the service, or you can use [Tor Expert Bundle](https://www.torproject.org/download/tor/)

3. Verify Tor is running on port 9050 (default):
   ```bash
   nc -z localhost 9050 && echo "Tor is running"
   ```

4. The scientific search MCP will automatically route Semantic Scholar requests through Tor.

## Usage

1. Start the MCP server:
```bash
python scientific_search_mcp.py
```

2. In Cursor, connect to the MCP server.

3. Use the scientific search tools:

### Search Scientific

Search for scientific papers across different engines:

```python
# Search arXiv for quantum computing papers
results = await mcp_scientific_search_search_scientific(
    query="quantum computing", 
    engine="arxiv"
)

# Search PubMed for COVID-19 research
results = await mcp_scientific_search_search_scientific(
    query="COVID-19 treatment", 
    engine="pubmed"
)

# Search Google Scholar for machine learning papers
results = await mcp_scientific_search_search_scientific(
    query="transformers neural networks", 
    engine="google_scholar"
)

# Search Semantic Scholar for AI ethics papers (via Tor)
results = await mcp_scientific_search_search_scientific(
    query="AI ethics", 
    engine="semantic_scholar"
)
```

### Get URL Content

Extract content from a web page:

```python
content = await mcp_scientific_search_get_url_content(
    url="https://example.com/research-paper"
)
```

### Grep URL Content

Search for specific text patterns in a web page:

```python
matches = await mcp_scientific_search_grep_url_content(
    url="https://example.com/research-paper",
    pattern="neural network"
)
```

### Process PDF

Extract text and images from a PDF:

```python
pdf_content = await mcp_scientific_search_process_pdf(
    url_or_path="https://example.com/paper.pdf",
    extract_images=True
)
```

## Notes

- Google Scholar searches may be rate-limited if used excessively. The implementation uses free proxies to reduce the risk of blocking.
- PubMed requires an email address for usage tracking. Set the `PUBMED_EMAIL` environment variable or it will use a default.
- Semantic Scholar searches are routed through Tor to avoid rate limits and IP blocking.
- For best results, use specific search terms relevant to the domain.

## Troubleshooting

- **Semantic Scholar Connection Issues**: Make sure Tor is running on localhost:9050. If you experience issues, restart the Tor service.
- **Slow Responses**: Tor routing may cause slower responses for Semantic Scholar searches.

## License

[MIT License](LICENSE)
