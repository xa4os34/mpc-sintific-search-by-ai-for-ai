#!/bin/bash

set -e

WORKSPACE_DIR="$(pwd)"
CURSOR_CONFIG_DIR="${HOME}/.cursor"
MCP_CONFIG_FILE="${CURSOR_CONFIG_DIR}/mcp.json"
VENV_DIR="${WORKSPACE_DIR}/.venv_mcp"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -f "${WORKSPACE_DIR}/scientific_search_mcp.py" ]; then
    echo "Error: scientific_search_mcp.py not found in workspace"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

echo "Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
deactivate

mkdir -p "${CURSOR_CONFIG_DIR}"

if [ ! -f "${MCP_CONFIG_FILE}" ]; then
    echo "{\"mcpServers\": {}}" > "${MCP_CONFIG_FILE}"
fi

# Make the script executable
chmod +x "${WORKSPACE_DIR}/scientific_search_mcp.py"

echo "Updating MCP configuration..."
TMP_FILE=$(mktemp)

# Use jq to add the scientific-search entry
jq --arg python "${VENV_PYTHON}" --arg script "${WORKSPACE_DIR}/scientific_search_mcp.py" --arg workdir "${WORKSPACE_DIR}" '.mcpServers["scientific-search"] = {
    "command": $python,
    "args": [$script],
    "cwd": $workdir,
    "enabled": true
}' "${MCP_CONFIG_FILE}" > "${TMP_FILE}" && mv "${TMP_FILE}" "${MCP_CONFIG_FILE}"

# Delete old log file to start fresh
rm -f "${WORKSPACE_DIR}/.logs/scientific_search_mcp.log"

echo "Scientific Search MCP tool installed successfully"
echo "Log files will be available at: ${WORKSPACE_DIR}/.logs/scientific_search_mcp.log"
echo ""
echo "Available tools:"
echo "- search_scientific: Search scientific sources using DuckDuckGo"
echo "- get_url_content: Extract content from a URL"
echo "- grep_url_content: Find text in a URL using regex"
echo "- process_pdf: Extract text and images from a PDF file"
echo ""
echo "IMPORTANT: You must COMPLETELY CLOSE and restart Cursor for changes to take effect."
echo "Your MCP tool will appear as @scientific-search in Cursor after restart."