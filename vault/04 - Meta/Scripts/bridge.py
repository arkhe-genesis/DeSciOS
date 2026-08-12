#!/usr/bin/env python3
# bridge.py — Conecta análises do BLOCK 11 ao Obsidian via Local REST API

import requests
import yaml
import hashlib
from datetime import datetime

OBSIDIAN_API = "http://localhost:27123"  # Local REST API plugin
VAULT_PATH = "/path/to/vault"

def post_analysis(title, content, domain, version, selo):
    """Envia uma análise para o Obsidian como nota."""
    hash_val = hashlib.sha256(content.encode()).hexdigest()
    frontmatter = {
        "title": title,
        "type": "analysis",
        "domain": domain,
        "version": version,
        "date": datetime.now().isoformat(),
        "status": "rascunho",
        "hash": hash_val,
        "selo": selo,
        "tags": ["analysis", domain]
    }
    note = f"---\n{yaml.dump(frontmatter)}\n---\n\n{content}"
    # Criar arquivo via API
    path = f"01 - Analyses/{domain}/{title}.md"
    response = requests.put(
        f"{OBSIDIAN_API}/vault/{path}",
        json={"content": note}
    )
    return response.status_code == 200
