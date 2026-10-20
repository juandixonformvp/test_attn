# Attention-network graph analytics

Read-only analytics for the Neo4j attention graph described by the supplied 76ers-event schema.
The scripts create purpose-specific graph projections rather than running algorithms over the full
heterogeneous graph, where structural Comment and Account nodes would dominate centrality.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NEO4J_URI='bolt://host:port'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='set-this-outside-git'
```

Do not commit `.env` files or passwords. All database sessions use Neo4j's read access mode.

### Proxy-aware HTTP transport

If raw Bolt TCP is unavailable, use Neo4j's existing HTTP transaction endpoint. The toolkit detects
an `http://` or `https://` URI and uses a proxy-aware HTTP client while keeping every analysis script
unchanged:

```bash
export NEO4J_URI='https://graph.example.com'
python scripts/01_claim_battlefield.py
```

HTTPS is required by default. For a deliberately trusted plaintext endpoint, the client requires an
explicit opt-in:

```bash
export NEO4J_URI='http://host:7474'
export NEO4J_ALLOW_INSECURE_HTTP=1
```

The HTTP adapter calls `/db/<database>/tx/commit`, honors the standard HTTP proxy environment, and
raises Neo4j query errors without logging credentials.

## Analyses

| Script | Analysis | Primary outputs |
|---|---|---|
| `01_claim_battlefield.py` | Claim contestation and positive/negative influence | ranked claims |
| `02_narrative_coalitions.py` | Account–narrative communities | coalitions and memberships |
| `03_bridge_actors_claims.py` | Cross-narrative brokers | account participation and claim bridges |
| `04_reaction_polarization.py` | Reaction divergence | distributions and pairwise JSD |
| `05_conversation_cascades.py` | Reply-tree shapes | cascade metrics and taxonomy |
| `06_cross_platform_migration.py` | Narrative timing across platforms | daily series, peaks, and lags |
| `07_entity_coattention.py` | Entity associations | PMI pairs and entity communities |
| `08_brand_halo.py` | Organic, promotional, and contractual halo | channel scores and counterfactual lift |
| `09_amplifier_analysis.py` | Account amplification and claim concentration | account and claim rankings |
| `10_contradiction_structure.py` | Signed argumentative structure | components, cycles, and unstable claims |

Run one analysis from the repository root:

```bash
python scripts/01_claim_battlefield.py --output-dir output --confidence 0.75
```

Run all analyses:

```bash
for script in scripts/[0-9][0-9]_*.py; do python "$script"; done
```

Every script has `--help`. Results are written beneath `output/<analysis>/`. Inferred edges are
filtered with `--confidence`; run at several thresholds (for example 0.60, 0.75, and 0.90) to test
sensitivity. Stub posts are excluded whenever authorship or post performance matters.

## Interpretation boundaries

- Results describe the captured conversation, not the general public.
- Temporal precedence is not proof that one platform caused activity on another.
- `RESTATES` is kept separate from argumentative support to avoid counting paraphrase as evidence.
- Reactions are grouped by actual target label because `REACTS` can target Post, Entity, or Theme.
- Sponsorship output separates organic attention from modeled contractual halo.
