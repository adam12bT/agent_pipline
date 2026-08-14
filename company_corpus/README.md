# Company corpus

Place real, approved company documents in these folders:

- `past_proposals/`: previous tender proposals
- `cvs/`: consultant CVs
- `project_references/`: completed-project references

Then run `python ingest_company_corpus.py` from the project root. The generated
`.ingestion_manifest.json` records content hashes and prevents duplicate
indexing. Do not commit confidential company documents or the manifest unless
your repository access policy explicitly allows it.
