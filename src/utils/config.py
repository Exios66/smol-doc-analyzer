@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    generation_model: str
    max_concurrency: int
    max_retries: int
    openrouter_app_url: str
    openrouter_app_name: str

    skeleton_output_dir: Path
    document_output_dir: Path
    memo_output_dir: Path
    provenance_log_path: Path

    taxonomy_path: Path
    claim_schema_path: Path

    @classmethod
    def load(cls) -> "Config":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            generation_model=os.getenv("GENERATION_MODEL", "anthropic/claude-sonnet-4.5"),
            max_concurrency=int(os.getenv("GENERATION_MAX_CONCURRENCY", "4")),
            max_retries=int(os.getenv("GENERATION_MAX_RETRIES", "5")),
            openrouter_app_url=os.getenv("OPENROUTER_APP_URL", ""),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", ""),
            skeleton_output_dir=_path("SKELETON_OUTPUT_DIR", "data/synthetic/skeletons"),
            document_output_dir=_path("DOCUMENT_OUTPUT_DIR", "data/synthetic/documents"),
            memo_output_dir=_path("MEMO_OUTPUT_DIR", "data/synthetic/memos"),
            provenance_log_path=_path("PROVENANCE_LOG_PATH", "data/provenance_log.jsonl"),
            taxonomy_path=REPO_ROOT / "taxonomy" / "acord_form_categories.yaml",
            claim_schema_path=REPO_ROOT / "data" / "schemas" / "claim_skeleton.schema.json",
        )
