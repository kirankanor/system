// Generated placeholder data. Regenerate from validated manifests and canonical source files.
window.RAG_COOKBOOK_DATA = {
  "categories": [
    {
      "id": "ingestion",
      "name": "Ingestion",
      "icon": "\u21e9",
      "description": "Reusable ingestion capabilities."
    },
    {
      "id": "chunking",
      "name": "Chunking",
      "icon": "\u2702",
      "description": "Reusable chunking capabilities."
    },
    {
      "id": "embeddings",
      "name": "Embeddings",
      "icon": "\u25a6",
      "description": "Reusable embeddings capabilities."
    },
    {
      "id": "retrieval",
      "name": "Retrieval",
      "icon": "\u2315",
      "description": "Reusable retrieval capabilities."
    },
    {
      "id": "reranking",
      "name": "Reranking",
      "icon": "\u25a5",
      "description": "Reusable reranking capabilities."
    },
    {
      "id": "context",
      "name": "Context Building",
      "icon": "\u25a4",
      "description": "Reusable context building capabilities."
    },
    {
      "id": "evaluation",
      "name": "Evaluation",
      "icon": "\u2713",
      "description": "Reusable evaluation capabilities."
    },
    {
      "id": "security",
      "name": "Security",
      "icon": "\u2b21",
      "description": "Reusable security capabilities."
    },
    {
      "id": "storage",
      "name": "Storage",
      "icon": "\u25a4",
      "description": "Reusable storage capabilities."
    }
  ],
  "components": [
    {
      "id": "text-normalizer",
      "name": "Text Normalizer",
      "category": "ingestion",
      "icon": "\u21e9",
      "description": "Normalize extracted text with explicit, non-destructive rules.",
      "source": "components/ingestion/text_normalizer/text_normalizer.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "ingestion",
        "text normalizer"
      ],
      "code": "# TODO: Implement Normalize extracted text with explicit, non-destructive rules.\n# Canonical source: components/ingestion/text_normalizer/text_normalizer.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "pdf-extraction",
      "name": "Pdf Extraction",
      "category": "ingestion",
      "icon": "\u21e9",
      "description": "Extract PDF content with declared parser dependencies and limitations.",
      "source": "components/ingestion/pdf_extraction/pdf_extraction.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "ingestion",
        "pdf extraction"
      ],
      "code": "# TODO: Implement Extract PDF content with declared parser dependencies and limitations.\n# Canonical source: components/ingestion/pdf_extraction/pdf_extraction.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "fixed-size-chunker",
      "name": "Fixed Size Chunker",
      "category": "chunking",
      "icon": "\u2702",
      "description": "Split text into fixed-size chunks with explicit overlap semantics.",
      "source": "components/chunking/fixed_size/fixed_size_chunker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "chunking",
        "fixed size chunker"
      ],
      "code": "# TODO: Implement Split text into fixed-size chunks with explicit overlap semantics.\n# Canonical source: components/chunking/fixed_size/fixed_size_chunker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "recursive-chunker",
      "name": "Recursive Chunker",
      "category": "chunking",
      "icon": "\u2702",
      "description": "Split text using preferred separators and recursive fallback.",
      "source": "components/chunking/recursive/recursive_chunker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "chunking",
        "recursive chunker"
      ],
      "code": "# TODO: Implement Split text using preferred separators and recursive fallback.\n# Canonical source: components/chunking/recursive/recursive_chunker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "token-chunker",
      "name": "Token Chunker",
      "category": "chunking",
      "icon": "\u2702",
      "description": "Split text according to a declared tokenizer's token budget.",
      "source": "components/chunking/token_based/token_chunker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "chunking",
        "token chunker"
      ],
      "code": "# TODO: Implement Split text according to a declared tokenizer's token budget.\n# Canonical source: components/chunking/token_based/token_chunker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "markdown-chunker",
      "name": "Markdown Chunker",
      "category": "chunking",
      "icon": "\u2702",
      "description": "Chunk Markdown while preserving heading hierarchy where feasible.",
      "source": "components/chunking/markdown_aware/markdown_chunker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "chunking",
        "markdown chunker"
      ],
      "code": "# TODO: Implement Chunk Markdown while preserving heading hierarchy where feasible.\n# Canonical source: components/chunking/markdown_aware/markdown_chunker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "structure-aware-chunker",
      "name": "Structure Aware Chunker",
      "category": "chunking",
      "icon": "\u2702",
      "description": "Chunk structured documents using their semantic boundaries.",
      "source": "components/chunking/structure_aware/structure_aware_chunker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "chunking",
        "structure aware chunker"
      ],
      "code": "# TODO: Implement Chunk structured documents using their semantic boundaries.\n# Canonical source: components/chunking/structure_aware/structure_aware_chunker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "embedding-protocol",
      "name": "Embedding Protocol",
      "category": "embeddings",
      "icon": "\u25a6",
      "description": "Define framework-neutral embedding input/output contracts.",
      "source": "components/embeddings/embedding_protocol/embedding_protocol.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "embeddings",
        "embedding protocol"
      ],
      "code": "# TODO: Implement Define framework-neutral embedding input/output contracts.\n# Canonical source: components/embeddings/embedding_protocol/embedding_protocol.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "local-embeddings",
      "name": "Local Embeddings",
      "category": "embeddings",
      "icon": "\u25a6",
      "description": "Provide a local model embedding adapter with explicit model dependencies.",
      "source": "components/embeddings/local_embeddings/local_embeddings.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "embeddings",
        "local embeddings"
      ],
      "code": "# TODO: Implement Provide a local model embedding adapter with explicit model dependencies.\n# Canonical source: components/embeddings/local_embeddings/local_embeddings.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "remote-embeddings",
      "name": "Remote Embeddings",
      "category": "embeddings",
      "icon": "\u25a6",
      "description": "Provide a remote embedding API adapter with explicit timeout and error behavior.",
      "source": "components/embeddings/remote_embeddings/remote_embeddings.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "embeddings",
        "remote embeddings"
      ],
      "code": "# TODO: Implement Provide a remote embedding API adapter with explicit timeout and error behavior.\n# Canonical source: components/embeddings/remote_embeddings/remote_embeddings.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "batch-embeddings",
      "name": "Batch Embeddings",
      "category": "embeddings",
      "icon": "\u25a6",
      "description": "Batch embedding requests with size limits and predictable failures.",
      "source": "components/embeddings/batch_embeddings/batch_embeddings.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "embeddings",
        "batch embeddings"
      ],
      "code": "# TODO: Implement Batch embedding requests with size limits and predictable failures.\n# Canonical source: components/embeddings/batch_embeddings/batch_embeddings.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "dense-retrieval",
      "name": "Dense Retrieval",
      "category": "retrieval",
      "icon": "\u2315",
      "description": "Retrieve candidates using vector similarity.",
      "source": "components/retrieval/dense/dense_retrieval.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "retrieval",
        "dense retrieval"
      ],
      "code": "# TODO: Implement Retrieve candidates using vector similarity.\n# Canonical source: components/retrieval/dense/dense_retrieval.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "sparse-retrieval",
      "name": "Sparse Retrieval",
      "category": "retrieval",
      "icon": "\u2315",
      "description": "Retrieve candidates using lexical or sparse signals.",
      "source": "components/retrieval/sparse/sparse_retrieval.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "retrieval",
        "sparse retrieval"
      ],
      "code": "# TODO: Implement Retrieve candidates using lexical or sparse signals.\n# Canonical source: components/retrieval/sparse/sparse_retrieval.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "hybrid-retrieval",
      "name": "Hybrid Retrieval",
      "category": "retrieval",
      "icon": "\u2315",
      "description": "Combine dense and sparse retrieval with documented score fusion.",
      "source": "components/retrieval/hybrid/hybrid_retrieval.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "retrieval",
        "hybrid retrieval"
      ],
      "code": "# TODO: Implement Combine dense and sparse retrieval with documented score fusion.\n# Canonical source: components/retrieval/hybrid/hybrid_retrieval.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "metadata-filter",
      "name": "Metadata Filter",
      "category": "retrieval",
      "icon": "\u2315",
      "description": "Filter retrieval candidates using validated metadata criteria.",
      "source": "components/retrieval/metadata_filter/metadata_filter.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "retrieval",
        "metadata filter"
      ],
      "code": "# TODO: Implement Filter retrieval candidates using validated metadata criteria.\n# Canonical source: components/retrieval/metadata_filter/metadata_filter.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "retrieval-result-normalizer",
      "name": "Retrieval Result Normalizer",
      "category": "retrieval",
      "icon": "\u2315",
      "description": "Normalize retrieval results into a stable framework-neutral contract.",
      "source": "components/retrieval/result_normalizer/retrieval_result_normalizer.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "retrieval",
        "retrieval result normalizer"
      ],
      "code": "# TODO: Implement Normalize retrieval results into a stable framework-neutral contract.\n# Canonical source: components/retrieval/result_normalizer/retrieval_result_normalizer.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "cross-encoder-reranker",
      "name": "Cross Encoder Reranker",
      "category": "reranking",
      "icon": "\u25a5",
      "description": "Rerank candidate passages with a cross-encoder model.",
      "source": "components/reranking/cross_encoder/cross_encoder_reranker.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "reranking",
        "cross encoder reranker"
      ],
      "code": "# TODO: Implement Rerank candidate passages with a cross-encoder model.\n# Canonical source: components/reranking/cross_encoder/cross_encoder_reranker.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "relevance-ranking",
      "name": "Relevance Ranking",
      "category": "reranking",
      "icon": "\u25a5",
      "description": "Order candidates using a documented relevance score.",
      "source": "components/reranking/relevance_ranking/relevance_ranking.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "reranking",
        "relevance ranking"
      ],
      "code": "# TODO: Implement Order candidates using a documented relevance score.\n# Canonical source: components/reranking/relevance_ranking/relevance_ranking.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "context-selector",
      "name": "Context Selector",
      "category": "context",
      "icon": "\u25a4",
      "description": "Select evidence passages for downstream generation.",
      "source": "components/context/context_selector/context_selector.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "context",
        "context selector"
      ],
      "code": "# TODO: Implement Select evidence passages for downstream generation.\n# Canonical source: components/context/context_selector/context_selector.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "token-budget",
      "name": "Token Budget",
      "category": "context",
      "icon": "\u25a4",
      "description": "Enforce a model-specific token budget for selected context.",
      "source": "components/context/token_budget/token_budget.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "context",
        "token budget"
      ],
      "code": "# TODO: Implement Enforce a model-specific token budget for selected context.\n# Canonical source: components/context/token_budget/token_budget.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "citation-formatter",
      "name": "Citation Formatter",
      "category": "context",
      "icon": "\u25a4",
      "description": "Format source references and citation metadata consistently.",
      "source": "components/context/citation_formatter/citation_formatter.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "context",
        "citation formatter"
      ],
      "code": "# TODO: Implement Format source references and citation metadata consistently.\n# Canonical source: components/context/citation_formatter/citation_formatter.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "retrieval-metrics",
      "name": "Retrieval Metrics",
      "category": "evaluation",
      "icon": "\u2713",
      "description": "Calculate retrieval quality metrics over labeled examples.",
      "source": "components/evaluation/retrieval_metrics/retrieval_metrics.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "evaluation",
        "retrieval metrics"
      ],
      "code": "# TODO: Implement Calculate retrieval quality metrics over labeled examples.\n# Canonical source: components/evaluation/retrieval_metrics/retrieval_metrics.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "answer-metrics",
      "name": "Answer Metrics",
      "category": "evaluation",
      "icon": "\u2713",
      "description": "Evaluate answer quality using declared metrics and evaluation dependencies.",
      "source": "components/evaluation/answer_metrics/answer_metrics.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "evaluation",
        "answer metrics"
      ],
      "code": "# TODO: Implement Evaluate answer quality using declared metrics and evaluation dependencies.\n# Canonical source: components/evaluation/answer_metrics/answer_metrics.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "evaluation-dataset",
      "name": "Evaluation Dataset",
      "category": "evaluation",
      "icon": "\u2713",
      "description": "Represent evaluation cases with explicit schemas and validation.",
      "source": "components/evaluation/evaluation_dataset/evaluation_dataset.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "evaluation",
        "evaluation dataset"
      ],
      "code": "# TODO: Implement Represent evaluation cases with explicit schemas and validation.\n# Canonical source: components/evaluation/evaluation_dataset/evaluation_dataset.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "input-validation",
      "name": "Input Validation",
      "category": "security",
      "icon": "\u2b21",
      "description": "Validate component inputs and apply size/type constraints.",
      "source": "components/security/input_validation/input_validation.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "security",
        "input validation"
      ],
      "code": "# TODO: Implement Validate component inputs and apply size/type constraints.\n# Canonical source: components/security/input_validation/input_validation.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "metadata-validation",
      "name": "Metadata Validation",
      "category": "security",
      "icon": "\u2b21",
      "description": "Validate document metadata against an explicit schema.",
      "source": "components/security/metadata_validation/metadata_validation.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "security",
        "metadata validation"
      ],
      "code": "# TODO: Implement Validate document metadata against an explicit schema.\n# Canonical source: components/security/metadata_validation/metadata_validation.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "retrieved-content-safety",
      "name": "Retrieved Content Safety",
      "category": "security",
      "icon": "\u2b21",
      "description": "Handle retrieved text as untrusted data and document prompt-injection boundaries.",
      "source": "components/security/retrieved_content/retrieved_content_safety.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "security",
        "retrieved content safety"
      ],
      "code": "# TODO: Implement Handle retrieved text as untrusted data and document prompt-injection boundaries.\n# Canonical source: components/security/retrieved_content/retrieved_content_safety.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "document-store",
      "name": "Document Store",
      "category": "storage",
      "icon": "\u25a4",
      "description": "Define document-store contracts independent of a specific backend.",
      "source": "components/storage/document_store/document_store.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "storage",
        "document store"
      ],
      "code": "# TODO: Implement Define document-store contracts independent of a specific backend.\n# Canonical source: components/storage/document_store/document_store.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "vector-store-protocol",
      "name": "Vector Store Protocol",
      "category": "storage",
      "icon": "\u25a4",
      "description": "Define vector-store operations and result contracts.",
      "source": "components/storage/vector_store_protocol/vector_store_protocol.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "storage",
        "vector store protocol"
      ],
      "code": "# TODO: Implement Define vector-store operations and result contracts.\n# Canonical source: components/storage/vector_store_protocol/vector_store_protocol.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    },
    {
      "id": "storage-adapters",
      "name": "Storage Adapters",
      "category": "storage",
      "icon": "\u25a4",
      "description": "Provide opt-in backend adapters with isolated optional dependencies.",
      "source": "components/storage/adapters/storage_adapters.py",
      "version": "0.1.0",
      "dependencies": [],
      "tags": [
        "storage",
        "storage adapters"
      ],
      "code": "# TODO: Implement Provide opt-in backend adapters with isolated optional dependencies.\n# Canonical source: components/storage/adapters/storage_adapters.py",
      "test_status": "Scaffold only \u2014 implementation and tests are pending."
    }
  ]
};
