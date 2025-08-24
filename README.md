# 🕷️ Crawl4AI RAG Agent

An intelligent documentation crawler and retrieval-augmented generation (RAG) system, powered by **Crawl4AI 0.6.2** and Pydantic AI. This project enables you to crawl, chunk, and vectorize documentation from any website, `.txt`/Markdown pages (llms.txt), or sitemap, and interact with the knowledge base using a Streamlit interface with Supabase cloud storage.

## 🚀 Features

- **🕷️ Flexible Crawling**: Handles regular websites, `.txt`/Markdown pages, and sitemaps
- **⚡ Parallel Processing**: Memory-adaptive crawling with intelligent resource management
- **🧠 Smart Chunking**: Intelligent content chunking that adapts to document structures
- **☁️ Cloud Storage**: Supabase PostgreSQL with pgvector for fast semantic retrieval
- **🎯 Streamlit UI**: Beautiful web interface for querying your documentation
- **🔧 Extensible**: Modular scripts for various crawling and RAG workflows

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- Supabase account and project
- Git

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd crawl4AI-agent-v2
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade "crawl4ai[all]"
playwright install
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key
MODEL_CHOICE=gpt-4o-mini
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

### 5. Set Up Supabase Database
```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create vector index
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

---

## 🎯 Usage

### Crawling Documentation
```bash
python insert_docs.py https://docs.pydantic.dev/2.11/ \
    --table-name documents \
    --chunk-size 1000 \
    --max-depth 3 \
    --max-concurrent 10
```

### Running the Streamlit Interface
```bash
streamlit run streamlit_app.py
```
Visit: http://localhost:8501

---

## Usage

### 1. Crawling and Inserting Documentation

The main entry point for crawling and vectorizing documentation is [`insert_docs.py`](insert_docs.py):

#### Supported URL Types

- **Regular documentation sites:** Recursively crawls all internal links, deduplicates by URL (ignoring fragments).
- **Markdown or .txt pages (such as llms.txt):** Fetches and chunks Markdown content.
- **Sitemaps (`sitemap.xml`):** Batch-crawls all URLs listed in the sitemap.

#### Example Usage

```bash
python insert_docs.py <URL> [--table-name mydocs] [--embedding-model all-MiniLM-L6-v2] [--chunk-size 1000] [--max-depth 3] [--max-concurrent 10] [--batch-size 100]
```

**Arguments:**
- `URL`: The root URL, .txt file, or sitemap to crawl.
- `--table-name`: Supabase table name (default: `documents`)
- `--embedding-model`: Embedding model for vector storage (default: `all-MiniLM-L6-v2`)
- `--chunk-size`: Maximum characters per chunk (default: `1000`)
- `--max-depth`: Recursion depth for regular URLs (default: `3`)
- `--max-concurrent`: Max parallel browser sessions (default: `10`)
- `--batch-size`: Batch size for Supabase insertion (default: `100`)

**Examples for each type (regular URL, .txt, sitemap):**
```bash
# Crawl Pydantic documentation (single page)
python insert_docs.py https://docs.pydantic.dev/2.11/ --max-depth 1

# Crawl more deeply with custom settings
python insert_docs.py https://docs.pydantic.dev/2.11/ --max-depth 3 --max-concurrent 5 --chunk-size 800

# Crawl a markdown file
python insert_docs.py https://ai.pydantic.dev/llms-full.txt

# Crawl from sitemap
python insert_docs.py https://ai.pydantic.dev/sitemap.xml
```

#### Chunking Strategy

The enhanced chunking algorithm intelligently adapts to different content structures:

1. **Header-based splitting:** First tries to split by `#`, `##`, `###`, `####` headers
2. **Paragraph-based splitting:** If no headers found, splits by double newlines (paragraphs)
3. **Sentence-based splitting:** If no paragraphs, splits by sentence boundaries
4. **Character-based splitting:** As a fallback, splits by character count
5. **Quality filtering:** Removes chunks shorter than 50 characters

All chunks are less than the specified `--chunk-size` (default: 1000 characters) and optimized for vector search.

#### Metadata

Each chunk is stored with:
- Source URL
- Chunk index
- Extracted headers
- Character and word counts

---

### 2. Example Scripts

The `crawl4AI-examples/` folder contains modular scripts illustrating different crawling and chunking strategies:

- **`3-crawl_sitemap_in_parallel.py`:** Batch-crawls a list of URLs from a sitemap in parallel with memory tracking.
- **`4-crawl_llms_txt.py`:** Crawls a Markdown or `.txt` file, splits by headers, and prints chunks.
- **`5-crawl_site_recursively.py`:** Recursively crawls all internal links from a root URL, deduplicating by URL (ignoring fragments).

You can use these scripts directly for experimentation or as templates for custom crawlers.

---

### 3. Running the Streamlit RAG Interface

After crawling and inserting docs, launch the Streamlit app for semantic search and question answering:

```bash
streamlit run streamlit_app.py
```

- The interface will be available at [http://localhost:8501](http://localhost:8501)
- Query your documentation using natural language and get context-rich answers.

---

## 📁 Project Structure
```
crawl4AI-agent-v2/
├── 🕷️ crawl4AI-examples/          # Example crawling scripts
│   ├── 1-crawl_single_page.py
│   ├── 2-crawl_docs_sequential.py
│   ├── 3-crawl_sitemap_in_parallel.py
│   ├── 4-crawl_llms_txt.py
│   └── 5-crawl_site_recursively.py
├── 📄 insert_docs.py              # Main crawling script
├── 📄 rag_agent.py                # RAG agent implementation
├── 📄 streamlit_app.py            # Streamlit web interface
├── 📄 utils.py                    # Utility functions
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env.example                # Environment template
└── 📄 README.md                   # This file
```

## 🔧 Configuration Options

### Crawling Parameters
- `--chunk-size`: Content chunk size (default: 1000)
- `--max-depth`: Maximum crawl depth (default: 3)
- `--max-concurrent`: Concurrent requests (default: 10)
- `--embedding-model`: Embedding model (default: all-MiniLM-L6-v2)

### Supported URL Types
- **Regular websites**: Recursive crawling with deduplication
- **Markdown/.txt files**: Direct content processing
- **Sitemaps**: Batch processing of URL lists

---

## Advanced Usage & Customization

- **Chunking:** Tune `--chunk-size` for your retrieval use case.
- **Embeddings:** Swap out the embedding model with `--embedding-model`.
- **Crawling:** Adjust `--max-depth` and `--max-concurrent` for large sites.
- **Vector DB:** Use your own Supabase table for multiple projects.

---

## Troubleshooting

- **Dependencies:** Ensure all dependencies are installed and environment variables are set.
- **Memory management:** For large sites, increase memory or decrease `--max-concurrent`.
- **Crawling issues:** If you encounter crawling issues, try running the example scripts for isolated debugging.
- **Supabase setup:** Make sure your Supabase project has the `vector` extension enabled.
- **Playwright browsers:** Run `playwright install` if you get browser-related errors.
- **Chunking issues:** The new chunking algorithm should handle most content types automatically.

## 🐛 Troubleshooting

### Common Issues
1. **Playwright Installation**: Run `playwright install` if browser errors occur
2. **Memory Issues**: Reduce `--max-concurrent` for large sites
3. **API Limits**: Check OpenAI and Supabase rate limits
4. **Environment Variables**: Ensure all required keys are set in `.env`

### Debug Mode
```bash
# Test crawling functionality
python debug_crawl.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Crawl4AI](https://github.com/Crawl4AI/crawl4ai) for the powerful crawling framework
- [Pydantic AI](https://github.com/jxnl/pydantic-ai) for the RAG agent framework
- [Supabase](https://supabase.com) for the vector database
- [Streamlit](https://streamlit.io) for the web interface

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-username/crawl4AI-agent-v2/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Made with ❤️ for the AI community**