import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_parse import LlamaParse

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# --- Global Placeholders - USER MUST VERIFY/UPDATE THESE --- #
OPENAI_DEFAULT_MODEL_NAME = "gpt-4o"
# --- End Global Placeholders --- #

def initialize_embed_model():
    """Initializes and sets the HuggingFace embedding model."""
    print("Initializing embedding model BAAI/bge-small-zh...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
    Settings.embed_model = embed_model
    # Settings.llm = None # Explicitly set LLM to None for now, will configure DeepSeek later
    # The above line is now handled by initialize_llms()
    print("Embedding model initialized.")
    return embed_model

def initialize_llms() -> OpenAI | None:
    """Initializes and sets OpenAI gpt-4o as the default LLM."""
    print("Initializing LLM...")
    
    openai_llm_instance = None
    Settings.llm = None # Start with None

    if OPENAI_API_KEY and OPENAI_DEFAULT_MODEL_NAME: # OPENAI_DEFAULT_MODEL_NAME should be "gpt-4o"
        print(f"Configuring OpenAI model: {OPENAI_DEFAULT_MODEL_NAME}")
        try:
            openai_llm_instance = OpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_DEFAULT_MODEL_NAME,
            )
            Settings.llm = openai_llm_instance # Set as default
            print(f"Default LLM set to OpenAI: {Settings.llm.model}")
        except Exception as e:
            print(f"Error initializing OpenAI LLM: {e}")
            print("OpenAI LLM will not be used.")
            Settings.llm = None # Ensure it remains None on error
    else:
        print("OpenAI API key or model name not configured. Skipping OpenAI LLM initialization.")
        Settings.llm = None

    if not Settings.llm:
        print("Warning: Default LLM (OpenAI) could not be configured.")
        print("RAG query functionality will likely not work.")
        
    return openai_llm_instance # Return the instance (which is also Settings.llm if successful)

def load_documents(input_sources: list[str]) -> list[Document]:
    """
    Loads documents from a list of input sources.
    Sources can be file paths, directory paths, or URLs.
    """
    documents = []
    file_sources = []
    url_sources = []

    for source in input_sources:
        if source.startswith("http://") or source.startswith("https://"):
            url_sources.append(source)
        else:
            # Assume it's a local file or directory path
            # SimpleDirectoryReader can handle both individual files and directories
            file_sources.append(source)

    if url_sources:
        print(f"Loading from URLs: {url_sources}")
        web_reader = TrafilaturaWebReader()
        # TrafilaturaWebReader's load_data expects a list of URLs
        web_docs = web_reader.load_data(urls=url_sources)
        documents.extend(web_docs)
        print(f"Loaded {len(web_docs)} document(s) from URLs.")

    # SimpleDirectoryReader can take a list of files or a single directory.
    # To handle a mix of files and directories robustly, we can iterate or
    # rely on its recursive behavior if a directory is given.
    # For simplicity here, if multiple file_sources are given, we assume they are individual files
    # or directories that SimpleDirectoryReader can process.
    # If input_files is a list of file paths, SimpleDirectoryReader handles it.
    # If one of the paths is a directory, it will load files from that directory.
    if file_sources:
        print(f"Loading from file/directory sources: {file_sources}")
        # We can pass multiple individual files or directories to input_files
        # However, if we pass a list of mixed files and directories where one dir might contain another file in the list,
        # it might lead to loading duplicates if not careful.
        # For now, let's assume users will provide distinct paths or one directory.
        # A more robust way would be to instantiate SimpleDirectoryReader for each directory
        # and then handle individual files.

        # Let's try to load them one by one to give more granular feedback
        # and allow SimpleDirectoryReader to figure out if it's a file or dir.
        temp_docs = []
        for source_path in file_sources:
            if os.path.isfile(source_path):
                print(f"Loading file: {source_path}")
                reader = SimpleDirectoryReader(input_files=[source_path])
            elif os.path.isdir(source_path):
                print(f"Loading files from directory: {source_path}")
                # required_exts can be used if we only want specific extensions from a directory
                reader = SimpleDirectoryReader(input_dir=source_path, recursive=True)
            else:
                print(f"Warning: Source path {source_path} is neither a file nor a directory. Skipping.")
                continue
            
            loaded = reader.load_data()
            temp_docs.extend(loaded)
            print(f"Loaded {len(loaded)} document(s) from {source_path}.")
        documents.extend(temp_docs)


    print(f"Total documents loaded: {len(documents)}")
    return documents

def parse_documents_with_llamaparse(file_paths: list[str], result_type: str = "text") -> list[Document]:
    """
    Parses documents using LlamaParse.

    Args:
        file_paths: A list of local file paths to parse.
        result_type: The desired output type from LlamaParse ("text" or "markdown").

    Returns:
        A list of LlamaIndex Document objects.
    """
    if not LLAMA_CLOUD_API_KEY:
        print("LLAMA_CLOUD_API_KEY not found in .env. Skipping LlamaParse.")
        # Fallback or error handling: maybe return empty list or raise exception
        # For now, let's return empty and let the caller handle it.
        return []

    print(f"Initializing LlamaParse with result_type='{result_type}'...")
    try:
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type=result_type,  # "text" or "markdown"
            verbose=True,
            # language="zh" # Uncomment if documents are primarily in Chinese and experiencing issues
        )
    except Exception as e:
        print(f"Error initializing LlamaParse: {e}")
        return []

    print(f"Starting LlamaParse for {len(file_paths)} file(s): {file_paths}")
    
    # LlamaParse's load_data expects a list of file paths.
    # It will return a list of LlamaIndex Document objects directly.
    try:
        parsed_documents = parser.load_data(file_paths)
        print(f"LlamaParse completed. Parsed {len(parsed_documents)} document(s).")
        return parsed_documents
    except Exception as e:
        print(f"Error during LlamaParse processing: {e}")
        # Handle potential errors, e.g., API errors, file not found by parser
        return []

def chunk_documents(documents: list[Document], 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200) -> list[BaseNode]:
    """
    Chunks a list of documents into smaller text nodes.

    Args:
        documents: A list of LlamaIndex Document objects.
        chunk_size: The target size of each text chunk (in tokens).
        chunk_overlap: The number of tokens to overlap between chunks.

    Returns:
        A list of LlamaIndex BaseNode objects (text chunks).
    """
    print(f"Starting chunking: {len(documents)} document(s), chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    node_parser = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separator=" ", # Default is " "
        # backup_separators=["\\n"], # Default backup
        # tokenizer=Settings.tokenizer # Uses the tokenizer from Settings.embed_model by default if available
    )
    
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f"Chunking complete. Generated {len(nodes)} nodes.")
    # At this point, 'nodes' is a list of TextNode or similar, which are subtypes of BaseNode.
    # Each node contains text and metadata.
    # If a specific JSON output format is required here, we would implement the serialization.
    # For now, we return the list of Node objects for further processing (e.g., embedding).
    return nodes

def build_vector_index(nodes: list[BaseNode]) -> VectorStoreIndex | None:
    """
    Builds a LlamaIndex VectorStoreIndex from a list of nodes.

    Args:
        nodes: A list of LlamaIndex BaseNode objects (text chunks).

    Returns:
        A LlamaIndex VectorStoreIndex object, or None if an error occurs or no nodes are provided.
    """
    if not nodes:
        print("No nodes provided to build the index. Skipping index construction.")
        return None

    print(f"Building vector index from {len(nodes)} nodes...")
    try:
        # VectorStoreIndex will use the globally configured Settings.embed_model
        # and Settings.llm (though llm is more for querying than indexing itself)
        index = VectorStoreIndex(nodes)
        print("Vector index built successfully.")
        return index
    except Exception as e:
        print(f"Error building vector index: {e}")
        return None

def create_query_engine(index: VectorStoreIndex, similarity_top_k: int = 3, llm: LLM | None = None):
    """
    Creates a query engine from a VectorStoreIndex.
    Uses the provided LLM if given, otherwise defaults to Settings.llm.

    Args:
        index: The VectorStoreIndex object.
        similarity_top_k: The number of top similar documents to retrieve.
        llm: Optional LLM instance to override the default from Settings.llm.

    Returns:
        A LlamaIndex query engine object, or None if an error occurs or no LLM is available.
    """
    # 确定要使用的 LLM：优先使用传入的 llm 参数，否则使用全局 Settings.llm
    llm_to_use = llm or Settings.llm

    if not llm_to_use: # 检查是否有可用的 LLM (无论是传入的还是全局的)
        print("错误：LLM 不可用 (既未提供，也未在 Settings 中配置)。")
        print("没有 LLM 无法创建查询引擎。")
        return None
    
    # 打印将要使用的 LLM 信息
    llm_model_name = getattr(llm_to_use, 'model', llm_to_use.__class__.__name__) # 获取模型名或类名
    print(f"正在创建查询引擎，similarity_top_k={similarity_top_k}，使用的 LLM: {llm_model_name}")
    
    try:
        # 使用确定的 llm_to_use 来创建查询引擎
        query_engine = index.as_query_engine(
            llm=llm_to_use, 
            similarity_top_k=similarity_top_k
        )
        print("查询引擎创建成功。")
        return query_engine
    except Exception as e:
        print(f"创建查询引擎时出错: {e}")
        return None

if __name__ == '__main__':
    initialize_embed_model()
    configured_openai_llm = initialize_llms()

    example_sources = [
        "data", 
        "https://www.google.com" # Example URL
    ]
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample text file.")

    loaded_documents = load_documents(example_sources)

    if loaded_documents:
        print(f"\nSuccessfully loaded {len(loaded_documents)} documents.")
        chunked_nodes = chunk_documents(loaded_documents)
        if chunked_nodes:
            print(f"\nSuccessfully chunked documents into {len(chunked_nodes)} nodes.")
            vector_index = build_vector_index(chunked_nodes)
            if vector_index:
                print("\nSuccessfully created vector index.")
                query_engine_instance = create_query_engine(vector_index)
                if query_engine_instance:
                    print("\n--- Ready to Query (Command Line) ---")
                    while True:
                        try:
                            user_query = input("Query: ")
                            if user_query.lower() == 'exit': break
                            if not user_query.strip(): continue
                            response = query_engine_instance.query(user_query)
                            print("\nResponse:", str(response), "\n")
                        except KeyboardInterrupt:
                            print("\nExiting.")
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                else:
                    print("Failed to create query engine.")
            else:
                 print("Failed to build vector index.")
        else:
            print("Chunking resulted in no nodes.")
    else:
        print("No documents loaded.") 