import streamlit as st
import os
from pathlib import Path
import tempfile # For creating temporary directories
import hashlib # For creating a hash of file contents for caching

# Import a reference to your rag_app module to access its functions and variables
import rag_app 
# Import LlamaIndex Settings to check and display configuration
from llama_index.core import Settings, VectorStoreIndex # Assuming VectorStoreIndex is returned by build_vector_index
# We might need Document and BaseNode if we are passing them around
from llama_index.core.schema import Document, BaseNode 
from llama_index.core.llms import LLM # For type hinting
from llama_index.core.query_engine import BaseQueryEngine # For type hinting

# Placeholder for functions that would come from rag_app.py or be redefined here
# In a real scenario, rag_app.py might be refactored into a library or its functions copied/adapted.

# Global settings (mimicking parts of rag_app.py's setup)
# These should ideally be loaded once.
# For Streamlit, we'll use its caching for expensive resources like models and indexes.

# --- Functions that would ideally be imported or adapted from rag_app.py ---

# Real model and settings initialization function
@st.cache_resource(show_spinner="正在初始化模型和设置 (首次运行可能需要一些时间)...")
def initialize_models_and_settings_real() -> bool:
    """
    Initializes embedding models, the default OpenAI LLM, and LlamaIndex Settings.
    Returns True if successful, False otherwise.
    """
    st.write("正在尝试初始化嵌入模型和 LLM (OpenAI gpt-4o)...")
    initialization_success = False
    
    try:
        rag_app.initialize_embed_model()
        embed_model_name = Settings.embed_model.model_name if hasattr(Settings.embed_model, 'model_name') else Settings.embed_model.__class__.__name__
        st.write(f"嵌入模型已初始化: {embed_model_name}")
        
        # Initialize the default OpenAI LLM and set it in Settings
        configured_llm = rag_app.initialize_llms() 

        if Settings.llm and hasattr(Settings.llm, 'model'):
            st.write(f"默认 LLM 已初始化: {Settings.llm.model}")
            st.success("模型和 LlamaIndex 设置已就绪！")
            initialization_success = True
        else:
            st.error("默认 LLM (OpenAI) 未能初始化。请检查 API 密钥和控制台日志。")
            initialization_success = False
            
    except Exception as e:
        st.error(f"模型初始化期间发生错误: {e}")
        st.warning("请检查控制台以获取来自 rag_app.py 的更详细错误信息。")
        initialization_success = False
    
    return initialization_success

# Function to create a hash from a list of UploadedFile objects
# This helps in making the cache specific to the set of uploaded files.
def create_files_hash(uploaded_files):
    hasher = hashlib.md5()
    for uploaded_file in sorted(uploaded_files, key=lambda f: f.name):
        uploaded_file.seek(0)
        hasher.update(uploaded_file.getvalue())
    return hasher.hexdigest()

# Refactored function to build the index, LlamaParse logic removed
@st.cache_resource(show_spinner="正在处理文档并构建索引...")
def build_index_from_uploaded_files(_files_tuple, chunk_size: int, chunk_overlap: int) -> VectorStoreIndex | None:
    """Processes uploaded files using LlamaParse if available, otherwise basic loaders, builds, and returns a VectorStoreIndex."""
    _uploaded_files = list(_files_tuple)

    if not _uploaded_files:
        st.warning("没有上传文件需要处理。") 
        return None

    all_loaded_documents = []
    with tempfile.TemporaryDirectory() as temp_dir_path_str:
        temp_dir = Path(temp_dir_path_str)
        st.write(f"已创建用于处理的临时目录: {temp_dir}")
        
        saved_file_paths = []
        for uploaded_file in _uploaded_files:
            file_path = temp_dir / uploaded_file.name
            try:
                with open(file_path, "wb") as f:
                    uploaded_file.seek(0) # Ensure we read from the beginning
                    f.write(uploaded_file.getbuffer())
                saved_file_paths.append(str(file_path))
            except Exception as e:
                st.error(f"保存上传文件 '{uploaded_file.name}' 到临时目录时出错: {e}")
                continue 
        
        if not saved_file_paths:
            st.error("未能保存任何上传的文件以供处理。")
            return None

        # --- LlamaParse First, Standard Loading Fallback ---
        llama_cloud_api_key_is_set = bool(rag_app.LLAMA_CLOUD_API_KEY) # Check if key is loaded in rag_app

        if llama_cloud_api_key_is_set:
            st.write(f"检测到 LLAMA_CLOUD_API_KEY。将尝试使用 LlamaParse 解析文件: {saved_file_paths}")
            try:
                # Assuming parse_documents_with_llamaparse is now in rag_app
                parsed_docs = rag_app.parse_documents_with_llamaparse(file_paths=saved_file_paths, result_type="text")
                if parsed_docs:
                    all_loaded_documents.extend(parsed_docs)
                    st.write(f"LlamaParse 成功解析了 {len(parsed_docs)} 个文档。")
                else:
                    st.write("LlamaParse 未能从文件中解析出任何文档，或者返回了空结果。")
            except Exception as e:
                st.warning(f"LlamaParse 处理期间出错: {e}。将尝试使用标准加载器回退。")
        
        # Fallback or if LlamaParse is not configured/failed for some files
        # We should process files that LlamaParse might not have (fully) processed or if it wasn't used.
        # If LlamaParse was attempted and returned some documents, we might not want to re-process them with load_documents.
        # For simplicity now, if LlamaParse didn't populate all_loaded_documents, load_documents will try all.
        # A more sophisticated approach would be to track which files LlamaParse failed on.
        
        if not all_loaded_documents: # If LlamaParse wasn't used or failed to produce any docs
            if llama_cloud_api_key_is_set:
                 st.write(f"LlamaParse 未返回任何文档，正在使用标准加载器回退处理: {saved_file_paths}")
            else:
                 st.write(f"LLAMA_CLOUD_API_KEY 未设置。正在使用标准加载器处理: {saved_file_paths}")
            
            try:
                # Use load_documents for all saved file paths
                loaded_docs_fallback = rag_app.load_documents(saved_file_paths)
                all_loaded_documents.extend(loaded_docs_fallback)
                st.write(f"标准加载器加载了 {len(loaded_docs_fallback)} 个文档。")
            except Exception as e:
                st.error(f"使用标准加载器加载文档时出错: {e}")
        
        if not all_loaded_documents:
            st.error("未能成功加载或解析任何文档。无法构建索引。")
            return None

        st.write(f"要进行分块的总文档数: {len(all_loaded_documents)}")
        try:
            # Pass the chunk_size and chunk_overlap to chunk_documents
            nodes = rag_app.chunk_documents(all_loaded_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not nodes:
                st.error("分块未产生任何节点。无法构建索引。")
                return None
            st.write(f"分块完成。生成了 {len(nodes)} 个节点。")
        except Exception as e:
            st.error(f"文档分块期间出错: {e}")
            return None

        try:
            vector_index = rag_app.build_vector_index(nodes)
            if not vector_index:
                st.error("构建向量索引失败。")
                return None
            st.success("向量索引构建成功！")
            return vector_index 
        except Exception as e:
            st.error(f"构建向量索引时出错: {e}")
            return None

# Function for getting response (remains largely the same)
@st.cache_data(show_spinner="正在查询...") 
def get_rag_response_real(_query_engine: BaseQueryEngine | None, query_text: str):
    # (Keep the implementation of get_rag_response_real as it was)
    if not query_text:
        return "请输入查询内容。"
    if not _query_engine:
        st.error("查询引擎不可用。")
        return "错误：查询引擎未就绪。"
    
    st.write(f"正在查询: '{query_text}'")
    try:
        response = _query_engine.query(query_text)
        return str(response)
    except Exception as e:
        st.error(f"查询执行期间出错: {e}")
        # Check for specific error types if needed
        if "Connection error" in str(e):
             st.error("无法连接到 LLM API。请检查网络连接和 API 密钥。")
        return f"错误：无法获取响应。 {e}"

# --- Streamlit App UI ---
st.set_page_config(page_title="我的 RAG 应用", layout="wide")
st.title("📚 我的定制 RAG 应用")
st.write("上传您的文档，提出问题，获取由 RAG 驱动的答案！")

models_initialized = initialize_models_and_settings_real()

# Use session state to store the vector index and query engine
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None 
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None 

with st.sidebar:
    st.header("⚙️ 文档处理") # CN: Simplified header
    
    # Removed LlamaParse Checkbox
    # use_llamaparse = st.checkbox("✨ 使用 LlamaParse ... ", value=True) 
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "选择文件 (.txt, .pdf, .md, .docx)", 
        accept_multiple_files=True,
        type=["txt", "pdf", "md", "docx"]
    )

    # Chunking parameters
    st.subheader("分块参数") # CN: Chunking Parameters
    chunk_size_input = st.number_input("块大小 (Chunk Size)", min_value=100, max_value=8000, value=1000, step=100, help="每个文本块的目标大小（以 Token 计）。")
    chunk_overlap_input = st.number_input("块重叠 (Chunk Overlap)", min_value=0, max_value=1000, value=200, step=50, help="相邻文本块之间的重叠 Token 数。")

    # Process Button Logic
    if uploaded_files:
        st.write(f"已选择 {len(uploaded_files)} 个文件:")
        for f in uploaded_files:
            st.caption(f.name)
            
        if st.button("⚙️ 处理文档并构建索引"):
            if models_initialized:
                # Pass tuple for caching
                files_tuple = tuple(uploaded_files) # Convert to tuple for st.cache_resource
                
                # Call the refactored function to build the index
                # The 'use_llamaparse' argument is no longer needed as logic is internal to build_index_from_uploaded_files
                # Pass the chunk_size and chunk_overlap from the UI
                vector_index_result = build_index_from_uploaded_files(files_tuple, chunk_size=chunk_size_input, chunk_overlap=chunk_overlap_input) 
                
                if vector_index_result:
                    st.session_state.vector_index = vector_index_result
                    st.success("文档处理和索引构建完成！")
                    # Now, create the query engine using the built index and default LLM
                    with st.spinner("正在创建查询引擎..."): # CN
                        try:
                            # create_query_engine now defaults to Settings.llm (gpt-4o)
                            query_engine = rag_app.create_query_engine(st.session_state.vector_index)
                            if query_engine:
                                st.session_state.query_engine = query_engine
                                st.success("查询引擎已就绪！") # CN
                            else:
                                st.error("未能创建查询引擎。") # CN 
                                st.session_state.query_engine = None # Ensure it's cleared
                        except Exception as e:
                             st.error(f"创建查询引擎时出错: {e}") # CN
                             st.session_state.query_engine = None
                else:
                    # If index building failed, clear relevant states
                    st.session_state.vector_index = None
                    st.session_state.query_engine = None
                    st.error("未能处理文档并构建索引。请查看上方错误信息。") # CN
            else:
                st.error("模型未初始化。无法处理文档。")
    else:
        st.write("请上传一些文档以开始。")

st.header("💬 提问")

# Check if query engine is ready before allowing queries
if st.session_state.query_engine:
    st.info("查询引擎已准备就绪，可以开始提问。") # CN
    query_text = st.text_input("在此输入您的问题:", key="query_input")
    if st.button("提交查询", key="submit_button"):
        if query_text:
            # Directly use the query engine stored in session state
            response_text = get_rag_response_real(st.session_state.query_engine, query_text)
            st.subheader("回答:")
            st.markdown(response_text)
            # TODO: Optionally display source nodes from response object if needed
        else:
            st.warning("请输入问题。")
elif uploaded_files and not st.session_state.vector_index: # Files uploaded but not processed
     st.warning("请先点击侧边栏按钮处理上传的文档。") # CN
elif not uploaded_files:
     st.info("请使用侧边栏上传文档以启用查询。")
else: # Index might be built, but query engine creation failed
     st.error("查询引擎未能创建，无法进行查询。请检查处理步骤中的错误。") # CN

# For running this app: streamlit run streamlit_app.py 