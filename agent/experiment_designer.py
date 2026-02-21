import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# LLM & Embeddings
from langchain_openai import ChatOpenAI
# Replacement: Import SentenceTransformerEmbeddings (instead of HuggingFaceEmbeddings)
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Document Loaders
from langchain_community.document_loaders import PyPDFLoader

# Try to import Docx2txtLoader; if not available, handle gracefully
try:
    from langchain_community.document_loaders import Docx2txtLoader
    HAS_DOCX_LOADER = True
except ImportError:
    HAS_DOCX_LOADER = False

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store
from langchain_community.vectorstores import Chroma

# Graph
from langgraph.graph import StateGraph, END

# ----------------------------
# 1. State Definition
# ----------------------------
class ExperimentState(BaseModel):
    user_query: str = Field(..., description="User's natural language experimental goal")
    platform_constraints: str = Field(..., description="Experimental platform capability constraints")
    background_knowledge: str = Field(default="", description="Background knowledge retrieved via RAG")
    measurable_indicators: List[str] = Field(default_factory=list)
    protocol_steps: List[str] = Field(default_factory=list)
    draft_json: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# 2. Experiment Design Agent (with SentenceTransformer Embeddings)
# ----------------------------
class ExperimentDesignAgent:
    def __init__(
        self,
        cfg,
        # LLM configuration (required)
        llm_api_key: str,
        llm_base_url: str,
        llm_model_name: str,
        temperature: float = 0.0
    ):
        """
        Agent for generating experimental protocols, using SentenceTransformer for local embedding (replaces HuggingFaceEmbeddings)
        """
        # === 1. Simplified Configuration: Keep only SentenceTransformer-related settings (remove OpenAI embedding config) ===
        st_model_name = cfg['LOCAL_EMBEDDING_MODEL']  # New: SentenceTransformer model name
        chunk_size = cfg['CHUNK_SIZE']
        chunk_overlap = cfg['CHUNK_OVERLAP']
        documents = self._load_documents(cfg['document_paths'])
        platform_constraints = cfg['PLATFORM_CONSTRAINTS']
        
        if not documents:
            raise ValueError("No valid documents loaded. Please check file paths and formats!")

        # === 2. Split documents ===
        splits = self._split_documents(documents, chunk_size, chunk_overlap)
        if not splits:
            raise ValueError("Document splitting resulted in empty chunks. Please verify document content!")

        # === 3. Initialize SentenceTransformer Embeddings (core replacement) ===
        print(f"🔄 Using local SentenceTransformer model: {st_model_name}")
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=st_model_name,
            model_kwargs={"device": "cpu"},  # Optional: Change to "cuda" if GPU is available
            encode_kwargs={"normalize_embeddings": True}
        )

        # === 4. Build vector store ===
        self.vector_db = Chroma.from_documents(documents=splits, embedding=self.embeddings)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})

        # === 5. Initialize LLM (unchanged) ===
        self.llm = ChatOpenAI(
            model=llm_model_name,
            api_key=SecretStr(llm_api_key),
            base_url=llm_base_url.strip(),
            temperature=temperature
        )

        self.platform_constraints = platform_constraints
        self.graph: Runnable = self._build_graph()
        print(f"✅ ExperimentDesignAgent initialized! Documents: {len(documents)}, Chunks: {len(splits)}")

    def _load_documents(self, paths: List[str]) -> List[Document]:
        docs = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            suffix = p.suffix.lower()
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(p))
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"📄 Loaded PDF: {p.name} ({len(loaded)} pages)")
                elif suffix == ".docx":
                    if not HAS_DOCX_LOADER:
                        raise ImportError("Missing docx2txt. Install with: pip install docx2txt")
                    loader = Docx2txtLoader(str(p))
                    loaded = loader.load()
                    docs.extend(loaded)
                    print(f"📘 Loaded DOCX: {p.name}")
                else:
                    print(f"⚠️ Skipping unsupported format: {p.name} (only .pdf and .docx supported)")
                    continue
            except Exception as e:
                print(f"❌ Failed to load {p.name}: {e}")
        return docs

    def _split_documents(self, docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "!", "?", "；", ";", " ", ""]
        )
        return splitter.split_documents(docs)

    def _retrieve_background(self, query: str) -> str:
        """Perform RAG retrieval (logic unchanged)"""
        retrieved_docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in retrieved_docs])

    # ----------------------------
    # LangGraph Four-Step Workflow (completely unchanged)
    # ----------------------------
    def _build_graph(self) -> Runnable:
        workflow = StateGraph(ExperimentState)
        workflow.add_node("step1_retrieve_background", self._step1_retrieve_background)
        workflow.add_node("step2_define_indicators", self._step2_define_indicators)
        workflow.add_node("step3_generate_protocol", self._step3_generate_protocol)
        workflow.add_node("step4_output_draft", self._step4_output_draft)

        workflow.set_entry_point("step1_retrieve_background")
        workflow.add_edge("step1_retrieve_background", "step2_define_indicators")
        workflow.add_edge("step2_define_indicators", "step3_generate_protocol")
        workflow.add_edge("step3_generate_protocol", "step4_output_draft")
        workflow.add_edge("step4_output_draft", END)
        return workflow.compile()

    def _step1_retrieve_background(self, state: ExperimentState) -> dict:
        bg = self._retrieve_background(state.user_query)
        return {"background_knowledge": bg}

    def _step2_define_indicators(self, state: ExperimentState) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """You are a senior cell biologist. Based on the following information, list feasible measurable indicators:

            User Goal: {user_query}
            Background Knowledge: {background}
            Platform Constraints: {constraints}

            Requirements:
            - Only output qualitative or quantitative indicators directly related to the goal (e.g., "cell circularity analysis")
            - Must be measurable under the given platform constraints
            - Keep only the single easiest-to-implement indicator, and explain it briefly

            Measurable Indicator:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "user_query": state.user_query,
            "background": state.background_knowledge,
            "constraints": state.platform_constraints
        })
        indicators = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        return {"measurable_indicators": indicators}

    def _step3_generate_protocol(self, state: ExperimentState) -> dict:
        prompt = ChatPromptTemplate.from_template(
            """You are defining the final experimental output for an automated microscopy and analysis system. 
            Generate concise, professional output descriptions based on the following:

            Experimental Goal: {user_query}
            Indicators to Observe: {indicators}
            Platform Constraints: {constraints}
            Relevant Mechanisms: {background}

            Requirements:
            1. Each description must include three elements:
               - Required image acquisition (specify objective magnification, channels, Z-stack, etc.)
               - Image processing or analysis operations (e.g., segmentation, denoising, merging)
               - Final output (quantitative data or processed image)
            2. Use standard bioimaging terminology to ensure system interpretability.
            3. For each indicator, keep only the easiest-to-implement experimental output.
            4. Be fluent, concise, and accurate—do not add explanations or justifications.

            Example:
            “Acquire 60× actin images, segment with Cellpose, and output single-cell circularity and area.”

            Now generate the experimental output list:"""
        )

        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "user_query": state.user_query,
            "indicators": ", ".join(state.measurable_indicators),
            "constraints": state.platform_constraints,
            "background": state.background_knowledge
        })

        steps = []
        for line in raw.strip().split("\n"):
            line_clean = line.strip()
            if line_clean:
                steps.append(line_clean)

        return {"protocol_steps": steps}

    def _step4_output_draft(self, state: ExperimentState) -> dict:
        bg = state.background_knowledge
        bg_summary = (bg[:600] + "...") if len(bg) > 600 else bg
        draft = {
            "experiment_goal": state.user_query,
            "platform_constraints": state.platform_constraints,
            "background_summary": bg_summary,
            "measurable_indicators": state.measurable_indicators,
            "protocol": state.protocol_steps
        }
        return {"draft_json": draft}

    def run(self, user_query: str, platform_constraints: Optional[str] = None) -> Dict[str, Any]:
        if platform_constraints is None:
            platform_constraints = self.platform_constraints
        initial_state = ExperimentState(
            user_query=user_query,
            platform_constraints=platform_constraints
        )
        result = self.graph.invoke(initial_state)
        return result["draft_json"]


# ----------------------------
# Usage Example (adapted to new configuration)
# ----------------------------
if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from config.agent_config import cfg_tabletop
    from config.agent_config import openai_api_key, base_url, model_name

    # === Configure your experimental environment ===
    LLM_API_KEY = openai_api_key  
    LLM_BASE_URL = base_url  
    LLM_MODEL = model_name  

    # === Initialize Agent ===
    agent = ExperimentDesignAgent(
        cfg_tabletop['lmps']['Task_designer'],
        llm_api_key=LLM_API_KEY,
        llm_base_url=LLM_BASE_URL,
        llm_model_name=LLM_MODEL,
    )

    # === Generate experimental protocol ===
    result = agent.run(
        "Taxol treatment of HeLa cells has been completed for 2 hours. Mitochondria and actin have been fluorescently labeled, and the field of view has been precisely aligned to the target region. I now wish to further analyze the effect of taxol on HeLa cell growth."
    )

    # === Print results ===
    print("\n" + "=" * 60)
    print("🧪 Experimental Protocol Draft")
    print("=" * 60)
    print(f"🎯 Goal: {result['experiment_goal']}\n")

    print("📊 Measurable Indicators:")
    for i, ind in enumerate(result["measurable_indicators"], 1):
        print(f"  {i}. {ind}")

    print("\n🔬 Protocol Steps:")
    for step in result["protocol"]:
        print(f"  - {step}")
        