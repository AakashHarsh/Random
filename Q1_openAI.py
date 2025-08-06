import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
from pathlib import Path
import pickle

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Additional imports
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class NewsArticle:
    """Represents a news article"""
    article_id: str
    title: str
    content: str
    author: str
    publication_date: datetime
    source: str
    category: str
    url: str
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "publication_date": self.publication_date.isoformat(),
            "source": self.source,
            "category": self.category,
            "url": self.url,
            "tags": self.tags or []
        }
    
    def to_document(self) -> Document:
        """Convert to LangChain Document"""
        return Document(
            page_content=f"{self.title}\n\n{self.content}",
            metadata={
                "article_id": self.article_id,
                "title": self.title,
                "author": self.author,
                "publication_date": self.publication_date.isoformat(),
                "source": self.source,
                "category": self.category,
                "url": self.url,
                "tags": json.dumps(self.tags or [])
            }
        )

@dataclass
class SearchResult:
    """Represents a search result"""
    article: NewsArticle
    relevance_score: float
    snippet: str
    
@dataclass
class QAResponse:
    """Represents a question-answering response"""
    answer: str
    source_articles: List[NewsArticle]
    confidence_score: float
    context_used: List[str]
    tokens_used: int = 0
    cost: float = 0.0

class NewsRAGSystem:
    """Advanced RAG system for news search and question answering using LangChain"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-ada-002",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 persist_directory: Optional[str] = "./news_rag_db"):
        """
        Initialize the News RAG System
        
        Args:
            openai_api_key: OpenAI API key (if not in environment)
            model_name: OpenAI model to use for generation
            embedding_model: OpenAI embedding model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist FAISS index
        """
        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Text splitter for chunking long articles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Initialize or load vector store
        self.persist_directory = persist_directory
        self.vector_store = None
        self.articles = {}  # article_id -> NewsArticle
        
        # Create persist directory if it doesn't exist
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
        # Load existing data if available
        self._load_existing_data()
        
        # Initialize memory for conversational QA
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _load_existing_data(self):
        """Load existing FAISS index and article data if available"""
        if self.persist_directory:
            faiss_path = Path(self.persist_directory) / "index"
            articles_path = Path(self.persist_directory) / "articles.pkl"
            
            if faiss_path.exists():
                try:
                    self.vector_store = FAISS.load_local(
                        str(faiss_path), 
                        self.embeddings
                    )
                    print(f"Loaded existing FAISS index from {faiss_path}")
                except Exception as e:
                    print(f"Error loading FAISS index: {e}")
            
            if articles_path.exists():
                try:
                    with open(articles_path, 'rb') as f:
                        self.articles = pickle.load(f)
                    print(f"Loaded {len(self.articles)} articles from cache")
                except Exception as e:
                    print(f"Error loading articles: {e}")
    
    def _save_data(self):
        """Save FAISS index and article data"""
        if self.persist_directory and self.vector_store:
            faiss_path = Path(self.persist_directory) / "index"
            articles_path = Path(self.persist_directory) / "articles.pkl"
            
            # Save FAISS index
            self.vector_store.save_local(str(faiss_path))
            
            # Save articles
            with open(articles_path, 'wb') as f:
                pickle.dump(self.articles, f)
    
    def add_article(self, article: NewsArticle) -> str:
        """Add a news article to the system"""
        # Generate article ID if not provided
        if not article.article_id:
            content_hash = hashlib.md5(article.content.encode()).hexdigest()[:8]
            article.article_id = f"article_{content_hash}"
        
        # Store article
        self.articles[article.article_id] = article
        
        # Convert to document
        document = article.to_document()
        
        # Split into chunks if content is long
        if len(article.content) > 1000:
            splits = self.text_splitter.split_documents([document])
            # Preserve metadata in all chunks
            for split in splits:
                split.metadata.update(document.metadata)
        else:
            splits = [document]
        
        # Add to vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vector_store.add_documents(splits)
        
        # Save data
        self._save_data()
        
        return article.article_id
    
    def add_articles_batch(self, articles: List[NewsArticle]) -> List[str]:
        """Add multiple articles efficiently"""
        article_ids = []
        documents = []
        
        for article in articles:
            # Generate ID if needed
            if not article.article_id:
                content_hash = hashlib.md5(article.content.encode()).hexdigest()[:8]
                article.article_id = f"article_{content_hash}"
            
            # Store article
            self.articles[article.article_id] = article
            article_ids.append(article.article_id)
            
            # Convert to document
            document = article.to_document()
            
            # Split if needed
            if len(article.content) > 1000:
                splits = self.text_splitter.split_documents([document])
                for split in splits:
                    split.metadata.update(document.metadata)
                documents.extend(splits)
            else:
                documents.append(document)
        
        # Add all documents at once
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Save data
        self._save_data()
        
        return article_ids
    
    def search_articles(self, 
                       query: str, 
                       k: int = 5,
                       category_filter: Optional[str] = None,
                       date_filter: Optional[Tuple[datetime, datetime]] = None,
                       use_compression: bool = True) -> List[SearchResult]:
        """Search for relevant news articles with advanced filtering"""
        if not self.vector_store:
            return []
        
        # Create base retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k * 2}  # Get more results for filtering
        )
        
        # Use contextual compression for better results
        if use_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
        
        # Retrieve documents
        with get_openai_callback() as cb:
            docs = retriever.get_relevant_documents(query)
        
        # Process and filter results
        search_results = []
        seen_articles = set()
        
        for doc in docs:
            article_id = doc.metadata.get('article_id')
            
            # Skip if we've already seen this article
            if article_id in seen_articles:
                continue
            
            if article_id not in self.articles:
                continue
            
            article = self.articles[article_id]
            seen_articles.add(article_id)
            
            # Apply filters
            if category_filter and article.category != category_filter:
                continue
            
            if date_filter:
                start_date, end_date = date_filter
                if not (start_date <= article.publication_date <= end_date):
                    continue
            
            # Calculate relevance score (simplified - could use doc.metadata['score'] if available)
            relevance_score = 1.0 / (len(search_results) + 1)
            
            # Generate snippet
            snippet = self._generate_snippet(doc.page_content, query)
            
            search_results.append(SearchResult(
                article=article,
                relevance_score=relevance_score,
                snippet=snippet
            ))
            
            if len(search_results) >= k:
                break
        
        return search_results
    
    def ask_question(self, 
                    question: str, 
                    k: int = 3,
                    use_memory: bool = False) -> QAResponse:
        """Answer a question using RAG with advanced features"""
        if not self.vector_store:
            return QAResponse(
                answer="No articles in the database to answer questions.",
                source_articles=[],
                confidence_score=0.0,
                context_used=[]
            )
        
        # Create custom prompt
        prompt_template = """You are a helpful news analyst assistant. Use the following pieces of context from news articles to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        Always cite the source articles when providing information.

        Context from news articles:
        {context}

        Question: {question}

        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        if use_memory:
            # Use conversational retrieval chain for follow-up questions
            from langchain.chains import ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        
        # Get answer with token tracking
        with get_openai_callback() as cb:
            if use_memory:
                result = qa_chain({"question": question})
                answer = result["answer"]
                source_docs = []  # Conversational chain doesn't return sources directly
            else:
                result = qa_chain({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
        
        # Extract source articles
        source_articles = []
        context_used = []
        seen_articles = set()
        
        for doc in source_docs:
            article_id = doc.metadata.get('article_id')
            if article_id and article_id not in seen_articles:
                if article_id in self.articles:
                    article = self.articles[article_id]
                    source_articles.append(article)
                    context_used.append(f"{article.title} ({article.source})")
                    seen_articles.add(article_id)
        
        # Calculate confidence score based on number of sources and token usage
        confidence_score = min(len(source_articles) / k, 1.0) * 0.8 + 0.2
        
        return QAResponse(
            answer=answer,
            source_articles=source_articles,
            confidence_score=confidence_score,
            context_used=context_used,
            tokens_used=cb.total_tokens,
            cost=cb.total_cost
        )
    
    def ask_follow_up(self, question: str) -> QAResponse:
        """Ask a follow-up question using conversation memory"""
        return self.ask_question(question, use_memory=True)
    
    def summarize_articles(self, 
                          article_ids: Optional[List[str]] = None,
                          category: Optional[str] = None,
                          max_articles: int = 5) -> str:
        """Generate a summary of multiple articles"""
        # Select articles to summarize
        articles_to_summarize = []
        
        if article_ids:
            articles_to_summarize = [self.articles[aid] for aid in article_ids if aid in self.articles]
        elif category:
            articles_to_summarize = [a for a in self.articles.values() if a.category == category][:max_articles]
        else:
            articles_to_summarize = list(self.articles.values())[:max_articles]
        
        if not articles_to_summarize:
            return "No articles found to summarize."
        
        # Create summary prompt
        summary_prompt = PromptTemplate(
            template="""Please provide a comprehensive summary of the following news articles:

{articles_text}

Summary:""",
            input_variables=["articles_text"]
        )
        
        # Prepare articles text
        articles_text = "\n\n---\n\n".join([
            f"Title: {a.title}\nSource: {a.source}\nDate: {a.publication_date.strftime('%Y-%m-%d')}\n\n{a.content[:500]}..."
            for a in articles_to_summarize
        ])
        
        # Generate summary
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        
        with get_openai_callback() as cb:
            summary = summary_chain.run(articles_text=articles_text)
        
        return summary
    
    def get_trending_topics(self, 
                           time_window: Optional[Tuple[datetime, datetime]] = None,
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract trending topics using LLM analysis"""
        # Filter articles by time window
        articles_to_analyze = []
        for article in self.articles.values():
            if time_window:
                start_date, end_date = time_window
                if start_date <= article.publication_date <= end_date:
                    articles_to_analyze.append(article)
            else:
                articles_to_analyze.append(article)
        
        if not articles_to_analyze:
            return []
        
        # Create trending topics prompt
        trending_prompt = PromptTemplate(
            template="""Analyze the following news articles and identify the top {top_k} trending topics or themes:

{articles_summary}

For each topic, provide:
1. Topic name
2. Brief description
3. Number of related articles
4. Key insights

Format as JSON array.""",
            input_variables=["articles_summary", "top_k"]
        )
        
        # Prepare articles summary
        articles_summary = "\n".join([
            f"- {a.title} ({a.category}): {a.content[:200]}..."
            for a in articles_to_analyze[:20]  # Limit to prevent token overflow
        ])
        
        # Generate trending topics
        trending_chain = LLMChain(llm=self.llm, prompt=trending_prompt)
        
        with get_openai_callback() as cb:
            result = trending_chain.run(
                articles_summary=articles_summary,
                top_k=top_k
            )
        
        # Parse result
        try:
            topics = json.loads(result)
            return topics
        except:
            return [{"topic": "Analysis failed", "description": result}]
    
    def _generate_snippet(self, content: str, query: str, snippet_length: int = 150) -> str:
        """Generate a relevant snippet from content"""
        # Simple approach: return the beginning of the content
        # In production, you might want to use more sophisticated extraction
        if len(content) > snippet_length:
            return content[:snippet_length] + "..."
        return content
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "total_articles": len(self.articles),
            "categories": {},
            "sources": {},
            "date_range": None
        }
        
        if self.articles:
            # Count by category
            for article in self.articles.values():
                stats["categories"][article.category] = stats["categories"].get(article.category, 0) + 1
                stats["sources"][article.source] = stats["sources"].get(article.source, 0) + 1
            
            # Get date range
            dates = [a.publication_date for a in self.articles.values()]
            stats["date_range"] = {
                "earliest": min(dates).isoformat(),
                "latest": max(dates).isoformat()
            }
        
        return stats

# Example usage and demonstration
def create_sample_articles() -> List[NewsArticle]:
    """Create sample news articles for demonstration"""
    articles = [
        NewsArticle(
            article_id="1",
            title="OpenAI Releases GPT-4O Mini: A Game-Changer for AI Applications",
            content="""OpenAI has announced the release of GPT-4O Mini, a more efficient and cost-effective version of their flagship language model. 
            The new model promises to deliver near GPT-4 performance at a fraction of the cost, making advanced AI capabilities accessible to a broader range of applications.
            
            Key features include:
            - 70% lower cost compared to GPT-4
            - Faster response times with optimized architecture
            - Support for 100+ languages
            - Enhanced reasoning capabilities
            
            Industry experts predict this release will accelerate AI adoption across various sectors, from healthcare to education.
            The model is immediately available through OpenAI's API, with special pricing tiers for startups and educational institutions.""",
            author="Sarah Chen",
            publication_date=datetime(2024, 1, 15),
            source="AI Weekly",
            category="Technology",
            url="https://aiweekly.com/gpt4o-mini-release",
            tags=["AI", "OpenAI", "GPT-4O", "Language Models"]
        ),
        NewsArticle(
            article_id="2",
            title="Global Climate Summit: Nations Pledge Net-Zero by 2050",
            content="""At the Global Climate Summit in Geneva, over 150 nations have committed to achieving net-zero carbon emissions by 2050.
            This historic agreement represents the most comprehensive climate action plan to date.
            
            Major commitments include:
            - $1 trillion annual investment in renewable energy
            - Phase-out of coal power by 2040
            - Protection of 30% of global forests
            - Support fund for developing nations
            
            Environmental groups have praised the agreement while emphasizing the need for immediate action.
            The summit also saw major corporations pledging to align their operations with climate goals.""",
            author="Michael Torres",
            publication_date=datetime(2024, 1, 10),
            source="Environmental Times",
            category="Environment",
            url="https://envtimes.com/climate-summit-2024",
            tags=["Climate Change", "Net Zero", "Renewable Energy", "Global Summit"]
        ),
        NewsArticle(
            article_id="3",
            title="Tech Stocks Surge as AI Revolution Drives Market Growth",
            content="""Technology stocks reached new heights today, with the NASDAQ climbing 3.5% driven by strong earnings from AI-focused companies.
            
            Market highlights:
            - NVIDIA up 8% on record datacenter revenue
            - Microsoft gains 5% on Azure AI growth
            - Emerging AI startups see 200% average valuation increase
            
            Analysts attribute the surge to increasing enterprise adoption of AI technologies and growing investor confidence in the AI sector.
            The broader market also benefited, with the S&P 500 closing up 2.1%.""",
            author="Jennifer Liu",
            publication_date=datetime(2024, 1, 20),
            source="Financial Daily",
            category="Finance",
            url="https://findaily.com/ai-stocks-surge",
            tags=["Stock Market", "AI", "Technology", "Investment"]
        ),
        NewsArticle(
            article_id="4",
            title="Breakthrough in Quantum Computing: 1000-Qubit Processor Achieved",
            content="""Researchers at Quantum Dynamics Inc. have successfully demonstrated a 1000-qubit quantum processor, marking a significant milestone in quantum computing.
            
            The breakthrough enables:
            - Complex molecular simulations for drug discovery
            - Advanced cryptographic applications
            - Optimization problems at unprecedented scale
            
            The team used a novel error correction technique that maintains quantum coherence for extended periods.
            This development brings practical quantum computing applications closer to reality, with potential impacts across industries from pharmaceuticals to finance.""",
            author="Dr. Robert Kim",
            publication_date=datetime(2024, 1, 18),
            source="Science Today",
            category="Science",
            url="https://scitoday.com/quantum-breakthrough",
            tags=["Quantum Computing", "Technology", "Research", "Innovation"]
        ),
        NewsArticle(
            article_id="5",
            title="EU Proposes Comprehensive AI Regulation Framework",
            content="""The European Union has unveiled a detailed framework for regulating artificial intelligence systems, setting global precedent for AI governance.
            
            Key provisions include:
            - Risk-based classification system for AI applications
            - Mandatory transparency requirements for AI systems
            - Strict controls on biometric identification
            - Heavy penalties for non-compliance (up to 6% of global revenue)
            
            Tech companies have expressed concerns about innovation impacts, while privacy advocates praise the consumer protections.
            The regulations are expected to influence AI policy worldwide, with other regions considering similar frameworks.""",
            author="Emma Martinez",
            publication_date=datetime(2024, 1, 12),
            source="EU Observer",
            category="Technology",
            url="https://euobserver.com/ai-regulation",
            tags=["AI", "Regulation", "EU", "Privacy", "Technology Policy"]
        )
    ]
    return articles

def demonstrate_rag_system():
    """Comprehensive demonstration of the News RAG system"""
    print("Initializing News RAG System with LangChain + FAISS + GPT-4O-mini...")
    print("="*80)
    
    # Initialize the system
    try:
        rag_system = NewsRAGSystem(
            model_name="gpt-4o-mini",
            persist_directory="./news_rag_demo"
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key in the environment or pass it to the constructor.")
        return
    
    # Add sample articles
    print("\nAdding sample articles to the system...")
    articles = create_sample_articles()
    article_ids = rag_system.add_articles_batch(articles)
    print(f"Successfully added {len(article_ids)} articles")
    
    # Display statistics
    stats = rag_system.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"- Total articles: {stats['total_articles']}")
    print(f"- Categories: {stats['categories']}")
    print(f"- Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate search functionality
    print("SEARCH DEMONSTRATION")
    print("-"*40)
    
    search_queries = [
        "artificial intelligence breakthroughs",
        "climate change initiatives",
        "quantum computing advances"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = rag_system.search_articles(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.article.title}")
            print(f"   Source: {result.article.source} | Date: {result.article.publication_date.strftime('%Y-%m-%d')}")
            print(f"   Relevance: {result.relevance_score:.3f}")
            print(f"   Snippet: {result.snippet}")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate question answering
    print("QUESTION ANSWERING DEMONSTRATION")
    print("-"*40)
    
    questions = [
        "What are the key features of GPT-4O Mini?",
        "What are the main climate commitments made at the Global Summit?",
        "How many qubits does the new quantum processor have?",
        "What percentage can EU fine companies for AI regulation violations?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag_system.ask_question(question, k=3)
        
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {', '.join(response.context_used)}")
        print(f"Tokens used: {response.tokens_used} (Cost: ${response.cost:.4f})")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate conversational QA
    print("CONVERSATIONAL QA DEMONSTRATION")
    print("-"*40)
    
    print("\nStarting a conversation about AI developments...")
    
    conversation = [
        "What recent AI developments have been announced?",
        "Can you tell me more about the cost savings?",  # Follow-up
        "How does this compare to the EU regulations?"    # Another follow-up
    ]
    
    for question in conversation:
        print(f"\nQ: {question}")
        response = rag_system.ask_follow_up(question)
        print(f"A: {response.answer}")
    
    # Clear memory for next conversation
    rag_system.clear_memory()
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate filtered search
    print("FILTERED SEARCH DEMONSTRATION")
    print("-"*40)
    
    print("\nSearching for Technology articles only:")
    tech_results = rag_system.search_articles(
        "innovation", 
        k=3, 
        category_filter="Technology"
    )
    
    for result in tech_results:
        print(f"- {result.article.title}")
        print(f"  Category: {result.article.category} | Score: {result.relevance_score:.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate article summarization
    print("ARTICLE SUMMARIZATION DEMONSTRATION")
    print("-"*40)
    
    print("\nSummarizing Technology articles:")
    summary = rag_system.summarize_articles(category="Technology", max_articles=3)
    print(summary)
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate trending topics
    print("TRENDING TOPICS ANALYSIS")
    print("-"*40)
    
    print("\nAnalyzing trending topics across all articles:")
    trending = rag_system.get_trending_topics(top_k=3)
    
    if isinstance(trending, list) and trending:
        for i, topic in enumerate(trending, 1):
            if isinstance(topic, dict):
                print(f"\n{i}. {topic.get('topic', 'Unknown Topic')}")
                print(f"   {topic.get('description', 'No description available')}")
    
    return rag_system

# Advanced usage examples
def advanced_examples():
    """Show advanced usage patterns"""
    print("\nADVANCED USAGE EXAMPLES")
    print("="*80)
    
    # Initialize system
    rag_system = NewsRAGSystem(persist_directory="./news_rag_advanced")
    
    # Example 1: Date-filtered search
    print("\n1. Date-Filtered Search")
    print("-"*40)
    
    from datetime import timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    recent_results = rag_system.search_articles(
        "technology",
        k=5,
        date_filter=(start_date, end_date)
    )
    
    print(f"Found {len(recent_results)} recent technology articles")
    
    # Example 2: Multi-turn conversation with context
    print("\n2. Multi-turn Conversation")
    print("-"*40)
    
    # First, ask about a topic
    response1 = rag_system.ask_question(
        "What are the latest developments in AI?",
        use_memory=True
    )
    print(f"Initial question response: {response1.answer[:200]}...")
    
    # Follow up with related question
    response2 = rag_system.ask_follow_up(
        "What are the potential risks mentioned?"
    )
    print(f"Follow-up response: {response2.answer[:200]}...")
    
    # Example 3: Batch processing
    print("\n3. Batch Article Processing")
    print("-"*40)
    
    # Create multiple articles
    batch_articles = [
        NewsArticle(
            article_id=f"batch_{i}",
            title=f"Article {i}: Latest Updates",
            content=f"This is the content of article {i} with various information...",
            author="Batch Author",
            publication_date=datetime.now(),
            source="Batch Source",
            category="Technology",
            url=f"https://example.com/article{i}"
        )
        for i in range(5)
    ]
    
    # Add them efficiently
    batch_ids = rag_system.add_articles_batch(batch_articles)
    print(f"Added {len(batch_ids)} articles in batch")
    
    # Example 4: Custom analysis
    print("\n4. Custom Analysis Query")
    print("-"*40)
    
    analysis_response = rag_system.ask_question(
        "Analyze the relationship between AI developments and market performance based on the articles",
        k=5
    )
    print(f"Analysis: {analysis_response.answer[:300]}...")
    
    return rag_system

if __name__ == "__main__":
    # Run the main demonstration
    rag_system = demonstrate_rag_system()
    
    # Uncomment to run advanced examples
    # print("\n" + "="*80)
    # advanced_examples()
