import numpy as np  #pip install numpy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from collections import defaultdict
import hashlib
from abc import ABC, abstractmethod

# Data structures
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
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "publication_date": self.publication_date.isoformat(),
            "source": self.source,
            "category": self.category,
            "url": self.url
        }

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

# Abstract base classes for modularity
class Embedder(ABC):
    """Abstract base class for text embedding"""
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        pass

class VectorStore(ABC):
    """Abstract base class for vector storage"""
    @abstractmethod
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[Dict[str, Any], float]]:
        pass

class LLM(ABC):
    """Abstract base class for language model"""
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        pass

# Concrete implementations
class SimpleEmbedder(Embedder):
    """Simple embedding implementation using TF-IDF-like approach"""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf_weights = {}
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = defaultdict(int)
        doc_counts = defaultdict(int)
        
        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_counts[token] += 1
            
            for token in self._tokenize(text):
                word_counts[token] += 1
        
        # Select top words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size]):
            self.vocabulary[word] = i
            # Calculate IDF weight
            self.idf_weights[word] = np.log(len(texts) / (doc_counts[word] + 1))
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        if not self.vocabulary:
            # Initialize with simple vocabulary if not built
            tokens = self._tokenize(text)
            for i, token in enumerate(tokens[:self.vocab_size]):
                self.vocabulary[token] = i
        
        # Create TF-IDF vector
        vector = np.zeros(self.vocab_size)
        tokens = self._tokenize(text)
        token_counts = defaultdict(int)
        
        for token in tokens:
            token_counts[token] += 1
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens)
                idf = self.idf_weights.get(token, 1.0)
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        self._build_vocabulary(texts)
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return np.array(embeddings)

class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store"""
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.index = {}
        
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """Add embedding to store"""
        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        # Create ID if not provided
        if 'id' not in metadata:
            metadata['id'] = f"doc_{idx}"
        
        self.index[metadata['id']] = idx
        return metadata['id']
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar embeddings"""
        if not self.embeddings:
            return []
        
        # Calculate cosine similarities
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return positive similarities
                results.append((self.metadata[idx], float(similarities[idx])))
        
        return results

class SimpleLLM(LLM):
    """Simple rule-based LLM simulator"""
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response based on prompt and context"""
        # Extract question type
        prompt_lower = prompt.lower()
        
        if context:
            # Extract key information from context
            sentences = context.split('.')
            relevant_sentences = []
            
            # Simple keyword matching
            question_keywords = set(re.findall(r'\b\w+\b', prompt_lower))
            
            for sentence in sentences:
                sentence_keywords = set(re.findall(r'\b\w+\b', sentence.lower()))
                if len(question_keywords & sentence_keywords) > 2:
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                # Generate answer based on question type
                if any(word in prompt_lower for word in ['who', 'whom']):
                    # Look for person names (capitalized words)
                    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ' '.join(relevant_sentences))
                    if names:
                        return f"Based on the articles, {names[0]} is mentioned in relation to your question."
                
                elif any(word in prompt_lower for word in ['when', 'what time']):
                    # Look for dates
                    dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' '.join(relevant_sentences))
                    if dates:
                        return f"The relevant time period appears to be {dates[0]}."
                
                elif any(word in prompt_lower for word in ['where', 'location']):
                    # Look for locations (simplified)
                    return f"The location mentioned in the context is related to the news events described."
                
                elif any(word in prompt_lower for word in ['why', 'reason']):
                    return f"The reason appears to be: {relevant_sentences[0]}" if relevant_sentences else "The reason is explained in the context provided."
                
                elif any(word in prompt_lower for word in ['how many', 'number']):
                    # Look for numbers
                    numbers = re.findall(r'\b\d+\b', ' '.join(relevant_sentences))
                    if numbers:
                        return f"The number mentioned is {numbers[0]}."
                
                else:
                    # Default response using relevant sentences
                    return f"Based on the articles: {' '.join(relevant_sentences[:2])}"
        
        return "I need more context to answer this question accurately."

class NewsRAGSystem:
    """Main RAG system for news search and question answering"""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore, llm: LLM):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.articles = {}  # article_id -> NewsArticle
        
    def add_article(self, article: NewsArticle) -> str:
        """Add a news article to the system"""
        # Generate article ID if not provided
        if not article.article_id:
            content_hash = hashlib.md5(article.content.encode()).hexdigest()[:8]
            article.article_id = f"article_{content_hash}"
        
        # Create embedding for the article
        combined_text = f"{article.title} {article.content}"
        embedding = self.embedder.embed(combined_text)
        article.embedding = embedding
        
        # Store in vector database
        metadata = article.to_dict()
        self.vector_store.add(embedding, metadata)
        
        # Store article object
        self.articles[article.article_id] = article
        
        return article.article_id
    
    def search_articles(self, query: str, k: int = 5, 
                       category_filter: Optional[str] = None,
                       date_filter: Optional[Tuple[datetime, datetime]] = None) -> List[SearchResult]:
        """Search for relevant news articles"""
        # Embed the query
        query_embedding = self.embedder.embed(query)
        
        # Perform vector search
        results = self.vector_store.search(query_embedding, k=k*2)  # Get more results for filtering
        
        search_results = []
        for metadata, score in results:
            article_id = metadata['article_id']
            if article_id not in self.articles:
                continue
                
            article = self.articles[article_id]
            
            # Apply filters
            if category_filter and article.category != category_filter:
                continue
                
            if date_filter:
                start_date, end_date = date_filter
                if not (start_date <= article.publication_date <= end_date):
                    continue
            
            # Generate snippet
            snippet = self._generate_snippet(article.content, query)
            
            search_results.append(SearchResult(
                article=article,
                relevance_score=score,
                snippet=snippet
            ))
            
            if len(search_results) >= k:
                break
        
        return search_results
    
    def ask_question(self, question: str, k: int = 3) -> QAResponse:
        """Answer a question using RAG"""
        # Search for relevant articles
        search_results = self.search_articles(question, k=k)
        
        if not search_results:
            return QAResponse(
                answer="No relevant articles found to answer this question.",
                source_articles=[],
                confidence_score=0.0,
                context_used=[]
            )
        
        # Prepare context from top articles
        context_parts = []
        source_articles = []
        
        for result in search_results:
            article = result.article
            context_part = f"Title: {article.title}\nContent: {article.content[:500]}..."
            context_parts.append(context_part)
            source_articles.append(article)
        
        combined_context = "\n\n".join(context_parts)
        
        # Generate prompt for LLM
        prompt = f"""Answer the following question based on the provided news articles:

Question: {question}

Context from news articles:
{combined_context}

Please provide a comprehensive answer based on the information in these articles."""
        
        # Generate answer
        answer = self.llm.generate(prompt, context=combined_context)
        
        # Calculate confidence score based on relevance scores
        avg_relevance = np.mean([r.relevance_score for r in search_results])
        confidence_score = min(avg_relevance * 1.5, 1.0)  # Scale and cap at 1.0
        
        return QAResponse(
            answer=answer,
            source_articles=source_articles,
            confidence_score=confidence_score,
            context_used=[f"{a.title} ({a.source})" for a in source_articles]
        )
    
    def _generate_snippet(self, content: str, query: str, snippet_length: int = 150) -> str:
        """Generate a relevant snippet from content"""
        # Simple approach: find sentences containing query terms
        query_terms = set(query.lower().split())
        sentences = content.split('.')
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        # Return snippet around best sentence
        if len(best_sentence) > snippet_length:
            return best_sentence[:snippet_length] + "..."
        return best_sentence
    
    def get_trending_topics(self, time_window: Optional[Tuple[datetime, datetime]] = None,
                           top_k: int = 5) -> List[Tuple[str, int]]:
        """Get trending topics from recent articles"""
        # Simple implementation: count category frequencies
        category_counts = defaultdict(int)
        
        for article in self.articles.values():
            if time_window:
                start_date, end_date = time_window
                if not (start_date <= article.publication_date <= end_date):
                    continue
            
            category_counts[article.category] += 1
        
        # Sort by count
        trending = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return trending[:top_k]

# Example usage and demonstration
def create_sample_articles() -> List[NewsArticle]:
    """Create sample news articles for demonstration"""
    articles = [
        NewsArticle(
            article_id="1",
            title="Tech Giant Announces Revolutionary AI Breakthrough",
            content="Leading technology company XYZ Corp unveiled their latest artificial intelligence system today, claiming it can process natural language with unprecedented accuracy. The system, named AlphaAI, demonstrated capabilities in real-time translation, complex reasoning, and creative writing. CEO John Smith stated that this breakthrough could revolutionize how businesses interact with AI technology. The announcement caused XYZ Corp's stock to surge by 15% in after-hours trading.",
            author="Jane Doe",
            publication_date=datetime(2024, 1, 15),
            source="Tech Daily",
            category="Technology",
            url="https://techdaily.com/ai-breakthrough"
        ),
        NewsArticle(
            article_id="2",
            title="Global Climate Summit Reaches Historic Agreement",
            content="World leaders at the Global Climate Summit in Paris have reached a groundbreaking agreement to reduce carbon emissions by 50% by 2030. The agreement, signed by 195 countries, includes specific targets for renewable energy adoption and forest conservation. Environmental groups praised the deal as a crucial step in combating climate change. However, some critics argue that the targets are still not ambitious enough to limit global warming to 1.5 degrees Celsius.",
            author="Michael Johnson",
            publication_date=datetime(2024, 1, 10),
            source="Environmental Times",
            category="Environment",
            url="https://envtimes.com/climate-summit"
        ),
        NewsArticle(
            article_id="3",
            title="Stock Market Hits Record High Amid Economic Recovery",
            content="The stock market reached an all-time high today, with the S&P 500 closing above 5,000 points for the first time in history. Analysts attribute the surge to strong corporate earnings reports and optimistic economic forecasts. The technology sector led the gains, with major tech companies seeing double-digit growth. Federal Reserve Chair Sarah Williams suggested that interest rates would remain stable, further boosting investor confidence.",
            author="Robert Chen",
            publication_date=datetime(2024, 1, 20),
            source="Financial News",
            category="Finance",
            url="https://finnews.com/record-high"
        ),
        NewsArticle(
            article_id="4",
            title="New AI Regulations Proposed by European Union",
            content="The European Union has proposed comprehensive regulations for artificial intelligence systems, aiming to ensure ethical use and protect citizen privacy. The proposed AI Act would classify AI systems based on risk levels and impose strict requirements on high-risk applications. Tech companies have expressed concerns about the potential impact on innovation, while privacy advocates welcome the move. The regulations would be the first of their kind globally.",
            author="Emma Wilson",
            publication_date=datetime(2024, 1, 18),
            source="EU Observer",
            category="Technology",
            url="https://euobserver.com/ai-regulations"
        ),
        NewsArticle(
            article_id="5",
            title="Breakthrough in Renewable Energy Storage Technology",
            content="Scientists at the National Energy Research Lab have developed a new battery technology that could store renewable energy for up to 30 days without significant loss. The breakthrough addresses one of the biggest challenges in renewable energy adoption - the intermittent nature of solar and wind power. The new batteries use a novel lithium-sulfur composition that is both more efficient and environmentally friendly than current technologies. Commercial production could begin within two years.",
            author="Dr. Lisa Park",
            publication_date=datetime(2024, 1, 12),
            source="Science Today",
            category="Science",
            url="https://sciencetoday.com/battery-breakthrough"
        )
    ]
    return articles

def demonstrate_rag_system():
    """Demonstrate the News RAG system"""
    # Initialize components
    embedder = SimpleEmbedder(vocab_size=1000)
    vector_store = SimpleVectorStore()
    llm = SimpleLLM()
    
    # Create RAG system
    rag_system = NewsRAGSystem(embedder, vector_store, llm)
    
    # Add sample articles
    articles = create_sample_articles()
    
    # First, collect all texts for vocabulary building
    all_texts = [f"{a.title} {a.content}" for a in articles]
    embedder.embed_batch(all_texts)
    
    # Now add articles to the system
    print("Adding articles to the system...")
    for article in articles:
        article_id = rag_system.add_article(article)
        print(f"Added: {article.title} (ID: {article_id})")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate search functionality
    print("SEARCH DEMONSTRATION")
    print("-"*40)
    
    search_queries = [
        "artificial intelligence breakthrough",
        "climate change agreement",
        "stock market performance"
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
        "What AI breakthrough was announced recently?",
        "What are the carbon emission reduction targets?",
        "Who is the CEO of XYZ Corp?",
        "What new technology addresses renewable energy storage?",
        "What regulations are being proposed for AI?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag_system.ask_question(question)
        
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {', '.join(response.context_used)}")
    
    print("\n" + "="*80 + "\n")
    
    # Demonstrate filtered search
    print("FILTERED SEARCH DEMONSTRATION")
    print("-"*40)
    
    print("\nSearching for Technology articles only:")
    tech_results = rag_system.search_articles("innovation", k=3, category_filter="Technology")
    
    for result in tech_results:
        print(f"- {result.article.title} (Score: {result.relevance_score:.3f})")
    
    # Demonstrate trending topics
    print("\n\nTRENDING TOPICS")
    print("-"*40)
    
    trending = rag_system.get_trending_topics(top_k=3)
    for topic, count in trending:
        print(f"- {topic}: {count} articles")
    
    return rag_system

if __name__ == "__main__":
    # Run the demonstration
    rag_system = demonstrate_rag_system()
