import logging
import re
from modules.preprocessing import sentence_segmentation, preprocess_text

logger = logging.getLogger(__name__)

# Lazy loader for BERT QA pipeline
_qa_pipeline = None

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading DistilBERT SQuAD QA pipeline...")
            _qa_pipeline = pipeline(
                "question-answering", 
                model="distilbert-base-cased-distilled-squad", 
                device=-1
            )
            logger.info("DistilBERT SQuAD pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT SQuAD model: {e}")
            raise e
    return _qa_pipeline

def split_into_paragraphs(text):
    """Splits text into paragraphs or chunks of roughly 150-200 words."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # If there are no clear paragraphs, chunk by sentences
    if len(paragraphs) <= 1:
        sentences = sentence_segmentation(text)
        chunks = []
        current_chunk = []
        current_words = 0
        for sent in sentences:
            words = len(sent.split())
            if current_words + words > 150:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_words = words
            else:
                current_chunk.append(sent)
                current_words += words
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
        
    return paragraphs

def retrieve_relevant_context(text, question, top_n=1):
    """
    Finds the paragraph(s) most relevant to the question using TF-IDF style term matching.
    Helps bypass token length limits of BERT.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return text
        
    # Preprocess question
    q_prep = preprocess_text(question)
    q_words = set(q_prep["lemmatized"])
    
    if not q_words:
        # Fallback to simple split if preprocessing is empty
        q_words = set(question.lower().split())

    scored_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        p_prep = preprocess_text(paragraph)
        p_words = p_prep["lemmatized"]
        
        # Calculate term overlap score
        score = sum(1 for w in q_words if w in p_words)
        
        # Add a tiny length penalty to prevent extremely long paragraphs from dominating
        word_count = len(paragraph.split())
        normalized_score = score / (1.0 + 0.01 * word_count)
        
        scored_paragraphs.append((normalized_score, paragraph))
        
    # Sort by score descending
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Combine top N paragraphs
    best_contexts = [p[1] for p in scored_paragraphs[:top_n]]
    combined_context = "\n\n".join(best_contexts)
    
    return combined_context

def keyword_overlap_qa_fallback(text, question):
    """
    Sentence overlap fallback:
    Looks for the sentence in the text that has the highest keyword overlap with the question.
    """
    logger.info("Running QA keyword overlap fallback.")
    try:
        sentences = sentence_segmentation(text)
        if not sentences:
            return "No text available to answer the question."
            
        q_prep = preprocess_text(question)
        q_words = set(q_prep["lemmatized"])
        
        if not q_words:
            q_words = set(question.lower().split())
            
        best_sentence = ""
        max_overlap = -1
        
        for sentence in sentences:
            s_prep = preprocess_text(sentence)
            s_words = s_prep["lemmatized"]
            
            overlap = sum(1 for w in q_words if w in s_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence
                
        if max_overlap > 0:
            return best_sentence
        return "I could not find a clear answer in the text. Try rephrasing the question."
    except Exception as e:
        logger.error(f"QA fallback failed: {e}")
        return "Sorry, an error occurred while trying to process the question."

def answer_question(text, question):
    """
    Answers a question based on the document text.
    Uses TF-IDF retriever to find relevant context, then runs DistilBERT.
    Falls back to sentence keyword-overlap retriever if BERT fails or is not installed.
    """
    if not text or not text.strip() or not question or not question.strip():
        return "Please provide both text context and a question."
        
    try:
        qa_pipeline = get_qa_pipeline()
        
        # Retrieve the most relevant chunk of context
        context = retrieve_relevant_context(text, question, top_n=1)
        logger.info(f"Retrieved relevant context (length: {len(context)} chars). Running DistilBERT.")
        
        result = qa_pipeline(question=question, context=context)
        answer = result.get('answer', '')
        score = result.get('score', 0.0)
        
        # If confidence score is too low, we can fall back or return the answer
        logger.info(f"BERT Answer: '{answer}' (Confidence score: {score:.4f})")
        
        if answer.strip() and score > 0.05:
            # Capitalize first letter
            return answer[0].upper() + answer[1:]
            
    except Exception as e:
        logger.warning(f"BERT QA failed: {e}. Using sentence overlap fallback.")
        
    return keyword_overlap_qa_fallback(text, question)
