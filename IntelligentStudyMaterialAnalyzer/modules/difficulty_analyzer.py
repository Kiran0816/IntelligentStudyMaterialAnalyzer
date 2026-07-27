import math
import logging
from modules.preprocessing import sentence_segmentation, preprocess_text

logger = logging.getLogger(__name__)

def analyze_difficulty(text):
    """
    Analyzes reading difficulty based on:
    - Average sentence length (number of words per sentence)
    - Vocabulary complexity (average word length and type-token ratio)
    - Returns a dictionary with:
      - difficulty_level: "Easy", "Medium", "Hard"
      - sentence_count: int
      - word_count: int
      - estimated_study_time: int (minutes)
    """
    if not text or not text.strip():
        return {
            "difficulty_level": "Easy",
            "sentence_count": 0,
            "word_count": 0,
            "estimated_study_time": 0
        }

    try:
        # Segment sentences
        sentences = sentence_segmentation(text)
        sentence_count = len(sentences)
        if sentence_count == 0:
            sentence_count = 1
            
        # Get preprocessed tokens
        prep = preprocess_text(text)
        tokens = prep["tokens"]
        word_count = len(tokens)
        
        if word_count == 0:
            return {
                "difficulty_level": "Easy",
                "sentence_count": sentence_count,
                "word_count": 0,
                "estimated_study_time": 0
            }
            
        # 1. Average Sentence Length (ASL)
        avg_sentence_length = word_count / sentence_count
        
        # 2. Type-Token Ratio (TTR) - vocabulary diversity
        unique_words = set(tokens)
        type_token_ratio = len(unique_words) / word_count
        
        # 3. Average Word Length (AWL)
        avg_word_length = sum(len(w) for w in tokens) / word_count if word_count > 0 else 0
        
        # Heuristic scoring system
        # ASL: > 20 is hard, < 12 is easy
        # TTR: > 0.6 is hard, < 0.35 is easy
        # AWL: > 5.5 is hard, < 4.5 is easy
        
        score = 0
        
        # Sentence length impact
        if avg_sentence_length > 20:
            score += 2
        elif avg_sentence_length > 12:
            score += 1
            
        # TTR impact (penalize short texts as TTR is naturally higher, normalize if necessary)
        if word_count > 100:
            if type_token_ratio > 0.55:
                score += 2
            elif type_token_ratio > 0.4:
                score += 1
        else:
            # For short texts, TTR is less reliable, reduce impact
            if type_token_ratio > 0.7:
                score += 1
                
        # Word length impact
        if avg_word_length > 5.3:
            score += 2
        elif avg_word_length > 4.6:
            score += 1
            
        # Final classification
        if score >= 4:
            difficulty = "Hard"
        elif score >= 2:
            difficulty = "Medium"
        else:
            difficulty = "Easy"
            
        # Estimated study time (Reading speed: 200 WPM, add buffer for complex texts)
        base_reading_time = word_count / 200.0 # minutes
        
        # Apply difficulty multiplier (Harder texts take longer to comprehend)
        if difficulty == "Hard":
            multiplier = 1.5
        elif difficulty == "Medium":
            multiplier = 1.2
        else:
            multiplier = 1.0
            
        estimated_study_time = math.ceil(base_reading_time * multiplier)
        # Ensure at least 1 minute study time if there are words
        if estimated_study_time == 0 and word_count > 0:
            estimated_study_time = 1
            
        logger.info(f"Analysis complete: Words={word_count}, Sentences={sentence_count}, ASL={avg_sentence_length:.2f}, Difficulty={difficulty}, Study Time={estimated_study_time}m")
        
        return {
            "difficulty_level": difficulty,
            "sentence_count": sentence_count,
            "word_count": word_count,
            "estimated_study_time": estimated_study_time
        }
    except Exception as e:
        logger.error(f"Difficulty analysis failed: {e}")
        # Standard safety return
        words = len(text.split())
        return {
            "difficulty_level": "Medium",
            "sentence_count": max(1, len(text.split('.'))),
            "word_count": words,
            "estimated_study_time": max(1, math.ceil(words / 200))
        }
