"""
Custom filters and utilities for FIN-bench tasks.

Handles answer extraction from both thinking models (gpt-oss with harmony format)
and non-thinking models (gemma3, llama with verbose chat responses).
"""

import re
import logging
import difflib
from lm_eval.filters.extraction import Filter

eval_logger = logging.getLogger("lm-eval")


def doc_to_text_with_instructions(doc):
    """
    Add formatting instructions to the prompt to guide models to respond correctly.
    
    This is critical for machine-readable responses. Without clear instructions,
    chat models tend to give verbose explanations instead of direct answers.
    
    Also ensures that answer choices are always shown in the prompt, even when
    the dataset's 'inputs' field doesn't include them.
    """
    inputs = doc.get("inputs", "")
    choices = doc.get("multiple_choice_targets", [])
    
    # Check if choices are already in the prompt (some datasets include them, some don't)
    choices_in_prompt = "vaihtoehto:" in inputs.lower() or any(choice in inputs for choice in choices)
    
    # Add instructions emphasizing direct word answers from the given options
    instructions = (
        "TÄRKEÄÄ: Vastaa lyhyesti ja selkeästi. "
        "Valitse täsmälleen yksi vastausvaihtoehto annetuista. "
        "Älä anna pitkiä selityksiä. "
        "Vastauksesi luetaan automaattisesti.\n\n"
    )
    
    # If choices aren't in the prompt already, add them
    if choices and not choices_in_prompt:
        choices_text = "\n".join(f"  vaihtoehto: {choice}" for choice in choices)
        # Insert choices before the final "Vastaus:" or "Assistentti:" line
        if "\nVastaus:" in inputs:
            inputs = inputs.replace("\nVastaus:", f"\n{choices_text}\nVastaus:")
        elif "\nAssistentti:" in inputs:
            inputs = inputs.replace("\nAssistentti:", f"\n{choices_text}\nAssistentti:")
        else:
            # Fallback: append choices to end
            inputs = inputs.rstrip() + "\n" + choices_text + "\n"
    
    return instructions + inputs


class FINBenchAnswerFilter(Filter):
    """
    Extracts and validates answers from FIN-bench model responses.
    
    Key feature: Separates VALID vs INVALID responses from CORRECT vs WRONG answers.
    - Returns the actual choice found (correct or wrong) for proper scoring
    - Returns empty string only for invalid/unparseable responses
    """
    
    def __init__(self) -> None:
        """Initialize the filter."""
        self.stats = {
            'total': 0,
            'correct': 0,
            'wrong': 0,
            'invalid': 0,
            'empty_response': 0,
            'no_target': 0
        }
    
    def apply(self, resps, docs):
        """
        Apply filter to extract and validate answers.
        
        Args:
            resps: List of response lists (one per document)
            docs: List of documents (with targets and multiple_choice info)
            
        Returns:
            List of extracted answers (matching any valid choice for exact_match)
        """
        
        def get_correct_letter(doc):
            """Get the correct answer letter (A, B, C, D, E) from multiple_choice_scores."""
            if "multiple_choice_scores" not in doc:
                return None
            
            scores = doc["multiple_choice_scores"]
            if not scores or sum(scores) != 1:
                return None
            
            # Find index of correct answer (score = 1)
            correct_idx = scores.index(1)
            # Convert to letter: 0->A, 1->B, etc.
            return chr(ord('A') + correct_idx)
        
        def extract_and_validate(resp: str, doc) -> str:
            """
            Extract answer from response and validate it.
            
            Strategy:
            1. Try to find ANY valid answer from the choices (correct or wrong)
            2. Return the found answer for exact_match comparison
            3. If no valid answer found, return empty string (invalid format)
            
            This separates:
            - Correct answers: returned answer == target → exact_match = 1.0
            - Wrong answers: returned answer != target → exact_match = 0.0  
            - Invalid format: return "" → exact_match = 0.0, but logged as ERROR
            
            Args:
                resp: Model response string
                doc: Document with targets and multiple_choice info
                
            Returns:
                Found answer string (any valid choice) or empty string (invalid)
            """
            if not resp or not resp.strip():
                eval_logger.error("Empty response received")
                self.stats['empty_response'] += 1
                self.stats['invalid'] += 1
                return ""
            
            # Get expected answer
            target = doc.get("targets", [""])[0]
            if not target:
                eval_logger.error("No target answer in document")
                self.stats['no_target'] += 1
                self.stats['invalid'] += 1
                return ""
            
            # Get correct letter
            correct_letter = get_correct_letter(doc)
            
            # Get all choice words for validation
            all_choices = doc.get("multiple_choice_targets", [])
            
            # Extract first line (before any newline)
            first_line = resp.split('\n')[0].strip()
            first_line_lower = first_line.lower()
            
            eval_logger.debug(f"Full response: '{resp[:200]}'")
            eval_logger.debug(f"First line: '{first_line[:80]}'")
            eval_logger.debug(f"Target: '{target}', Letter: {correct_letter}")
            eval_logger.debug(f"All choices: {all_choices}")
            
            def normalize_text(text):
                """Normalize text for robust matching: lowercase, strip punctuation, normalize whitespace."""
                # Lowercase
                text = text.lower()
                # Strip leading/trailing whitespace and punctuation
                text = text.strip().strip('.,!?;:')
                # Normalize internal whitespace (multiple spaces -> single space)
                text = ' '.join(text.split())
                return text
            
            def strip_special_chars(text):
                """Remove all punctuation and special characters, keep only alphanumeric and spaces."""
                return re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
            
            # Prepare normalized versions
            first_line_normalized = normalize_text(first_line)
            first_line_stripped = strip_special_chars(first_line_normalized)
            
            # Strategy 1: Exact substring match (case-insensitive, with lowercase choices in response)
            for choice in all_choices:
                if choice and choice.lower() in first_line_lower:
                    self.stats['total'] += 1
                    if choice.lower() == target.lower():
                        eval_logger.debug(f"✓ Found CORRECT answer (substring match): '{choice}'")
                        self.stats['correct'] += 1
                    else:
                        eval_logger.debug(f"✗ Found WRONG answer (substring match): '{choice}' (expected: '{target}')")
                        self.stats['wrong'] += 1
                    return choice  # Return what we found (correct or wrong)
            
            # Strategy 2: Normalized exact match (case-insensitive, stripped punctuation)
            for choice in all_choices:
                if choice:
                    choice_normalized = normalize_text(choice)
                    if first_line_normalized == choice_normalized:
                        self.stats['total'] += 1
                        if choice.lower() == target.lower():
                            eval_logger.debug(f"✓ Found CORRECT answer (normalized exact): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            eval_logger.debug(f"✗ Found WRONG answer (normalized exact): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        return choice
            
            # Strategy 3: Stripped special characters match (alphanumeric only)
            for choice in all_choices:
                if choice:
                    choice_stripped = strip_special_chars(normalize_text(choice))
                    if first_line_stripped == choice_stripped:
                        self.stats['total'] += 1
                        if choice.lower() == target.lower():
                            eval_logger.debug(f"✓ Found CORRECT answer (stripped special chars): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            eval_logger.debug(f"✗ Found WRONG answer (stripped special chars): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        return choice
            
            # Strategy 4: Fuzzy matching with 80% similarity threshold
            # Handles whitespace differences, minor typos, etc.
            for choice in all_choices:
                if choice:
                    choice_normalized = normalize_text(choice)
                    similarity = difflib.SequenceMatcher(
                        None, 
                        first_line_normalized, 
                        choice_normalized
                    ).ratio()
                    
                    if similarity >= 0.80:
                        self.stats['total'] += 1
                        if choice.lower() == target.lower():
                            eval_logger.debug(f"✓ Found CORRECT answer (fuzzy {similarity:.2%}): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            eval_logger.debug(f"✗ Found WRONG answer (fuzzy {similarity:.2%}): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        return choice
            
            # Strategy 2: Check for bold/emphasized answers (e.g., **word**)
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', first_line)
            for bold_word in bold_matches:
                # Check against all choices (with inflection tolerance)
                for choice in all_choices:
                    if choice and len(choice) >= 4:
                        stem = choice.lower()[:max(4, len(choice)-3)]
                        if bold_word.lower() == choice.lower() or bold_word.lower().startswith(stem):
                            self.stats['total'] += 1
                            if choice.lower() == target.lower():
                                eval_logger.debug(f"✓ Found CORRECT answer in bold: '{bold_word}' → '{choice}'")
                                self.stats['correct'] += 1
                            else:
                                eval_logger.debug(f"✗ Found WRONG answer in bold: '{bold_word}' → '{choice}' (expected: '{target}')")
                                self.stats['wrong'] += 1
                            return choice
            
            # Strategy 3: Check for Finnish inflections of ANY choice
            for choice in all_choices:
                if choice and len(choice) >= 4:
                    stem = choice.lower()[:max(4, len(choice)-3)]
                    if re.search(r'\b' + re.escape(stem), first_line_lower):
                        self.stats['total'] += 1
                        if choice.lower() == target.lower():
                            eval_logger.debug(f"✓ Found CORRECT answer stem: '{stem}' → '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            eval_logger.debug(f"✗ Found WRONG answer stem: '{stem}' → '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        return choice
            
            # If we get here, couldn't parse the response into any valid choice
            self.stats['total'] += 1
            self.stats['invalid'] += 1
            eval_logger.error(
                f"✗✗ INVALID FORMAT - could not parse response!\n"
                f"   First line: '{first_line}'\n"
                f"   Full response: '{resp[:300]}...'\n"
                f"   Expected: answer word from choices\n"
                f"   Valid choices: {all_choices}\n"
                f"   This response cannot be evaluated"
            )
            
            return ""  # Invalid format
        
        # Apply extraction to all responses
        filtered_resps = []
        for resp_list, doc in zip(resps, docs):
            filtered_list = []
            for resp in resp_list:
                filtered_list.append(extract_and_validate(resp, doc))
            filtered_resps.append(filtered_list)
        
        # Log summary statistics
        if self.stats['total'] > 0:
            accuracy = self.stats['correct'] / self.stats['total'] * 100
            invalid_rate = self.stats['invalid'] / self.stats['total'] * 100
            eval_logger.info(
                f"\n{'='*70}\n"
                f"FINBenchAnswerFilter Summary:\n"
                f"  Total responses: {self.stats['total']}\n"
                f"  Correct: {self.stats['correct']} ({accuracy:.1f}%)\n"
                f"  Wrong: {self.stats['wrong']} ({self.stats['wrong']/self.stats['total']*100:.1f}%)\n"
                f"  Invalid format: {self.stats['invalid']} ({invalid_rate:.1f}%)\n"
                f"    - Empty responses: {self.stats['empty_response']}\n"
                f"    - No target: {self.stats['no_target']}\n"
                f"    - Unparseable: {self.stats['invalid'] - self.stats['empty_response'] - self.stats['no_target']}\n"
                f"{'='*70}"
            )
        
        return filtered_resps
