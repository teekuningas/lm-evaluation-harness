"""
Custom filters and utilities for FINBench-v2 tasks.

Handles answer extraction from both thinking models (gpt-oss with reasoning_content)
and non-thinking models (gemma3, llama with verbose chat responses).

Key differences from v1:
- Schema-aware: Handles multiple dataset schemas (arc_c, belebele, goldenswag, etc.)
- Per-task choice extraction: Different tasks have different field names for choices
- Extensive debugging and validation to track migration progress

Author: Generated as part of FINBench-v2 migration
Date: 2026-01-21
See: FINBENCH_V2_TASK_QUIRKS.md, FINBENCH_MIGRATION_ANALYSIS.md
"""

import re
import logging
import difflib
from lm_eval.filters.extraction import Filter

eval_logger = logging.getLogger("lm-eval")


def doc_to_text_with_choices_injected(doc):
    """
    Runtime doc_to_text function for generate_until tasks.
    
    Ensures choices are visible and adds instructions for clean, machine-readable responses.
    This mirrors FINBench v1's approach (finbench/utils.py::doc_to_text_with_instructions)
    but adapted for v2's multiple dataset schemas.
    
    **Key features**:
    - Adds instructions discouraging verbose explanations
    - Shows choices with labels (0., 1., 2., etc.) 
    - Instructs model to answer with the choice TEXT (not the index)
    - Schema-aware: handles arc_c, belebele, goldenswag, scandisent, sib200
    
    Args:
        doc: Document dict with task-specific schema
        
    Returns:
        Complete prompt string with instructions and choices
    """
    prompt_parts = []
    
    # Add critical instructions (similar to FINBench v1)
    instructions = (
        "TÄRKEÄÄ: Vastaa lyhyesti ja selkeästi. "
        "Valitse täsmälleen yksi vastausvaihtoehto annetuista. "
        "Vastaa VAIN valintavaihtoehdolla, ÄLÄ numerolla. "
        "Älä anna pitkiä selityksiä. "
        "Vastauksesi luetaan automaattisesti.\n\n"
    )
    prompt_parts.append(instructions)
    
    # Add main question/context (schema-aware with priority order)
    # Check in specific order to avoid conflicts
    if 'flores_passage' in doc and 'question' in doc:
        # Belebele format: has both passage and question
        prompt_parts.append(f"{doc['flores_passage']}\n\nKysymys: {doc['question']}\n")
    elif 'query' in doc:
        # Goldenswag/Scandisent format (after process_docs)
        prompt_parts.append(f"{doc['query']}\n")
    elif 'question' in doc:
        # ARC format: just question
        prompt_parts.append(f"{doc['question']}\n")
    elif 'text' in doc:
        # SIB200 format
        prompt_parts.append(f"{doc['text']}\n")
    elif 'inputs' in doc:
        # Generic format or other tasks
        prompt_parts.append(f"{doc['inputs']}\n")
    else:
        # Fallback: try to find any text field
        for key in ['context', 'sentence', 'passage']:
            if key in doc:
                prompt_parts.append(f"{doc[key]}\n")
                break
    
    # Extract and add choices with numbering
    choices = get_choices_from_doc(doc, debug=False)
    
    if choices:
        prompt_parts.append("\nVaihtoehdot:\n")
        for i, choice in enumerate(choices):
            prompt_parts.append(f"{i}. {choice}\n")
        prompt_parts.append("\nVastaa valintavaihtoehdolla (EI numerolla):")
    else:
        # No choices found - just ask for answer
        prompt_parts.append("\nVastaus:")
    
    return "".join(prompt_parts)


def get_choices_from_doc(doc, debug=False):
    """
    Schema-aware choice extraction for FINBench-v2 tasks.
    
    Different v2 tasks use different schemas:
    - arc_c: doc["choices"]["text"] (list)
    - belebele: [doc["mc_answer1"], doc["mc_answer2"], doc["mc_answer3"], doc["mc_answer4"]]
    - goldenswag: doc["choices"] (already preprocessed)
    - scandisent: doc["choices"] (preprocessed: ["positiivinen", "negatiivinen"])
    - sib200: Hardcoded 7 topics (not per-doc)
    - truthfulqa mc1/mc2: doc["mc1_targets"]["choices"] or doc["mc2_targets"]["choices"]
    
    Args:
        doc: Document dict with task-specific schema
        debug: If True, log detection steps
        
    Returns:
        List of choice strings, or [] if not found
    """
    # Strategy 1: Pre-processed "choices" field (goldenswag, scandisent after process_docs)
    if "choices" in doc:
        choices = doc["choices"]
        if isinstance(choices, list):
            if debug:
                eval_logger.debug(f"[Schema Detection] Found doc['choices'] list: {choices}")
            return choices
        elif isinstance(choices, dict) and "text" in choices:
            # arc_c format: {"text": [...], "label": [...]}
            if debug:
                eval_logger.debug(f"[Schema Detection] Found doc['choices']['text']: {choices['text']}")
            return choices["text"]
    
    # Strategy 2: arc_c explicit format
    if "choices" in doc and isinstance(doc["choices"], dict):
        if "text" in doc["choices"]:
            if debug:
                eval_logger.debug(f"[Schema Detection] arc_c format: doc['choices']['text']")
            return doc["choices"]["text"]
    
    # Strategy 3: belebele format (individual mc_answer fields)
    if all(f"mc_answer{i}" in doc for i in [1, 2, 3, 4]):
        choices = [doc[f"mc_answer{i}"] for i in [1, 2, 3, 4]]
        if debug:
            eval_logger.debug(f"[Schema Detection] belebele format: mc_answer1-4: {choices}")
        return choices
    
    # Strategy 4: v1_multiprompt format (multiple_choice_targets)
    if "multiple_choice_targets" in doc:
        choices = doc["multiple_choice_targets"]
        if isinstance(choices, list):
            if debug:
                eval_logger.debug(f"[Schema Detection] v1_multiprompt format: multiple_choice_targets: {choices}")
            return choices
    
    # Strategy 5: truthfulqa mc1/mc2 format
    if "mc1_targets" in doc and isinstance(doc["mc1_targets"], dict):
        if "choices" in doc["mc1_targets"]:
            choices = doc["mc1_targets"]["choices"]
            if debug:
                eval_logger.debug(f"[Schema Detection] truthfulqa mc1: {choices}")
            return choices
    
    if "mc2_targets" in doc and isinstance(doc["mc2_targets"], dict):
        if "choices" in doc["mc2_targets"]:
            choices = doc["mc2_targets"]["choices"]
            if debug:
                eval_logger.debug(f"[Schema Detection] truthfulqa mc2: {choices}")
            return choices
    
    # Strategy 6: Fallback - check for common field names
    for field_name in ["options", "answers", "alternatives"]:
        if field_name in doc and isinstance(doc[field_name], list):
            if debug:
                eval_logger.debug(f"[Schema Detection] Fallback: doc['{field_name}']: {doc[field_name]}")
            return doc[field_name]
    
    # Not found
    if debug:
        eval_logger.warning(f"[Schema Detection] No choices found in doc. Keys: {list(doc.keys())[:10]}")
    return []


def get_target_from_doc(doc, choices=None, debug=False):
    """
    Schema-aware target extraction for FINBench-v2 tasks.
    
    Different v2 tasks use different target formats:
    - arc_c: doc["answerKey"] = "A" → need to find index in choices["label"]
    - belebele: doc["correct_answer_num"] = "1" (1-indexed) → convert to 0-indexed
    - goldenswag: doc["gold"] or doc["label"] = 0/1/2/3 (0-indexed)
    - scandisent: doc["gold"] = "positiivinen" or "negatiivinen"
    - truthfulqa: Always index 0 (first choice is correct)
    - sib200: doc["answer_idx"] = 0-6 (0-indexed)
    
    Args:
        doc: Document dict with task-specific schema
        choices: List of choices (needed for arc_c letter→index conversion)
        debug: If True, log detection steps
        
    Returns:
        Target answer string (the actual choice word), or None if not found
    """
    # Strategy 1: v1_multiprompt format (targets[0])
    if "targets" in doc and isinstance(doc["targets"], list) and len(doc["targets"]) > 0:
        target = doc["targets"][0]
        if debug:
            eval_logger.debug(f"[Target Detection] v1_multiprompt: targets[0] = '{target}'")
        return target
    
    # Strategy 2: Direct target word (scandisent)
    if "gold" in doc and isinstance(doc["gold"], str):
        # Could be a word or an index as string
        if doc["gold"].isdigit():
            idx = int(doc["gold"])
            if choices and 0 <= idx < len(choices):
                target = choices[idx]
                if debug:
                    eval_logger.debug(f"[Target Detection] doc['gold'] index {idx} → '{target}'")
                return target
        else:
            # It's a word
            if debug:
                eval_logger.debug(f"[Target Detection] doc['gold'] word: '{doc['gold']}'")
            return doc["gold"]
    
    # Strategy 3: Integer index (goldenswag, sib200)
    if "gold" in doc and isinstance(doc["gold"], int):
        idx = doc["gold"]
        if choices and 0 <= idx < len(choices):
            target = choices[idx]
            if debug:
                eval_logger.debug(f"[Target Detection] doc['gold'] int {idx} → '{target}'")
            return target
    
    if "label" in doc and isinstance(doc["label"], int):
        idx = doc["label"]
        if choices and 0 <= idx < len(choices):
            target = choices[idx]
            if debug:
                eval_logger.debug(f"[Target Detection] doc['label'] int {idx} → '{target}'")
            return target
    
    if "answer_idx" in doc:
        idx = int(doc["answer_idx"]) if isinstance(doc["answer_idx"], str) else doc["answer_idx"]
        if choices and 0 <= idx < len(choices):
            target = choices[idx]
            if debug:
                eval_logger.debug(f"[Target Detection] doc['answer_idx'] {idx} → '{target}'")
            return target
    
    # Strategy 3: arc_c letter format (answerKey = "A", "B", "C", "D")
    if "answerKey" in doc:
        answer_key = doc["answerKey"]
        if "choices" in doc and isinstance(doc["choices"], dict) and "label" in doc["choices"]:
            # Find index of answer_key in choices["label"]
            labels = doc["choices"]["label"]
            if answer_key in labels:
                idx = labels.index(answer_key)
                if choices and 0 <= idx < len(choices):
                    target = choices[idx]
                    if debug:
                        eval_logger.debug(f"[Target Detection] arc_c: answerKey '{answer_key}' → index {idx} → '{target}'")
                    return target
    
    # Strategy 4: belebele 1-indexed correct_answer_num
    if "correct_answer_num" in doc:
        # Convert from 1-indexed to 0-indexed
        answer_num = doc["correct_answer_num"]
        idx = int(answer_num) - 1
        if choices and 0 <= idx < len(choices):
            target = choices[idx]
            if debug:
                eval_logger.debug(f"[Target Detection] belebele: correct_answer_num {answer_num} → index {idx} → '{target}'")
            return target
    
    # Strategy 5: truthfulqa - always index 0
    if "mc1_targets" in doc or "mc2_targets" in doc:
        if choices and len(choices) > 0:
            target = choices[0]
            if debug:
                eval_logger.debug(f"[Target Detection] truthfulqa: index 0 → '{target}'")
            return target
    
    # Not found
    if debug:
        eval_logger.warning(f"[Target Detection] No target found in doc. Keys: {list(doc.keys())[:10]}")
    return None


class FINBenchV2AnswerFilter(Filter):
    """
    Schema-aware answer extraction filter for FINBench-v2 tasks.
    
    Adapted from FINBenchAnswerFilter (v1) with:
    - Schema detection for different v2 task formats
    - Extensive debugging for migration validation
    - Stats tracking per task type
    
    Key feature: Separates VALID vs INVALID responses from CORRECT vs WRONG answers.
    - Returns the actual choice found (correct or wrong) for proper scoring
    - Returns empty string only for invalid/unparseable responses
    """
    
    def __init__(self, verbose=False) -> None:
        """
        Initialize the filter.
        
        Args:
            verbose: If True, enable debug logging for every response (very verbose!)
        """
        self.verbose = verbose
        self.stats = {
            'total': 0,
            'correct': 0,
            'wrong': 0,
            'invalid': 0,
            'empty_response': 0,
            'no_target': 0,
            'no_choices': 0,
            'schema_detection_failed': 0
        }
        # Track which extraction strategy worked
        self.strategy_stats = {
            'substring_match': 0,
            'normalized_exact': 0,
            'stripped_special': 0,
            'fuzzy_80': 0,
            'bold_match': 0,
            'stem_match': 0,
            'failed': 0
        }
    
    def apply(self, resps, docs):
        """
        Apply filter to extract and validate answers.
        
        Args:
            resps: List of response lists (one per document)
            docs: List of documents (with task-specific schemas)
            
        Returns:
            List of extracted answers (matching any valid choice for exact_match)
        """
        
        def extract_and_validate(resp: str, doc) -> str:
            """
            Extract answer from response and validate it.
            
            Strategy (same as v1):
            1. Try to find ANY valid answer from the choices (correct or wrong)
            2. Return the found answer for exact_match comparison
            3. If no valid answer found, return empty string (invalid format)
            
            This separates:
            - Correct answers: returned answer == target → exact_match = 1.0
            - Wrong answers: returned answer != target → exact_match = 0.0  
            - Invalid format: return "" → exact_match = 0.0, but logged as ERROR
            
            Args:
                resp: Model response string
                doc: Document with task-specific schema
                
            Returns:
                Found answer string (any valid choice) or empty string (invalid)
            """
            self.stats['total'] += 1
            
            # Validation 1: Empty response
            if not resp or not resp.strip():
                eval_logger.error("[FINBenchV2] Empty response received")
                self.stats['empty_response'] += 1
                self.stats['invalid'] += 1
                self.strategy_stats['failed'] += 1
                return ""
            
            # Validation 2: Extract choices from document (schema-aware)
            all_choices = get_choices_from_doc(doc, debug=self.verbose)
            if not all_choices:
                eval_logger.error(f"[FINBenchV2] No choices found in document. Doc keys: {list(doc.keys())[:10]}")
                self.stats['no_choices'] += 1
                self.stats['invalid'] += 1
                self.strategy_stats['failed'] += 1
                return ""
            
            # Validation 3: Get target answer (schema-aware)
            target = get_target_from_doc(doc, choices=all_choices, debug=self.verbose)
            if not target:
                eval_logger.error(f"[FINBenchV2] No target answer found in document. Doc keys: {list(doc.keys())[:10]}")
                self.stats['no_target'] += 1
                self.stats['invalid'] += 1
                self.strategy_stats['failed'] += 1
                return ""
            
            # Extract first line (before any newline)
            first_line = resp.split('\n')[0].strip()
            first_line_lower = first_line.lower()
            
            if self.verbose:
                eval_logger.debug(f"[FINBenchV2] Full response: '{resp[:200]}'")
                eval_logger.debug(f"[FINBenchV2] First line: '{first_line[:80]}'")
                eval_logger.debug(f"[FINBenchV2] Target: '{target}'")
                eval_logger.debug(f"[FINBenchV2] All choices: {all_choices}")
            
            def normalize_text(text):
                """Normalize text for robust matching: lowercase, strip punctuation, normalize whitespace."""
                text = text.lower()
                text = text.strip().strip('.,!?;:')
                text = ' '.join(text.split())
                return text
            
            def strip_special_chars(text):
                """Remove all punctuation and special characters, keep only alphanumeric and spaces."""
                return re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
            
            # Prepare normalized versions
            first_line_normalized = normalize_text(first_line)
            first_line_stripped = strip_special_chars(first_line_normalized)
            
            # Strategy 1: Exact substring match (case-insensitive)
            for choice in all_choices:
                if choice and choice.lower() in first_line_lower:
                    is_correct = choice.lower() == target.lower()
                    if is_correct:
                        if self.verbose:
                            eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (substring): '{choice}'")
                        self.stats['correct'] += 1
                    else:
                        if self.verbose:
                            eval_logger.debug(f"[FINBenchV2] ✗ WRONG (substring): '{choice}' (expected: '{target}')")
                        self.stats['wrong'] += 1
                    self.strategy_stats['substring_match'] += 1
                    return choice
            
            # Strategy 2: Normalized exact match
            for choice in all_choices:
                if choice:
                    choice_normalized = normalize_text(choice)
                    if first_line_normalized == choice_normalized:
                        is_correct = choice.lower() == target.lower()
                        if is_correct:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (normalized): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✗ WRONG (normalized): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        self.strategy_stats['normalized_exact'] += 1
                        return choice
            
            # Strategy 3: Stripped special characters match
            for choice in all_choices:
                if choice:
                    choice_stripped = strip_special_chars(normalize_text(choice))
                    if first_line_stripped == choice_stripped:
                        is_correct = choice.lower() == target.lower()
                        if is_correct:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (stripped): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✗ WRONG (stripped): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        self.strategy_stats['stripped_special'] += 1
                        return choice
            
            # Strategy 4: Fuzzy matching with 80% similarity threshold
            for choice in all_choices:
                if choice:
                    choice_normalized = normalize_text(choice)
                    similarity = difflib.SequenceMatcher(
                        None, 
                        first_line_normalized, 
                        choice_normalized
                    ).ratio()
                    
                    if similarity >= 0.80:
                        is_correct = choice.lower() == target.lower()
                        if is_correct:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (fuzzy {similarity:.2%}): '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✗ WRONG (fuzzy {similarity:.2%}): '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        self.strategy_stats['fuzzy_80'] += 1
                        return choice
            
            # Strategy 5: Check for bold/emphasized answers (e.g., **word**)
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', first_line)
            for bold_word in bold_matches:
                for choice in all_choices:
                    if choice and len(choice) >= 4:
                        stem = choice.lower()[:max(4, len(choice)-3)]
                        if bold_word.lower() == choice.lower() or bold_word.lower().startswith(stem):
                            is_correct = choice.lower() == target.lower()
                            if is_correct:
                                if self.verbose:
                                    eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (bold): '{bold_word}' → '{choice}'")
                                self.stats['correct'] += 1
                            else:
                                if self.verbose:
                                    eval_logger.debug(f"[FINBenchV2] ✗ WRONG (bold): '{bold_word}' → '{choice}' (expected: '{target}')")
                                self.stats['wrong'] += 1
                            self.strategy_stats['bold_match'] += 1
                            return choice
            
            # Strategy 6: Check for Finnish inflections
            for choice in all_choices:
                if choice and len(choice) >= 4:
                    stem = choice.lower()[:max(4, len(choice)-3)]
                    if re.search(r'\b' + re.escape(stem), first_line_lower):
                        is_correct = choice.lower() == target.lower()
                        if is_correct:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✓ CORRECT (stem): '{stem}' → '{choice}'")
                            self.stats['correct'] += 1
                        else:
                            if self.verbose:
                                eval_logger.debug(f"[FINBenchV2] ✗ WRONG (stem): '{stem}' → '{choice}' (expected: '{target}')")
                            self.stats['wrong'] += 1
                        self.strategy_stats['stem_match'] += 1
                        return choice
            
            # If we get here, couldn't parse the response into any valid choice
            self.stats['invalid'] += 1
            self.strategy_stats['failed'] += 1
            eval_logger.error(
                f"[FINBenchV2] ✗✗ INVALID FORMAT - could not parse response!\n"
                f"   First line: '{first_line}'\n"
                f"   Full response: '{resp[:300]}...'\n"
                f"   Expected: answer word from choices\n"
                f"   Valid choices: {all_choices}\n"
                f"   Target: '{target}'\n"
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
                f"\n{'='*80}\n"
                f"FINBenchV2AnswerFilter Summary:\n"
                f"  Total responses: {self.stats['total']}\n"
                f"  Correct: {self.stats['correct']} ({accuracy:.1f}%)\n"
                f"  Wrong: {self.stats['wrong']} ({self.stats['wrong']/self.stats['total']*100:.1f}%)\n"
                f"  Invalid format: {self.stats['invalid']} ({invalid_rate:.1f}%)\n"
                f"    - Empty responses: {self.stats['empty_response']}\n"
                f"    - No choices found: {self.stats['no_choices']}\n"
                f"    - No target found: {self.stats['no_target']}\n"
                f"    - Unparseable: {self.stats['invalid'] - self.stats['empty_response'] - self.stats['no_choices'] - self.stats['no_target']}\n"
                f"\n"
                f"  Extraction Strategy Breakdown:\n"
                f"    - Substring match: {self.strategy_stats['substring_match']} ({self.strategy_stats['substring_match']/self.stats['total']*100:.1f}%)\n"
                f"    - Normalized exact: {self.strategy_stats['normalized_exact']} ({self.strategy_stats['normalized_exact']/self.stats['total']*100:.1f}%)\n"
                f"    - Stripped special: {self.strategy_stats['stripped_special']} ({self.strategy_stats['stripped_special']/self.stats['total']*100:.1f}%)\n"
                f"    - Fuzzy match (80%): {self.strategy_stats['fuzzy_80']} ({self.strategy_stats['fuzzy_80']/self.stats['total']*100:.1f}%)\n"
                f"    - Bold emphasis: {self.strategy_stats['bold_match']} ({self.strategy_stats['bold_match']/self.stats['total']*100:.1f}%)\n"
                f"    - Stem/inflection: {self.strategy_stats['stem_match']} ({self.strategy_stats['stem_match']/self.stats['total']*100:.1f}%)\n"
                f"    - Failed: {self.strategy_stats['failed']} ({self.strategy_stats['failed']/self.stats['total']*100:.1f}%)\n"
                f"{'='*80}"
            )
            
            # Validation check: Warn if invalid rate is high
            if invalid_rate > 10.0:
                eval_logger.warning(
                    f"\n⚠️  WARNING: High invalid format rate ({invalid_rate:.1f}%)!\n"
                    f"   This may indicate:\n"
                    f"   - Prompt issues (model not understanding instructions)\n"
                    f"   - Schema detection problems (choices/target not found)\n"
                    f"   - Model generating verbose responses instead of direct answers\n"
                    f"   Review logs for '✗✗ INVALID FORMAT' messages\n"
                )
        
        return filtered_resps
