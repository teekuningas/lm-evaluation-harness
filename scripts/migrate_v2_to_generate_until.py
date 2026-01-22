#!/usr/bin/env python3
"""
Migration script to convert FINBench-v2 multiple_choice tasks to generate_until format.

This script creates generate_until task configurations for thinking models (gpt-oss, deepseek-r1)
that cannot provide token probabilities.

SIMPLIFIED APPROACH:
- ALL generated tasks go to finbench_v2/gen/ (centralized)
- ONE shared utils.py with choice extraction logic
- Schema-aware doc_to_target based on dataset structure
- No need to copy utils.py to each category

Usage:
    # Migrate all tasks
    python scripts/migrate_v2_to_generate_until.py
    
    # Dry-run
    python scripts/migrate_v2_to_generate_until.py --dry-run

Author: Rewritten for FINBench-v2 migration
Date: 2026-01-21
"""

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset schemas - defines how to extract answer text for doc_to_target
# AND how to list choices in CF prompts
DATASET_SCHEMAS = {
    'TurkuNLP/finbenchv2-arc-c-fi-ht': {
        # Schema: choices.text=[...], choices.label=['A','B','C','D'], answerKey='A'
        'doc_to_target': '{{ choices.text[choices.label.index(answerKey)] }}',
        'doc_to_choice': '{{ choices.text }}',
        'choices_template': "{% for s in choices.text %}{{ loop.index0 }}. {{ s }}\n{% endfor %}",
    },
    'TurkuNLP/finbenchv2-belebele-fi-og': {
        # Schema: mc_answer1, mc_answer2, mc_answer3, mc_answer4, correct_answer_num='1'
        'doc_to_target': "{{ [mc_answer1, mc_answer2, mc_answer3, mc_answer4][['1','2','3','4'].index(correct_answer_num)] }}",
        'doc_to_choice': "{{ [mc_answer1, mc_answer2, mc_answer3, mc_answer4] }}",
        'choices_template': "0. {{ mc_answer1 }}\n1. {{ mc_answer2 }}\n2. {{ mc_answer3 }}\n3. {{ mc_answer4 }}\n",
    },
    'TurkuNLP/finbenchv2-goldenswag-fi-ht': {
        # Schema AFTER process_docs: choices=[...] (flat list), gold=<index>
        'doc_to_target': '{{ choices[gold] }}',
        'doc_to_choice': '{{ choices }}',
        'choices_template': "{% for s in choices %}{{ loop.index0 }}. {{ s }}\n{% endfor %}",
    },
    'TurkuNLP/finbenchv2-scandisent-fi-mini': {
        # Schema AFTER process_docs: choices=['positiivinen', 'negatiivinen'], gold='positiivinen' (answer string)
        'doc_to_target': '{{ gold }}',
        'doc_to_choice': '{{ choices }}',
        'choices_template': "{% for s in choices %}{{ loop.index0 }}. {{ s }}\n{% endfor %}",
    },
    'TurkuNLP/finbenchv2-sib-200-fi-og': {
        # Schema: choices=['A','B','C','D'], answer_idx=0 (direct index)
        'doc_to_target': '{{ choices[answer_idx] }}',
        'doc_to_choice': '{{ choices }}',
        'choices_template': "{% for s in choices %}{{ loop.index0 }}. {{ s }}\n{% endfor %}",
    },
    'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht': {
        # Schema: multiple_choice_targets=[...] (list of choices), targets=[correct_answer] (list with one element)
        'doc_to_target': '{{ targets[0] }}',
        'doc_to_choice': '{{ multiple_choice_targets }}',
        'choices_template': "{% for s in multiple_choice_targets %}{{ loop.index0 }}. {{ s }}\n{% endfor %}",
    },
}

# Task categories
TASK_CATEGORIES = {
    'arc_c': {
        'path': 'lm_eval/tasks/finbench_v2/arc_c',
        'task_prefix': 'arc_challenge_fi',
        'dataset_path': 'TurkuNLP/finbenchv2-arc-c-fi-ht',
    },
    'belebele_fin': {
        'path': 'lm_eval/tasks/finbench_v2/belebele_fin',
        'task_prefix': 'belebele_fin',
        'dataset_path': 'TurkuNLP/finbenchv2-belebele-fi-og',
    },
    'goldenswag': {
        'path': 'lm_eval/tasks/finbench_v2/goldenswag',
        'task_prefix': 'goldenswag_fi',
        'dataset_path': 'TurkuNLP/finbenchv2-goldenswag-fi-ht',
    },
    'scandisent': {
        'path': 'lm_eval/tasks/finbench_v2/scandisent',
        'task_prefix': 'scandisent_fi',
        'dataset_path': 'TurkuNLP/finbenchv2-scandisent-fi-mini',
    },
    'sib200': {
        'path': 'lm_eval/tasks/finbench_v2/sib200',
        'task_prefix': 'sib200_fi',
        'dataset_path': 'TurkuNLP/finbenchv2-sib-200-fi-og',
    },
    'finbench_analogies': {
        'path': 'lm_eval/tasks/finbench_v2/finbench_v1_multiprompt',
        'task_prefix': 'finbench_analogies',
        'dataset_path': 'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht',
    },
    'finbench_emotions_1k': {
        'path': 'lm_eval/tasks/finbench_v2/finbench_v1_multiprompt',
        'task_prefix': 'finbench_emotions_1k',
        'dataset_path': 'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht',
    },
    'finbench_general_knowledge': {
        'path': 'lm_eval/tasks/finbench_v2/finbench_v1_multiprompt',
        'task_prefix': 'finbench_general_knowledge',
        'dataset_path': 'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht',
    },
    'finbench_hhh_alignment': {
        'path': 'lm_eval/tasks/finbench_v2/finbench_v1_multiprompt',
        'task_prefix': 'finbench_hhh_alignment',
        'dataset_path': 'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht',
    },
    'finbench_similarities_abstraction': {
        'path': 'lm_eval/tasks/finbench_v2/finbench_v1_multiprompt',
        'task_prefix': 'finbench_similarities_abstraction',
        'dataset_path': 'TurkuNLP/finbenchv2-fbv1-stripped-fi-ht',
    },
}


def load_yaml(filepath: Path) -> Dict:
    """Load YAML file safely, handling custom tags like !function."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Add a custom constructor for !function tags - just keep them as strings
            yaml.add_constructor('!function', lambda loader, node: loader.construct_scalar(node), Loader=yaml.SafeLoader)
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return {}


def save_yaml(data: Dict, filepath: Path, dry_run: bool = False) -> bool:
    """Save YAML file with proper formatting, handling !function tags."""
    if dry_run:
        logger.info(f"[DRY-RUN] Would create: {filepath}")
        logger.debug(f"Content:\n{yaml.dump(data, sort_keys=False, allow_unicode=True)}")
        return True
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # First dump to string
        yaml_str = yaml.dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
        
        # Post-process to add !function tags where needed
        # If a field contains process_docs or doc_to_choice with function reference, add !function prefix
        yaml_str = re.sub(r'^(process_docs: )(.+)$', r'\1!function \2', yaml_str, flags=re.MULTILINE)
        
        # Post-process to unquote !function tags
        # Replace '!function ...' or "!function ..." with !function ... (remove quotes)
        yaml_str = re.sub(r"['\"](!function [^'\"]+)['\"]", r'\1', yaml_str)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_str)
        
        logger.info(f"✓ Created: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return False


def create_base_yaml_content(dataset_path: str, original_base: Dict) -> Dict:
    """
    Create the base YAML template for generate_until tasks.
    Uses schema-aware doc_to_target based on dataset_path.
    Note: doc_to_text is NOT included here - each prompt variant defines its own.
    """
    # Get schema-specific doc_to_target
    # Note: doc_to_choice is NOT needed for generate_until tasks
    schema = DATASET_SCHEMAS.get(dataset_path, {})
    doc_to_target = schema.get('doc_to_target', '{{ answer }}')  # fallback
    
    base_config = {
        'output_type': 'generate_until',
        # No doc_to_text here - each prompt variant (p0-p4) defines its own
        'doc_to_target': doc_to_target,
        # No doc_to_choice - not needed for generate_until, and it causes doc modification issues
        'generation_kwargs': {
            'until': [],
            'max_gen_toks': 4096,
            'do_sample': False,
            'temperature': 0,
        },
        'filter_list': [
            {
                'name': 'finbench_v2_answer_extraction',
                'filter': [
                    {'function': '!function utils.FINBenchV2AnswerFilter'}
                ]
            }
        ],
        'metric_list': [
            {
                'metric': 'exact_match',
                'aggregation': 'mean',
                'higher_is_better': True,
                'ignore_punctuation': True,
            }
        ],
    }
    
    # Copy over relevant fields from original
    if 'dataset_path' in original_base:
        base_config['dataset_path'] = original_base['dataset_path']
    # Skip dataset_name - most datasets only have 'default' config
    # if 'dataset_name' in original_base:
    #     base_config['dataset_name'] = original_base['dataset_name']
    if 'test_split' in original_base:
        base_config['test_split'] = original_base['test_split']
    if 'training_split' in original_base:
        base_config['training_split'] = original_base['training_split']
    if 'validation_split' in original_base:
        base_config['validation_split'] = original_base['validation_split']
    if 'process_docs' in original_base:
        logger.debug(f"Copying process_docs: {original_base['process_docs']}")
        base_config['process_docs'] = original_base['process_docs']
    if 'should_decontaminate' in original_base:
        base_config['should_decontaminate'] = original_base['should_decontaminate']
    if 'doc_to_decontamination_query' in original_base:
        base_config['doc_to_decontamination_query'] = original_base['doc_to_decontamination_query']
    
    logger.debug(f"Final base_config keys: {list(base_config.keys())}")
    logger.debug(f"Has process_docs in final: {'process_docs' in base_config}")
    
    # Update tags
    tags = original_base.get('tag', [])
    new_tags = []
    for tag in tags:
        if 'multiple_choice' in tag:
            new_tags.append(tag.replace('multiple_choice', 'generation'))
        else:
            new_tags.append(tag)
    # Add generation-specific tag
    if 'finbench_v2_generation' not in new_tags:
        new_tags.append('finbench_v2_generation')
    base_config['tag'] = new_tags
    
    # Add metadata
    base_config['metadata'] = {'version': '1.0'}
    
    return base_config


def create_prompt_yaml_content(task_name: str, base_yaml_name: str, doc_to_text: str, 
                               format_type: str, choices_template: str, 
                               original_prompt: Dict = None) -> Dict:
    """
    Create prompt variant YAML that includes the base template.
    
    Each prompt variant will have its own doc_to_text function generated in doc_to_texts.py
    that preserves the unique template while adding instructions and choices.
    
    Args:
        task_name: New task name
        base_yaml_name: Base YAML to include
        doc_to_text: Original doc_to_text template from the prompt variant
        format_type: cf or mcf
        choices_template: Template for choices (not used, kept for signature)
        original_prompt: Original prompt dict (to extract dataset_name for v1 tasks)
    """
    # Generate function name based on task name
    # Replace hyphens with underscores for valid Python identifiers
    func_name = f"doc_to_text_{task_name.replace('-', '_')}"
    
    config = {
        'task': task_name,
        'include': base_yaml_name,
        # Point to the generated function in doc_to_texts module
        'doc_to_text': f'!function doc_to_texts.{func_name}'
    }
    
    # For v1_multiprompt tasks, include dataset_name from original
    if original_prompt and 'dataset_name' in original_prompt:
        config['dataset_name'] = original_prompt['dataset_name']
    
    return config


def create_group_yaml_content(group_name: str, task_list: List[str]) -> Dict:
    """Create group YAML for aggregating prompt variants."""
    return {
        'group': group_name,
        'task': task_list,
        'aggregate_metric_list': [
            {
                'metric': 'exact_match',
                'aggregation': 'mean',
                'weight_by_size': True,
            }
        ],
    }


def extract_task_info(yaml_path: Path) -> Tuple[str, str, str]:
    """
    Extract task information from YAML filename.
    
    Returns:
        (task_base_name, format_type, prompt_variant)
        e.g., ('arc_challenge_fi', 'cf', 'p0')
    """
    filename = yaml_path.stem
    
    # Extract format (cf or mcf)
    if '_cf_' in filename:
        format_type = 'cf'
    elif '_mcf_' in filename:
        format_type = 'mcf'
    else:
        format_type = None
    
    # Extract prompt variant (p0-p4)
    match = re.search(r'_p(\d+)$', filename)
    prompt_variant = f'p{match.group(1)}' if match else None
    
    # Extract base task name (everything before format)
    if format_type:
        task_base_name = filename.split(f'_{format_type}_')[0]
    else:
        task_base_name = filename
    
    return task_base_name, format_type, prompt_variant


def generate_doc_to_texts_file(templates: Dict[str, str], gen_path: Path, dry_run: bool = False) -> bool:
    """
    Generate doc_to_texts.py file with all task-specific doc_to_text functions.
    
    Args:
        templates: Dict mapping task_name -> original doc_to_text template string
        gen_path: Path to gen directory
        dry_run: If True, don't actually write the file
        
    Returns:
        True if successful
    """
    doc_to_texts_path = gen_path / 'doc_to_texts.py'
    
    # Build the Python file content
    lines = []
    lines.append('"""')
    lines.append('Auto-generated doc_to_text functions for FINBench-v2 generate_until tasks.')
    lines.append('')
    lines.append('Each function corresponds to a specific prompt variant (p0-p4) and preserves')
    lines.append('the unique template while adding instructions.')
    lines.append('')
    lines.append('Note: MCF (multiple-choice-first) templates include choices inline,')
    lines.append('so we do NOT add them again at the end.')
    lines.append('')
    lines.append('Generated by: scripts/migrate_v2_to_generate_until.py')
    lines.append('DO NOT EDIT MANUALLY - regenerate by running the migration script')
    lines.append('"""')
    lines.append('')
    lines.append('from jinja2 import Template')
    lines.append('try:')
    lines.append('    from . import utils')
    lines.append('except ImportError:')
    lines.append('    # If relative import fails (when loaded as standalone), use absolute import')
    lines.append('    from lm_eval.tasks.finbench_v2.gen import utils')
    lines.append('')
    lines.append('')
    lines.append('def render_template_with_instructions(doc, template_str):')
    lines.append('    """')
    lines.append('    Render a Jinja2 template with doc data and add instructions.')
    lines.append('    ')
    lines.append('    For MCF templates, choices are already included inline in the template,')
    lines.append('    so we only add instructions at the beginning.')
    lines.append('    ')
    lines.append('    Args:')
    lines.append('        doc: Document dict from dataset')
    lines.append('        template_str: Jinja2 template string for the question')
    lines.append('        ')
    lines.append('    Returns:')
    lines.append('        Complete prompt with instructions and question (including inline choices)')
    lines.append('    """')
    lines.append('    prompt_parts = []')
    lines.append('    ')
    lines.append('    # Add critical instructions')
    lines.append('    instructions = (')
    lines.append('        "TÄRKEÄÄ: Vastaa lyhyesti ja selkeästi. "')
    lines.append('        "Valitse täsmälleen yksi vastausvaihtoehto annetuista. "')
    lines.append('        "Vastaa VAIN valintavaihtoehdon TÄYDELLÄ TEKSTILLÄ. "')
    lines.append('        "ÄLÄ käytä pelkkää kirjainta (A, B, C, D) tai numeroa (1, 2, 3, 4). "')
    lines.append('        "Älä anna pitkiä selityksiä. "')
    lines.append('        "Vastauksesi luetaan automaattisesti.\\n\\n"')
    lines.append('    )')
    lines.append('    prompt_parts.append(instructions)')
    lines.append('    ')
    lines.append('    # Render the template with doc data')
    lines.append('    try:')
    lines.append('        template = Template(template_str)')
    lines.append('        question_text = template.render(**doc)')
    lines.append('        prompt_parts.append(question_text)')
    lines.append('    except Exception as e:')
    lines.append('        # If template rendering fails, fall back to basic extraction')
    lines.append('        if "query" in doc:')
    lines.append('            prompt_parts.append(f"{doc[\'query\']}\\n")')
    lines.append('        elif "question" in doc:')
    lines.append('            prompt_parts.append(f"{doc[\'question\']}\\n")')
    lines.append('        elif "text" in doc:')
    lines.append('            prompt_parts.append(f"{doc[\'text\']}\\n")')
    lines.append('        elif "inputs" in doc:')
    lines.append('            prompt_parts.append(f"{doc[\'inputs\']}\\n")')
    lines.append('        ')
    lines.append('        # Add choices if template failed and they exist')
    lines.append('        choices = utils.get_choices_from_doc(doc, debug=False)')
    lines.append('        if choices:')
    lines.append('            prompt_parts.append("\\nVaihtoehdot:\\n")')
    lines.append('            for i, choice in enumerate(choices):')
    lines.append('                prompt_parts.append(f"{i}. {choice}\\n")')
    lines.append('    ')
    lines.append('    return "".join(prompt_parts)')
    lines.append('')
    lines.append('')
    
    # Generate a function for each task
    for task_name, template_str in sorted(templates.items()):
        func_name = f"doc_to_text_{task_name.replace('-', '_')}"
        # Escape the template string for Python
        template_escaped = template_str.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        
        lines.append(f'def {func_name}(doc):')
        lines.append(f'    """Generated doc_to_text for {task_name}."""')
        lines.append(f'    template = "{template_escaped}"')
        lines.append(f'    return render_template_with_instructions(doc, template)')
        lines.append('')
    
    content = '\n'.join(lines)
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would create: {doc_to_texts_path}")
        logger.debug(f"Content preview:\n{content[:500]}...")
        return True
    
    try:
        with open(doc_to_texts_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✓ Generated: {doc_to_texts_path} with {len(templates)} functions")
        return True
    except Exception as e:
        logger.error(f"Failed to generate {doc_to_texts_path}: {e}")
        return False


def migrate_category(category: str, dry_run: bool = False) -> Tuple[int, Dict[str, str]]:
    """
    Migrate all tasks in a category to generate_until format.
    ALL generated files go to finbench_v2/gen/ (centralized).
    
    Returns:
        Tuple of (number of tasks migrated, dict of task_name -> template)
    """
    if category not in TASK_CATEGORIES:
        logger.error(f"Unknown category: {category}")
        return 0, {}
    
    cat_info = TASK_CATEGORIES[category]
    base_path = Path(cat_info['path'])
    
    if not base_path.exists():
        logger.error(f"Category path does not exist: {base_path}")
        return 0, {}
    
    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Migrating category: {category}")
    logger.info(f"Source: {base_path}")
    
    # Centralized gen directory
    gen_path = Path('lm_eval/tasks/finbench_v2/gen')
    if not dry_run:
        gen_path.mkdir(exist_ok=True)
    
    migrated_count = 0
    templates = {}  # Collect templates: task_name -> doc_to_text template
    dataset_path = cat_info['dataset_path']
    
    # Get schema for this dataset
    schema = DATASET_SCHEMAS.get(dataset_path, {})
    choices_template = schema.get('choices_template', '{% for s in choices %}{{ loop.index0 }}. {{ s }}\n{% endfor %}')
    
    # Only process MCF (multiple-choice-first) format
    # MCF templates include choices inline, so we don't need CF variants
    for format_type in ['mcf']:
        format_path = base_path / format_type
        if not format_path.exists():
            logger.debug(f"Format directory does not exist: {format_path}")
            continue
        
        # Find original base YAML
        # v1_multiprompt uses shared template, others use task-specific base files
        base_yaml_files = list(format_path.glob(f'_{cat_info["task_prefix"]}*{format_type}*yaml'))
        if not base_yaml_files:
            # Try shared template (for v1_multiprompt)
            base_yaml_files = list(format_path.glob('_*template*yaml'))
        if not base_yaml_files:
            logger.warning(f"No base YAML found in {format_path}")
            continue
        
        original_base_path = base_yaml_files[0]
        original_base = load_yaml(original_base_path)
        
        if not original_base:
            logger.error(f"Failed to load base YAML: {original_base_path}")
            continue
        
        # Create new base YAML for generate_until (schema-aware)
        new_base_yaml_name = f'_{cat_info["task_prefix"]}_gen_{format_type}_fbv2_yaml'
        new_base_path = gen_path / new_base_yaml_name
        new_base_content = create_base_yaml_content(dataset_path, original_base)
        
        # Save base YAML (always overwrite to ensure updates)
        save_yaml(new_base_content, new_base_path, dry_run)
        
        # Find all prompt variant files
        prompt_files = sorted(format_path.glob(f'{cat_info["task_prefix"]}*{format_type}*p[0-4].yaml'))
        
        if not prompt_files:
            logger.warning(f"No prompt files found in {format_path}")
            continue
        
        logger.info(f"Found {len(prompt_files)} prompt files in {format_path}")
        
        gen_tasks = []
        
        for prompt_file in prompt_files:
            task_base, fmt, prompt_var = extract_task_info(prompt_file)
            
            if not prompt_var:
                logger.warning(f"Could not extract prompt variant from {prompt_file}")
                continue
            
            # Load original prompt file to get doc_to_text and dataset_name
            original_prompt = load_yaml(prompt_file)
            doc_to_text = original_prompt.get('doc_to_text', '')
            
            # IMPORTANT: Some MCF tasks (like emotions p0/p4, analogies p3, etc.) 
            # have choices/instructions in 'description' field.
            # In multiple_choice tasks, description is prepended to create full context.
            # We must do the same for generate_until to preserve prompt structure.
            description = original_prompt.get('description', '')
            if description:
                # Prepend description to doc_to_text (matches lm-eval's fewshot_context behavior)
                doc_to_text = description + doc_to_text
                logger.info(f"  Added description to {prompt_var}: '{description[:50]}...'")
            
            
            # Create new task name
            new_task_name = f'{task_base}_gen_{format_type}_fbv2_{prompt_var}'
            
            # Store the template for this task
            templates[new_task_name] = doc_to_text
            
            # Create new prompt YAML
            new_prompt_content = create_prompt_yaml_content(
                new_task_name,
                new_base_yaml_name,
                doc_to_text,
                format_type,
                choices_template,
                original_prompt  # Pass original to extract dataset_name for v1 tasks
            )
            
            new_prompt_path = gen_path / f'{new_task_name}.yaml'
            
            if save_yaml(new_prompt_content, new_prompt_path, dry_run):
                gen_tasks.append(new_task_name)
                migrated_count += 1
        
        # Create group file for this format's gen tasks
        if gen_tasks:
            group_name = f'group_{task_base}_multi_gen_{format_type}_fbv2'
            group_path = gen_path / f'_group_{task_base}_multi_gen_{format_type}_fbv2.yaml'
            group_content = create_group_yaml_content(group_name, gen_tasks)
            save_yaml(group_content, group_path, dry_run)
    
    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Completed: {migrated_count} tasks migrated in {category}")
    return migrated_count, templates


def migrate_all_categories(dry_run: bool = False) -> int:
    """Migrate all categories and generate doc_to_texts.py."""
    total = 0
    all_templates = {}
    
    for category in TASK_CATEGORIES.keys():
        count, templates = migrate_category(category, dry_run)
        total += count
        all_templates.update(templates)
    
    # Generate doc_to_texts.py with all collected templates
    gen_path = Path('lm_eval/tasks/finbench_v2/gen')
    if all_templates:
        generate_doc_to_texts_file(all_templates, gen_path, dry_run)
    
    return total


def main():
    parser = argparse.ArgumentParser(
        description='Migrate FINBench-v2 multiple_choice tasks to generate_until format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--category',
        choices=list(TASK_CATEGORIES.keys()) + ['all'],
        default='all',
        help='Task category to migrate (default: all)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run migration
    if args.category == 'all':
        total = migrate_all_categories(args.dry_run)
    else:
        count, templates = migrate_category(args.category, args.dry_run)
        total = count
        # Generate doc_to_texts.py even for single category
        gen_path = Path('lm_eval/tasks/finbench_v2/gen')
        if templates:
            generate_doc_to_texts_file(templates, gen_path, args.dry_run)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Migration {'(DRY-RUN) ' if args.dry_run else ''}complete: {total} tasks")
    logger.info(f"Output directory: lm_eval/tasks/finbench_v2/gen/")
    logger.info(f"Generated doc_to_texts.py with task-specific functions")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
