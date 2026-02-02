"""
DRS Preprocessor - Convert DRS files to graph-structured JSONL

Extracts four components for DRS embeddings:
1. Temporal Graph: events as nodes, temporal relations as edges
2. Logical Layer: event types, VerbNet classes, semantic predicates
3. Participant Graph: actors as nodes, coreference as edges
4. Semantic Relations: agent/patient links between actors and events

Output: JSONL with one line per triplet (anchor, story_a, story_b)
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EventNode:
    """Represents an event in the temporal graph."""
    id: str                    # Variable (a, b, c...)
    text: str                  # Verb text (e.g., "follows", "named")
    event_id: str              # Original ID (T40, T41...)
    type: Optional[str]        # State, Process, Transition
    tense: Optional[str]       # Past, Present, Future
    aspect: Optional[str]      # Perfective, Progressive
    vform: Optional[str]       # Participle, Infinitive
    position: int              # Document order


@dataclass
class ActorNode:
    """Represents a participant in the participant graph."""
    id: str                    # ID (T1, T2...)
    text: str                  # Description text
    is_event_ref: bool         # True if this is actually an event reference


@dataclass
class TemporalEdge:
    """Edge in the temporal graph."""
    source: str                # Event variable
    target: str                # Event variable
    relation: str              # occursBefore, overlaps, during, etc.


@dataclass
class SemanticEdge:
    """Edge linking actors to events."""
    actor_id: str              # Actor ID
    event_id: str              # Event ID
    role: str                  # agent, patient, theme, etc.


@dataclass
class CoreferenceEdge:
    """Edge in the participant graph (coreference)."""
    source: str                # Actor ID
    target: str                # Actor ID


@dataclass
class DRSGraph:
    """Complete DRS representation as graphs."""
    # Temporal component
    events: List[Dict]
    temporal_edges: List[Dict]

    # Logical component
    event_types: Dict[str, str]       # event_var -> type
    verbnet_classes: Dict[str, str]   # event_var -> VerbNet class
    logic_predicates: List[str]       # Semantic predicates from VerbNet

    # Participant component
    actors: List[Dict]
    coreference_edges: List[Dict]

    # Semantic relations component
    semantic_edges: List[Dict]

    # Metadata
    num_events: int
    num_actors: int
    num_temporal_edges: int
    num_coreference_edges: int
    num_semantic_edges: int


class DRSGraphExtractor:
    """Extract graph structures from DRS files."""

    # VerbNet normalizer (shared across instances)
    _verbnet_normalizer = None
    _verbnet_predicates = None

    def __init__(self):
        if DRSGraphExtractor._verbnet_normalizer is None:
            self._init_verbnet()

    @classmethod
    def _init_verbnet(cls):
        """Initialize VerbNet lookup tables."""
        cls._verbnet_normalizer = {}
        cls._verbnet_predicates = {}
        cls._class_predicates = {}  # Also store by class name for fallback

        try:
            from nltk.corpus import verbnet as vn

            predicates_found = 0

            for classid in vn.classids():
                vnclass = vn.vnclass(classid)
                parent_class = '-'.join(classid.split('-')[:2])

                # Get semantic predicates - they're nested inside FRAMES/FRAME/SEMANTICS/PRED
                # Use recursive search (.//PRED) to find all PRED elements anywhere in the tree
                preds = vnclass.findall('.//PRED')
                pred_values = list(set([p.get('value') for p in preds if p.get('value')]))

                if pred_values:
                    predicates_found += len(pred_values)
                    # Store by class name for fallback
                    cls._class_predicates[parent_class] = pred_values
                    cls._class_predicates[classid] = pred_values

                for member in vnclass.findall('MEMBERS/MEMBER'):
                    verb = member.get('name')
                    if verb:
                        cls._verbnet_normalizer[verb] = parent_class
                        # Store predicates for this verb (may be overwritten if verb appears in multiple classes)
                        if pred_values:  # Only store if we found predicates
                            cls._verbnet_predicates[verb] = pred_values

            print(f"✓ VerbNet loaded: {len(cls._verbnet_normalizer)} verbs, {len(cls._class_predicates)} classes with predicates")

        except Exception as e:
            print(f"Note: VerbNet not available ({e}), using lemma-based fallback")
            cls._class_predicates = {}

    # Cached spacy model
    _spacy_nlp = None

    @classmethod
    def _get_spacy(cls):
        """Get cached spacy model."""
        if cls._spacy_nlp is None:
            try:
                import spacy
                # Try larger model first, fall back to smaller
                for model in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                    try:
                        cls._spacy_nlp = spacy.load(model, disable=['ner', 'parser'])
                        print(f"✓ Spacy model loaded: {model}")
                        break
                    except OSError:
                        continue
            except ImportError:
                pass
        return cls._spacy_nlp

    def _get_verb_lemma(self, text: str) -> str:
        """Extract verb lemma from event text."""
        if not text:
            return ""

        # Try spacy first for proper lemmatization
        nlp = self._get_spacy()
        if nlp is not None:
            doc = nlp(text.lower())
            # Find the main verb
            for token in doc:
                if token.pos_ == 'VERB':
                    return token.lemma_
            # Fallback to first token's lemma
            if len(doc) > 0:
                return doc[0].lemma_

        # Fallback: simple heuristic (less accurate)
        word = text.lower().split()[0]

        # Basic stemming for common verb endings
        if word.endswith('ing') and len(word) > 4:
            # Handle doubling: running -> run, but arriving -> arrive
            stem = word[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:  # doubled consonant
                stem = stem[:-1]
            return stem
        elif word.endswith('ed') and len(word) > 3:
            stem = word[:-2]
            if stem.endswith('i'):  # tried -> try
                stem = stem[:-1] + 'y'
            return stem
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('s') and len(word) > 2:
            return word[:-1]

        return word

    def _get_verbnet_class(self, text: str) -> Tuple[str, List[str]]:
        """Get VerbNet class and predicates for event text."""
        lemma = self._get_verb_lemma(text)

        vn_class = self._verbnet_normalizer.get(lemma, lemma)

        # Try to get predicates by lemma first
        predicates = self._verbnet_predicates.get(lemma, [])

        # Fallback: if we matched a VerbNet class, look up predicates by class name
        if not predicates and vn_class != lemma:
            predicates = self._class_predicates.get(vn_class, [])

        return vn_class, predicates

    def parse_drs_file(self, filepath: str) -> Optional[DRSGraph]:
        """Parse a single DRS file and extract graph structures."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        except FileNotFoundError:
            return None

        # Split into sections
        sections = self._split_sections(content)

        # Parse each section
        events, temporal_edges, event_types, vn_classes, all_predicates = \
            self._parse_events_section(sections.get('EVENTS', ''))

        actors, event_refs = self._parse_actors_section(sections.get('ACTORS', ''))

        semantic_edges, coref_edges = self._parse_relations_section(
            sections.get('RELATIONS', ''), event_refs
        )

        return DRSGraph(
            events=[asdict(e) for e in events],
            temporal_edges=[asdict(e) for e in temporal_edges],
            event_types=event_types,
            verbnet_classes=vn_classes,
            logic_predicates=list(set(all_predicates)),
            actors=[asdict(a) for a in actors],
            coreference_edges=[asdict(e) for e in coref_edges],
            semantic_edges=[asdict(e) for e in semantic_edges],
            num_events=len(events),
            num_actors=len(actors),
            num_temporal_edges=len(temporal_edges),
            num_coreference_edges=len(coref_edges),
            num_semantic_edges=len(semantic_edges)
        )

    def _split_sections(self, content: str) -> Dict[str, str]:
        """Split DRS content into sections."""
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            line_stripped = line.strip()

            # Check for section headers - handle various encodings of »
            # The » character (U+00BB) can appear as:
            # - '»' (proper UTF-8)
            # - 'Â»' (UTF-8 bytes read as latin-1)
            # - '\xbb' (raw byte)
            is_header = False
            section_name = None

            for marker in ['»', 'Â»', '\xbb', 'Â\xbb']:
                if marker in line_stripped:
                    is_header = True
                    # Remove all variants of the marker
                    section_name = line_stripped
                    for m in ['»', 'Â»', '\xbb', 'Â\xbb', 'Â']:
                        section_name = section_name.replace(m, '')
                    section_name = section_name.strip().upper()
                    break

            if is_header and section_name:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = section_name
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content)

        return sections

    def _parse_events_section(self, section: str) -> Tuple[
        List[EventNode], List[TemporalEdge], Dict, Dict, List[str]
    ]:
        """Parse EVENTS section, extracting nodes and temporal edges."""
        events = []
        temporal_edges = []
        event_types = {}
        vn_classes = {}
        all_predicates = []

        lines = section.split('\n')
        position = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for event header: # T40 (word) -> variable
            if line.startswith('#') and '->' in line:
                match = re.match(r'#\s*(T\d+)\s+\((.+?)\)\s+->\s+(\w+)', line)
                if match:
                    event_id, text, variable = match.groups()

                    # Get context lines - ONLY DRS line to avoid duplicates
                    # (FOL and DRS contain the same relations)
                    context = []
                    j = i + 1
                    while j < len(lines) and j < i + 4:
                        ctx_line = lines[j].strip()
                        if ctx_line.startswith('#') and 'DRS:' in ctx_line:
                            context.append(ctx_line)
                            break  # Only need one DRS line
                        j += 1

                    # Extract attributes
                    attrs = self._extract_event_attributes('\n'.join(context))

                    # Extract temporal relations
                    temp_rels = self._extract_temporal_relations('\n'.join(context), variable)
                    temporal_edges.extend(temp_rels)

                    # Get VerbNet info
                    vn_class, predicates = self._get_verbnet_class(text)
                    vn_classes[variable] = vn_class
                    all_predicates.extend(predicates)

                    # Create event node
                    event = EventNode(
                        id=variable,
                        text=text,
                        event_id=event_id,
                        type=attrs.get('Type'),
                        tense=attrs.get('Tense'),
                        aspect=attrs.get('Aspect'),
                        vform=attrs.get('Vform'),
                        position=position
                    )
                    events.append(event)
                    event_types[variable] = attrs.get('Type', 'Unknown')
                    position += 1

            i += 1

        return events, temporal_edges, event_types, vn_classes, all_predicates

    def _extract_event_attributes(self, context: str) -> Dict[str, str]:
        """Extract attributes like Type, Tense, Aspect from FOL/DRS lines."""
        attrs = {}

        # Pattern: Attribute(var, value)
        pattern = r'(Type|Tense|Aspect|Vform|Pos)\([a-z]+,\s*([^)]+)\)'
        matches = re.findall(pattern, context)

        for attr_name, attr_value in matches:
            attrs[attr_name] = attr_value.strip()

        return attrs

    def _extract_temporal_relations(self, context: str, source_var: str) -> List[TemporalEdge]:
        """Extract temporal relations from FOL/DRS lines."""
        relations = []

        # Patterns for temporal relations
        patterns = [
            (r'occursBefore\(' + source_var + r',(\w+)\)', 'occursBefore'),
            (r'occursAfter\(' + source_var + r',(\w+)\)', 'occursAfter'),
            (r'overlaps\(' + source_var + r',(\w+)\)', 'overlaps'),
            (r'during\(' + source_var + r',(\w+)\)', 'during'),
        ]

        for pattern, rel_type in patterns:
            matches = re.findall(pattern, context)
            for target in matches:
                if target != 'now':  # Skip "completedBefore(x, now)"
                    relations.append(TemporalEdge(
                        source=source_var,
                        target=target,
                        relation=rel_type
                    ))

        return relations

    def _parse_actors_section(self, section: str) -> Tuple[List[ActorNode], set]:
        """Parse ACTORS section."""
        actors = []
        event_refs = set()  # IDs that are actually event references

        for line in section.split('\n'):
            line = line.strip()
            if line.startswith('#') and '->' in line:
                # Pattern: # T1 -> text
                match = re.match(r'#\s*(T\d+)\s+->\s+(.+)', line)
                if match:
                    actor_id, text = match.groups()

                    # Check if this is an event reference (verb-like text)
                    is_event = bool(re.match(r'^[a-z]+$', text.strip()) and
                                   len(text.strip()) < 20)

                    if is_event:
                        event_refs.add(actor_id)

                    actors.append(ActorNode(
                        id=actor_id,
                        text=text.strip(),
                        is_event_ref=is_event
                    ))

        return actors, event_refs

    def _parse_relations_section(
        self, section: str, event_refs: set
    ) -> Tuple[List[SemanticEdge], List[CoreferenceEdge]]:
        """Parse RELATIONS section, separating semantic and coreference edges."""
        semantic_edges = []
        coref_edges = []

        for line in section.split('\n'):
            line = line.strip()
            if line.startswith('#') and ' - ' in line:
                # Pattern: # T1 - relation - T40
                parts = line[1:].strip().split(' - ')
                if len(parts) == 3:
                    source, relation, target = [p.strip() for p in parts]

                    if relation == 'objIdentity':
                        # Coreference edge
                        coref_edges.append(CoreferenceEdge(
                            source=source,
                            target=target
                        ))
                    elif relation in ['agent', 'patient', 'theme', 'experiencer',
                                     'instrument', 'location', 'goal', 'source']:
                        # Semantic role edge
                        semantic_edges.append(SemanticEdge(
                            actor_id=source,
                            event_id=target,
                            role=relation
                        ))

        return semantic_edges, coref_edges


def process_single_story(
        extractor: DRSGraphExtractor,
        drs_path: str,
        text: str = ""
) -> Optional[Dict]:
    """Process a single story DRS file (for Track B inference)."""
    graph = extractor.parse_drs_file(drs_path)

    if graph is None:
        return None

    result = asdict(graph)
    result['text'] = text
    return result

def process_triplet(
    extractor: DRSGraphExtractor,
    drs_dir: str,
    idx: int,
    jsonl_item: Optional[Dict] = None
) -> Optional[Dict]:
    """Process a single triplet (anchor, story_a, story_b)."""

    # File paths
    anchor_path = os.path.join(drs_dir, f"{idx}_anchor_drs.txt")
    a_path = os.path.join(drs_dir, f"{idx}_a_drs.txt")
    b_path = os.path.join(drs_dir, f"{idx}_b_drs.txt")

    # Parse all three
    anchor_graph = extractor.parse_drs_file(anchor_path)
    a_graph = extractor.parse_drs_file(a_path)
    b_graph = extractor.parse_drs_file(b_path)

    if any(g is None for g in [anchor_graph, a_graph, b_graph]):
        return None

    result = {
        'idx': idx,
        'anchor': asdict(anchor_graph),
        'story_a': asdict(a_graph),
        'story_b': asdict(b_graph),
    }

    # Add metadata from original jsonl if available
    if jsonl_item:
        result['label'] = jsonl_item.get('text_a_is_closer', None)
        result['anchor_text'] = jsonl_item.get('anchor_text', '')
        result['text_a'] = jsonl_item.get('text_a', '')
        result['text_b'] = jsonl_item.get('text_b', '')

    return result


def preprocess_dataset(
    drs_dir: str,
    jsonl_path: Optional[str],
    output_path: str,
    max_instances: Optional[int] = None
) -> Dict[str, Any]:
    """
    Preprocess entire dataset to graph-structured JSONL.

    Args:
        drs_dir: Directory containing DRS files
        jsonl_path: Path to original JSONL with labels (optional)
        output_path: Where to save preprocessed JSONL
        max_instances: Limit number of instances (for testing)

    Returns:
        Statistics about the preprocessing
    """
    extractor = DRSGraphExtractor()

    # Load original JSONL if available
    jsonl_data = {}
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                jsonl_data[idx] = item

    # Find all triplets by looking for anchor files
    anchor_files = sorted([
        f for f in os.listdir(drs_dir)
        if f.endswith('_anchor_drs.txt')
    ])

    if max_instances:
        anchor_files = anchor_files[:max_instances]

    stats = {
        'total': len(anchor_files),
        'processed': 0,
        'skipped': 0,
        'avg_events': [],
        'avg_temporal_edges': [],
        'avg_actors': [],
        'avg_coref_edges': [],
        'avg_semantic_edges': [],
    }

    print(f"\nPreprocessing {len(anchor_files)} triplets from {drs_dir}")
    print(f"Output: {output_path}\n")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for anchor_file in anchor_files:
            # Extract index from filename
            idx = int(anchor_file.split('_')[0])

            # Process triplet
            result = process_triplet(
                extractor, drs_dir, idx,
                jsonl_data.get(idx)
            )

            if result is None:
                stats['skipped'] += 1
                continue

            # Write to output
            f_out.write(json.dumps(result) + '\n')
            stats['processed'] += 1

            # Collect stats
            for story_key in ['anchor', 'story_a', 'story_b']:
                graph = result[story_key]
                stats['avg_events'].append(graph['num_events'])
                stats['avg_temporal_edges'].append(graph['num_temporal_edges'])
                stats['avg_actors'].append(graph['num_actors'])
                stats['avg_coref_edges'].append(graph['num_coreference_edges'])
                stats['avg_semantic_edges'].append(graph['num_semantic_edges'])

            if stats['processed'] % 100 == 0:
                print(f"  Processed {stats['processed']}/{stats['total']}...")

    # Compute averages
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    stats['avg_events'] = safe_mean(stats['avg_events'])
    stats['avg_temporal_edges'] = safe_mean(stats['avg_temporal_edges'])
    stats['avg_actors'] = safe_mean(stats['avg_actors'])
    stats['avg_coref_edges'] = safe_mean(stats['avg_coref_edges'])
    stats['avg_semantic_edges'] = safe_mean(stats['avg_semantic_edges'])

    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {stats['processed']}/{stats['total']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"\nAverage per story:")
    print(f"  Events: {stats['avg_events']:.1f}")
    print(f"  Temporal edges: {stats['avg_temporal_edges']:.1f}")
    print(f"  Actors: {stats['avg_actors']:.1f}")
    print(f"  Coreference edges: {stats['avg_coref_edges']:.1f}")
    print(f"  Semantic edges: {stats['avg_semantic_edges']:.1f}")

    return stats


def inspect_sample(jsonl_path: str, idx: int = 0):
    """Inspect a single preprocessed sample."""
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i == idx:
                data = json.loads(line)
                break

    print(f"\n{'='*60}")
    print(f"SAMPLE {idx}")
    print(f"{'='*60}")

    if 'label' in data:
        print(f"Label: {'A is closer' if data['label'] else 'B is closer'}")

    for story_key in ['anchor', 'story_a', 'story_b']:
        graph = data[story_key]
        print(f"\n--- {story_key.upper()} ---")
        print(f"Events ({graph['num_events']}):")
        for e in graph['events'][:5]:  # Show first 5
            print(f"  {e['id']}: {e['text']} (type={e['type']}, tense={e['tense']})")
        if len(graph['events']) > 5:
            print(f"  ... and {len(graph['events'])-5} more")

        print(f"\nTemporal edges ({graph['num_temporal_edges']}):")
        for e in graph['temporal_edges'][:5]:
            print(f"  {e['source']} --{e['relation']}--> {e['target']}")
        if len(graph['temporal_edges']) > 5:
            print(f"  ... and {len(graph['temporal_edges'])-5} more")

        print(f"\nVerbNet classes: {list(graph['verbnet_classes'].items())[:5]}")
        print(f"Logic predicates: {graph['logic_predicates'][:10]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python drs_preprocessor.py <drs_dir> <output_jsonl> [original_jsonl]")
        print("\nExample:")
        print("  python drs_preprocessor.py ./drs_dev ./preprocessed_dev.jsonl ./dev_track_a.jsonl")
        sys.exit(1)

    drs_dir = sys.argv[1]
    output_path = sys.argv[2]
    jsonl_path = sys.argv[3] if len(sys.argv) > 3 else None

    stats = preprocess_dataset(drs_dir, jsonl_path, output_path)

    # Show a sample
    print("\n" + "="*60)
    print("SAMPLE OUTPUT")
    print("="*60)
    inspect_sample(output_path, 0)