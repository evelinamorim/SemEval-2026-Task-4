from typing import List, Dict, Tuple, Union
import re
import spacy

try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    spacy.cli.download("xx_ent_wiki_sm")
    nlp = spacy.load("xx_ent_wiki_sm")


from graph import Event, Participant
from utils import get_instance_id
from read_data import load_drs_instances


def score_canonical_name_nlp(name: str, nlp_model: Union[spacy.Language, None] = None) -> tuple:
    """
    Retorna uma tupla para ordenação. Maior score = melhor candidato canônico.
    """
    num_words = len(name.split())
    num_chars = len(name.strip())
    has_upper = any(c.isupper() for c in name)

    if nlp_model is None:
        is_likely_pronoun = (num_words <= 2 and num_chars < 10 and not has_upper)
        return (not is_likely_pronoun, num_words, num_chars)

    doc = nlp_model(name)
    has_pronoun = any(token.pos_ == "PRON" for token in doc)
    has_entity = len(doc.ents) > 0

    return (
        not has_pronoun,
        has_entity,
        num_words,
        num_chars
    )


def extract_triplets(file_name: str, resolve_coreference: bool = True) -> List[Tuple]:
    """
    Extrai triplas do arquivo DRS.

    :param file_name: caminho do arquivo DRS
    :param resolve_coreference: se True, resolve correferências; se False, mantém entidades separadas
    :return: lista de triplas
    """
    triplets = []
    event_map = {}
    actor_map = {}
    events_with_relations = set()

    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    section = None

    # Ler ACTORS
    for line in lines:
        line = line.strip()

        if line.startswith('» EVENTS'):
            section = 'EVENTS'
            continue
        elif line.startswith('» RELATIONS'):
            section = 'RELATIONS'
            continue
        elif line.startswith('» ACTORS'):
            section = 'ACTORS'
            continue

        if section == 'ACTORS':
            match = re.match(r"# (T\d+) (?:->|-&gt;) (.+)", line)
            if match:
                tid, name = match.groups()
                actor_map[tid] = name.strip()

    # Processar correferência SE solicitado
    canonical_map = {}
    if resolve_coreference:
        # Union-Find para correferência
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Construir grafo de correferência
        section = None
        for line in lines:
            line = line.strip()

            if line.startswith('» RELATIONS'):
                section = 'RELATIONS'
                continue
            elif line.startswith('» ACTORS'):
                section = 'ACTORS'
                continue

            if section == 'RELATIONS':
                match = re.match(r"# (T\d+) - (\w+) - (T\d+)", line)
                if match:
                    left, relation, right = match.groups()
                    if relation == "objIdentity":
                        union(left, right)

        # Criar mapeamento canônico
        clusters = {}
        for tid in actor_map:
            root = find(tid)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(tid)

        for root, members in clusters.items():
            candidates = [(tid, actor_map[tid]) for tid in members]
            best_tid, best_name = max(
                candidates,
                key=lambda x: score_canonical_name_nlp(x[1], nlp)
            )

            for tid in members:
                canonical_map[tid] = Participant(best_name)
    else:
        # Sem resolução: cada entidade mantém seu nome original
        for tid, name in actor_map.items():
            canonical_map[tid] = Participant(name)

    # Extrair eventos e relações
    section = None
    for line in lines:
        line = line.strip()

        if line.startswith('» EVENTS'):
            section = 'EVENTS'
            continue
        elif line.startswith('» RELATIONS'):
            section = 'RELATIONS'
            continue
        elif line.startswith('» ACTORS'):
            section = 'ACTORS'
            continue

        if section == 'EVENTS':
            match = re.match(r"# (T\d+) \((.*?)\) (?:->|-&gt;) (\w+)", line)
            if match:
                tid, verb, var = match.groups()
                event_map[tid] = Event(verb)

        elif section == 'RELATIONS':
            match = re.match(r"# (T\d+) - (\w+) - (T\d+)", line)
            if match:
                left, relation, right = match.groups()

                # Se não resolver correferência, incluir relações objIdentity como triplas
                if relation == "objIdentity" and not resolve_coreference:
                    left_label = canonical_map.get(left, Participant(actor_map.get(left, left)))
                    right_label = canonical_map.get(right, Participant(actor_map.get(right, right)))
                    triplets.append((left_label, relation, right_label))
                    continue
                elif relation == "objIdentity":
                    # Já processado na resolução de correferência
                    continue

                if right in event_map:
                    entity_label = canonical_map.get(left, Participant(actor_map.get(left, left)))
                    triplets.append((event_map[right], relation, entity_label))
                    events_with_relations.add(right)

    # Adicionar eventos isolados
    for tid, event in event_map.items():
        if tid not in events_with_relations:
            triplets.append((event, "occurs", None))

    return triplets


def process_triplets(
        file_drs: List[Tuple[str, str, str]],
        resolve_coreference: bool = True
) -> Dict[int, Tuple[List[Tuple], List[Tuple], List[Tuple]]]:
    """
    Processa arquivos DRS e extrai triplas.

    :param file_drs: lista de tuplas (anchor_file, a_file, b_file)
    :param resolve_coreference: se True, resolve correferências usando objIdentity;
                                 se False, mantém todas as entidades separadas
    :return: dicionário com triplas (anchor, a, b) para cada instância
    """
    result = {}

    for anchor_file, a_file, b_file in file_drs:
        anchor_triples = extract_triplets(anchor_file, resolve_coreference)
        a_triples = extract_triplets(a_file, resolve_coreference)
        b_triples = extract_triplets(b_file, resolve_coreference)

        instance_id = get_instance_id(anchor_file)
        result[instance_id] = (anchor_triples, a_triples, b_triples)

    return result

if __name__ == "__main__":
    instances = load_drs_instances("sample_drs/")
    result = process_triplets(instances, True)
    print(len(result[0][0]))




