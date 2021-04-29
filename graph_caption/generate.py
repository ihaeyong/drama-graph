import sys
import json
import inflection
import template
from pylanguagetool import api

LANGUAGE_TOOL_URL = 'https://languagetool.org/api/v2/'

def correct_text(text):
    result = api.check(text, lang='en-US', api_url=LANGUAGE_TOOL_URL)
    matches = [m for m in result['matches']]
    matches.sort(key=lambda x: x['offset'])
    chunks = []
    cursor = 0
    for match in matches:
        replacements = match['replacements']
        if not replacements:
            continue
        rep_value = replacements[0]['value']
        offset = match['offset']
        length = match['length']
        chunks.append(text[cursor:offset])
        chunks.append(rep_value)
        cursor = offset + length
    if cursor < len(text):
        chunks.append(text[cursor:])
    return ''.join(chunks)

def load_graph(node_iter):
    graph = {}
    for node in node_iter:
        node_type = node['type']
        node_id = node['id']
        if node_type not in graph:
            graph[node_type] = {}
        graph[node_type][node_id] = node
    return graph

def get_stdin_json_lines_iter():
    for line in sys.stdin:
        yield json.loads(line.strip())

def get_string_from_item_node(item_node):
    item_class = item_node['class']
    result = item_node['label']
    if result.startswith('object_'):
        result = result[len('object_'):]
    elif result.startswith('person_'):
        result = result[len('person_'):]
    result = inflection.underscore(result)
    if item_class == 'unknown':
        return None 
    elif item_class == 'person':
        result = inflection.humanize(result)
        result = inflection.titleize(result)
    elif item_class == 'video':
        result = 'the video'
    if item_class != 'person':
        result = inflection.humanize(result)
        result = result.lower()
    return result

def generate_prop_sentence_dict(graph, prop_node):
    classname = prop_node['classname']
    try:
        prop_template = template.property_template_dict[classname]
    except KeyError:
        prop_template = ''
    
    source_string = get_string_from_item_node(graph['item'][prop_node['source_item_id']])
    if source_string is None:
        return None
    target_string = get_string_from_item_node(graph['item'][prop_node['target_item_id']])
    if target_string is None:
        return None

    if 'relation_item_id' in prop_node and prop_node['relation_item_id'] is not None:
        relation_string = get_string_from_item_node(graph['item'][prop_node['relation_item_id'] ])
    else:
        relation_string = 'is related to'

    result_string = prop_template % {
        'source': source_string,
        'target': target_string,
        'relation': relation_string,
    }

    corrected_result_string = correct_text(result_string)

    return {
        'content': corrected_result_string,
        'time_start': prop_node['time_start'],
        'time_end': prop_node['time_end'],
    }

def get_sentence_dicts_iter(graph):
    for prop_id, prop_node in graph['property'].items():
        sentence_dict = generate_prop_sentence_dict(graph, prop_node)
        if sentence_dict is None:
            continue
        yield sentence_dict

def main():
    # Load the graph data from stdin. 
    json_lines_iter = get_stdin_json_lines_iter()
    graph = load_graph(json_lines_iter)
    for sentence_dict in get_sentence_dicts_iter(graph):
        print(json.dumps(sentence_dict))
    

if __name__ == '__main__':
    main()