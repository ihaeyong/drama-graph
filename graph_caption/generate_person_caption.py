import json
import jsonlines
import argparse
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

def read_jsonl(input_file):
    with jsonlines.open(input_file, 'r') as f:
        lines = [l for l in f]
    return lines

def write_jsonl(data, fname):
    with jsonlines.open(fname, mode="w") as writer:
        for d in data:
            writer.write(d)

# def get_stdin_json_lines_iter():
#     for line in sys.stdin:
#         yield json.loads(line.strip())

def generate_prop_sentence_dict(json_obj):
    source_type = json_obj['source_type']
    if source_type == "triples":
        result_string = json_obj['subject'] + ' ' + json_obj['relation'] + ' ' + json_obj['object']
    elif source_type == "frames":
        args_string = [list(x.values())[0] for x in json_obj['args']]
        args_string.insert(1, json_obj['lu'])
        result_string = ' '.join(args_string)
    else:
        return None

    corrected_result_string = correct_text(result_string)

    return {
        'content': corrected_result_string,
        'scene': json_obj['scene'],
    }


def get_sentence_dicts(json_lines_iter):
    sentence_dicts = []
    for json_obj in json_lines_iter:
        sentence_dict = generate_prop_sentence_dict(json_obj)
        if sentence_dict is None:
            continue
        sentence_dicts.append(sentence_dict)
    
    return sentence_dicts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str, required=True)
    parser.add_argument('--output_file', default=None, type=str, required=True)

    args = parser.parse_args()

    json_lines_iter = read_jsonl(args.input_file)
    sentence_dicts = get_sentence_dicts(json_lines_iter)
    
    write_jsonl(sentence_dicts, args.output_file)

main()