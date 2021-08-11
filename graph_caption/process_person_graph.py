import json
import jsonlines
import argparse

def load_json(input_file):
    with open(input_file, encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    return data

def write_jsonl(data, fname):
    with jsonlines.open(fname, mode="w") as writer:
        for d in data:
            writer.write(d)

def parse_input_graph(json_data):
    script_json = []
    for scene in json_data:
        # iterate through triples
        triples = json_data[scene]['triples']
        for triple in triples:
            triple_json = triple
            triple_json['source_type'] = 'triples'
            triple_json['scene'] = scene
            script_json.append(triple_json)
        
        frames = json_data[scene]['frames']
        for frame in frames:
            frames_json = frame
            frames_json['source_type'] = 'frames'
            frames_json['scene'] = scene
            script_json.append(frames_json)

    return script_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, type=str, required=True)
    parser.add_argument('--output_file', default=None, type=str, required=True)

    args = parser.parse_args()
    
    data = load_json(args.input_file)
    output_jsons = parse_input_graph(data)

    write_jsonl(output_jsons, args.output_file)

main()