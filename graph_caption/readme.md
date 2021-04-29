# Knowledge Graph Sentence Generator

This is a Python application that generates natural language sentences based on a knowledge graph built from a video file.

## How to use

Before following the instruction below, 
please make sure to install Python 3 on your environment. 

### Set up

#### Setting up virtual environment

Set up Python3 virtual environment to install dependencies.

```bash
knowledge-graph-sentence-generator$ python3 -m venv ./env
```

#### Activate the virtual environment

Activate your virtual environment for Python.

```bash
knowledge-graph-sentence-generator$ source ./env/bin/activate
```

#### Install dependencies

Install Python dependencies.

```bash
(env)knowledge-graph-sentence-generator$ pip install -r requirements.txt
```

### Generate the sentences

#### Run main script

Provide the graph data in JSON Lines format via standard input to the `generate.py` script.

Input data should follow the format of an output data from the project [knowledge-graph-builder](https://github.com/uilab-vtt/knowledge-graph-builder). 

If you want to have an output file consist of sentence JSON objects, 
make a file with data coming out via standard output from the script. 

```bash
(env)knowledge-graph-sentence-generator$ python generate.py < graph.jsonl > output_sentences.jsonl
```

# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
 