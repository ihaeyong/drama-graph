import yaml
import os
import logging
from preprocessing.preprocessor import *
from knowledge_extraction.extractor import *
from graph_maker.graph_maker import *
from background_knowledge.background import *

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        pass


with open(os.path.join('config', 'config.yaml'), 'r', encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

logger.info('Config: %s' % config)

output_dir = config['output_dir']
create_dir(output_dir)
create_dir(output_dir + '/graphs')


subtitles_dir = config['preprocessing']['substitle_file']
print('preprocessing..')
preprocessor = preprocessor(config, '')
preprocessor.save_output()
print('done..')

print('extracting knowledge..')
extractor = extractor(config, preprocessor.output)
extractor.save_output()
back_KB = background_knowledge(config)
print('done..')

print('building graph..')
graph_maker = graph_maker(config, extractor.output, back_KB.output)
print('done..')
