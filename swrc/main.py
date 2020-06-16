import yaml
import os
import logging
from preprocessing.preprocessor import *
from knowledge_extraction.extractor import *
from graph_maker.graph_maker import *
from background_knowledge.background import *

with open(os.path.join('config', 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

logger.info('Config: %s' % config)

subtitles_dir = config['preprocessing']['substitle_file']

preprocessor = preprocessor(config)
preprocessor.save_output()

extractor = extractor(config, preprocessor.output)
extractor.save_output()

back_KB = background_knowledge(config)

graph_maker = graph_maker(config, extractor.output, back_KB.output)

