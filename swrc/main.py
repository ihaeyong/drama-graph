import yaml
import os
import logging
from preprocessing.preprocessor import *
from knowledge_extraction.extractor import *

with open(os.path.join('config', 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

logger.info('Config: %s' % config)

subtitles_dir = config['preprocessing']['substitle_file']
mode = 'qa'

preprocessor = preprocessor(config, mode)
preprocessor.save_output()

extractor = extractor(config, preprocessor.output)
extractor.save_output()

