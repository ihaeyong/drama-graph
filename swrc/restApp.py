
import os
import sys
sys.path.append('../')

from flask import Flask
from flask_restful import Api, Resource, reqparse

from DemoParser import Parser


app = Flask(__name__)
api = Api(app)

# import jpype
# jpype.attachThreadToJVM()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=False, default=1258)
args = parser.parse_args()

# In[1]:

demo_parser = Parser()

class WebService(Resource):
    def __init__(self):
        print()
    def post(self):
        try:
            req_parser = reqparse.RequestParser()
            req_parser.add_argument('script', type=str, action='append')
            req_parser.add_argument('mode', type=str)
            req_parser.add_argument('utter_info', type=str)
            args = req_parser.parse_args()
            print(args)

            result = demo_parser.parser(args['script'], args['mode'])
            if args['utter_info'] == 'False':
                del result['utters']

            return result, 200
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {'error':str(e)}

api.add_resource(WebService, '/vttDemo')
app.run(debug=True, host='0.0.0.0', port=args.port)

