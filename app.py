import pickle
import numpy as np
import os
import json

import tornado.ioloop
import tornado.web



if not os.path.exists('model.pkl'):
    exit("Can't run without the model!")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Tornado!")

class predictHandler(tornado.web.RequestHandler):

    def post(self):
        params = self.request.arguments
        x = np.array(list(map(float, params['input'])))
        y = model.predict([x])[0]
        self.write(json.dumps({'prediction': y.item()}))
        self.finish()

if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/predictMnist", predictHandler),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
