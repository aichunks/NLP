import time
import argparse
import torch
import msgpack
from tools.model import DocReaderModel
from tools.utils import str2bool
from DataPrepration import annotate, to_id, init
from train import BatchGen
from flask import Flask,request,jsonify
from flask_cors import CORS


"""
This script serves as a template to be modified to suit all possible testing environments, including and not limited
to files (json, xml, csv, ...), web service, databases and so on.
To change this script to batch model, simply modify line 70 from "BatchGen([model_in], batch_size=1, ...)" to
"BatchGen([model_in_1, model_in_2, ...], batch_size=batch_size, ...)".
"""


args={
    "model-file":'models/best_model.pt',
    "cuda":False,
    "meta":"SQuAD/meta.msgpack"
}


if args["cuda"]:
    checkpoint = torch.load(args["model-file"])
else:
    checkpoint = torch.load(args["model-file"], map_location=lambda storage, loc: storage)

state_dict = checkpoint['state_dict']
opt = checkpoint['config']
with open(args["meta"], 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
embedding = torch.Tensor(meta['embedding'])
opt['pretrained_words'] = True
opt['vocab_size'] = embedding.size(0)
opt['embedding_dim'] = embedding.size(1)
opt['pos_size'] = len(meta['vocab_tag'])
opt['ner_size'] = len(meta['vocab_ent'])
opt['cuda'] = args["cuda"]
BatchGen.pos_size = opt['pos_size']
BatchGen.ner_size = opt['ner_size']
model = DocReaderModel(opt, embedding, state_dict)
w2id = {w: i for i, w in enumerate(meta['vocab'])}
tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
init()


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/',methods = ['POST'])
def index():
    try:
        id_ = 0
        content = request.get_json()
        evidence=content['Evidence'].strip()
        question=content['Question'].strip()

        start_time = time.time()
        annotated = annotate(('interact-{}'.format(id_), evidence, question), False)
        model_in = to_id(annotated, w2id, tag2id, ent2id)
        model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=args["cuda"], evaluation=True)))
        prediction = model.predict(model_in)[0]
        end_time = time.time()
        print('Quetsion: {}'.format(question))
        print('Answer: {}'.format(prediction))
        print('Time: {:.4f}s'.format(end_time - start_time))
    except:
        return jsonify({'Error': "Please Enter in correct format by {'Evidence':'.....','Question': '...'}"})


    return jsonify({'result':prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
