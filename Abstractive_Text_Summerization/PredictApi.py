# -*-coding:utf-8 -*-
import os
from collections import namedtuple
from batcher import Batcher
from data import Vocab
from generator import Generator
from discriminator import Discriminator
from decode import BeamSearchDecoder
import trainer as trainer
import util,numpy as np
import tensorflow as tf
from flask import Flask,request,jsonify

tf.logging.set_verbosity(tf.logging.INFO)

# Data dirs
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles.\
                           Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', './data/vocab', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_root', './log', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('pretrain_dis_data_path', '', 'Data path for pretraining discriminator')

# Important settings
tf.app.flags.DEFINE_string('mode', 'predict', 'must be one of pretrain/train/decode/predict')
tf.app.flags.DEFINE_boolean('single_pass', True,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, \
                            i.e. take the current checkpoint, and use it to produce one summary for each example in \
                            the  dataset, write the summaries to file and then get ROUGE scores for the whole dataset. \
                             If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, \
                             use it to produce summaries for randomly-chosen examples and log the results to screen, \
                             indefinitely.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 32, 'minibatch size')
tf.app.flags.DEFINE_integer('dis_batch_size', 256, 'batch size for pretrain discriminator')
tf.app.flags.DEFINE_integer('max_enc_steps', 500, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 30,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. \
                            If the vocabulary file contains fewer words than this number, or if this number is \
                            set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_integer('rollout', 24, 'Size of rollout number')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('seqgan', True, 'If False disable seqgan')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train \
                            WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. \
                             i.e. to reproduce the results in the ACL paper, turn this off for most of training then \
                              turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss. If zero, then no incentive \
                           to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be \
                             used for further training. Useful for early stopping, or if your training checkpoint \
                              has become corrupted with e.g. NaN values.')
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
FLAGS = tf.app.flags.FLAGS


def prepare_hps():
    hparam_list = ['mode', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
                   'coverage', 'cov_loss_wt', 'pointer_gen', 'seqgan', 'rollout', 'lr',
                   'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hparam_list:
            hps_dict[key] = val.value
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    return hps
from data import article2ids
import re

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def genrateArray(text,max_len):
    min_extra=.2
    length=len(text.split())
    ar_len=length/max_len
    ar_len=int(ar_len)+1 if ar_len-int(ar_len)>min_extra else int(ar_len)
    arrMax_len=length/ar_len+ar_len
    arr_sent=[]
    sent,sent_len="",0
    for sentence in split_into_sentences(text):
        sent_len+=len(sentence.split())
        if(sent_len>arrMax_len):
            arr_sent.append(sent)
            sent,sent_len="",len(sentence.split())
        sent+=sentence+" "
    if(len(arr_sent)==0):
        arr_sent.append(sent)
    return arr_sent


def genrate_Input(text,hps,vocab):
    class Data:
        pass
    # Process the article
    article_words = text.lower()
    # for large text divide into equal parts
    article_wordsArray=genrateArray(article_words,hps.max_enc_steps)
    input_dataArray=[]
    for article_sent in article_wordsArray:
        article_words = article_sent.split()
        input_data=Data()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        enc_len = len(article_words)  # store the length after truncation but before padding
        enc_input = [vocab.word2id(w) for w in article_words]  # list of word ids; OOVs are represented by the id for UNK token
        enc_input_extend_vocab, article_oovs = article2ids(article_words, vocab)

        max_enc_seq_len = enc_len
        input_data.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        input_data.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        input_data.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        input_data.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        input_data.max_art_oovs = len(article_oovs)
        input_data.batch_reward = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        input_data.batch_rouge_reward = np.zeros((hps.batch_size, 1), dtype=np.float32)
        input_data.art_oovs = [article_oovs ]
        input_data.min_dec_steps = enc_len/10
        input_data.max_dec_steps=min(hps.max_dec_steps,enc_len/2)

        input_data.enc_batch[0] = enc_input
        input_data.enc_lens[0]=enc_len
        input_data.enc_batch_extend_vocab[0, :] = enc_input_extend_vocab[:]
        for j in range(enc_len):
            input_data.enc_padding_mask[0][j] = 1
        input_dataArray.append(input_data)
    return input_dataArray


def loadModel():
    FLAGS.batch_size = FLAGS.beam_size
    hps = prepare_hps()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    decode_model_hps = hps._replace(max_dec_steps=1)
    generator = Generator(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(generator, None, vocab)
    return decoder,hps,vocab

parm=loadModel()

def formatOutput(text):
    text_array=split_into_sentences(text)
    if len(text_array)<2:
        return text_array[0]
    text=" ".join(text_array[:-1])
    if "." in text_array[-1]:
        text+=" "+text_array[-1]
    return text


app = Flask(__name__)
@app.route('/',methods = ['POST'])
def predict():
    decoder, hps, vocab = parm
    try:
        content = request.get_json()
        evidence=content['Evidence'].strip()
        if len(evidence.split())<40:
            return jsonify({'result': "Too small String for summerization"})
        data = genrate_Input(evidence, hps, vocab)
        summary = "".join([decoder.predict(d)+" " for d in data])
        summary = formatOutput(summary.replace("[UNK]", ""))
    except:
        return jsonify({'Error': "Please Enter in correct format by {'Evidence':'.....'}"})
    return jsonify({'result':summary})


if __name__ == '__main__':
    app.run()


