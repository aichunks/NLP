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
    ar_len=1 if ar_len==0 else ar_len
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


def formatOutput(text):
    text_array=split_into_sentences(text)
    if len(text_array)<2:
        return text_array[0]
    text=" ".join(text_array[:-1])
    if "." in text_array[-1]:
        text+=" "+text_array[-1]
    return text

parm=loadModel()

def predict():
    text=" tHE PRESIDENT wILL REMAIN IN THE uNITED sTATES TO OVERSE THE AMERICAN RESPONSE TO sYRIA AND TO MONITOR DEVELOPMENTS AROUND THE WORLD. NOW YOU REMEMBER YESTERDAY THE PRESIDENT SENT OUT A SET OUT A TIMELINE OF 24 TO 48 HOURS AS TO WHEN HE MOULD HAKE HIS DECISION KNOWN AS IT RELATES TO THE us REsp ONSE TO THE CRISIS IN sYRIA SO HE COULD GET SOME MORE CLARITY ON THAT TODAY. buT sARAH sANDERS IS NOM HITTING THE FACT THAT THIS TRIP IS CANCELED ON SOME OF ThaT DECISION-MAKING cHRIS."
    # text = """It quickly became obvious that such an effort would require nothing less than Google-scale data and computing power. “I could try to give you some access to it,” Page told Kurzweil. “But it’s going to be very difficult to do that for an independent company.” So Page suggested that Kurzweil, who had never held a job anywhere but his own companies, join Google instead. It didn’t take Kurzweil long to make up his mind: in January he started working for Google as a director of engineering. “This is the culmination of literally 50 years of my focus on artificial intelligence,” he says.
    #
    # Kurzweil was attracted not just by Google’s computing resources but also by the startling progress the company has made in a branch of AI called deep learning. Deep-learning software attempts to mimic the activity in layers of neurons in the neocortex, the wrinkly 80 percent of the brain where thinking occurs. The software learns, in a very real sense, to recognize patterns in digital representations of sounds, images, and other data.
    #
    # The basic idea—that software can simulate the neocortex’s large array of neurons in an artificial “neural network”—is decades old, and it has led to as many disappointments as breakthroughs. But because of improvements in mathematical formulas and increasingly powerful computers, computer scientists can now model many more layers of virtual neurons than ever before.
    #
    # With this greater depth, they are producing remarkable advances in speech and image recognition. Last June, a Google deep-learning system that had been shown 10 million images from YouTube videos proved almost twice as good as any previous image recognition effort at identifying objects such as cats. Google also used the technology to cut the error rate on speech recognition in its latest Android mobile software. In October, Microsoft chief research officer Rick Rashid wowed attendees at a lecture in China with a demonstration of speech software that transcribed his spoken words into English text with an error rate of 7 percent, translated them into Chinese-language text, and then simulated his own voice uttering them in Mandarin. That same month, a team of three graduate students and two professors won a contest held by Merck to identify molecules that could lead to new drugs. The group used deep learning to zero in on the molecules most likely to bind to their targets.
    #
    # Google in particular has become a magnet for deep learning and related AI talent. In March the company bought a startup cofounded by Geoffrey Hinton, a University of Toronto computer science professor who was part of the team that won the Merck contest. Hinton, who will split his time between the university and Google, says he plans to “take ideas out of this field and apply them to real problems” such as image recognition, search, and natural-language understanding, he says.
    #
    # All this has normally cautious AI researchers hopeful that intelligent machines may finally escape the pages of science fiction. Indeed, machine intelligence is starting to transform everything from communications and computing to medicine, manufacturing, and transportation. The possibilities are apparent in IBM’s Jeopardy!-winning Watson computer, which uses some deep-learning techniques and is now being trained to help doctors make better decisions. Microsoft has deployed deep learning in its Windows Phone and Bing voice search.
    #
    # Extending deep learning into applications beyond speech and image recognition will require more conceptual and software breakthroughs, not to mention many more advances in processing power. And we probably won’t see machines we all agree can think for themselves for years, perhaps decades—if ever. But for now, says Peter Lee, head of Microsoft Research USA, “deep learning has reignited some of the grand challenges in artificial intelligence.”
    #
    # Building a Brain
    #
    # There have been many competing approaches to those challenges. One has been to feed computers with information and rules about the world, which required programmers to laboriously write software that is familiar with the attributes of, say, an edge or a sound. That took lots of time and still left the systems unable to deal with ambiguous data; they were limited to narrow, controlled applications such as phone menu systems that ask you to make queries by saying specific words.
    #
    # Neural networks, developed in the 1950s not long after the dawn of AI research, looked promising because they attempted to simulate the way the brain worked, though in greatly simplified form. A program maps out a set of virtual neurons and then assigns random numerical values, or “weights,” to connections between them. These weights determine how each simulated neuron responds—with a mathematical output between 0 and 1—to a digitized feature such as an edge or a shade of blue in an image, or a particular energy level at one frequency in a phoneme, the individual unit of sound in spoken syllables.
    #
    # Some of today’s artificial neural networks can train themselves to recognize complex patterns.
    #
    # Programmers would train a neural network to detect an object or phoneme by blitzing the network with digitized versions of images containing those objects or sound waves containing those phonemes. If the network didn’t accurately recognize a particular pattern, an algorithm would adjust the weights. The eventual goal of this training was to get the network to consistently recognize the patterns in speech or sets of images that we humans know as, say, the phoneme “d” or the image of a dog. This is much the same way a child learns what a dog is by noticing the details of head shape, behavior, and the like in furry, barking animals that other people call dogs.
    #
    # But early neural networks could simulate only a very limited number of neurons at once, so they could not recognize patterns of great complexity. They languished through the 1970s.
    #
    #
    # In the mid-1980s, Hinton and others helped spark a revival of interest in neural networks with so-called “deep” models that made better use of many layers of software neurons. But the technique still required heavy human involvement: programmers had to label data before feeding it to the network. And complex speech or image recognition required more computer power than was then available.
    #
    # Finally, however, in the last decade ­Hinton and other researchers made some fundamental conceptual breakthroughs. In 2006, Hinton developed a more efficient way to teach individual layers of neurons. The first layer learns primitive features, like an edge in an image or the tiniest unit of speech sound. It does this by finding combinations of digitized pixels or sound waves that occur more often than they should by chance. Once that layer accurately recognizes those features, they’re fed to the next layer, which trains itself to recognize more complex features, like a corner or a combination of speech sounds. The process is repeated in successive layers until the system can reliably recognize phonemes or objects.
    #
    # Like cats. Last June, Google demonstrated one of the largest neural networks yet, with more than a billion connections. A team led by Stanford computer science professor Andrew Ng and Google Fellow Jeff Dean showed the system images from 10 million randomly selected YouTube videos. One simulated neuron in the software model fixated on images of cats. Others focused on human faces, yellow flowers, and other objects. And thanks to the power of deep learning, the system identified these discrete objects even though no humans had ever defined or labeled them.
    #
    # What stunned some AI experts, though, was the magnitude of improvement in image recognition. The system correctly categorized objects and themes in the ­YouTube images 16 percent of the time. That might not sound impressive, but it was 70 percent better than previous methods. And, Dean notes, there were 22,000 categories to choose from; correctly slotting objects into some of them required, for example, distinguishing between two similar varieties of skate fish. That would have been challenging even for most humans. When the system was asked to sort the images into 1,000 more general categories, the accuracy rate jumped above 50 percent."""

    decoder, hps, vocab= parm
    data = genrate_Input(text, hps, vocab)
    print(data[0].enc_batch.shape)
    out="".join([decoder.predict(d) for d in data])
    out=formatOutput(out.replace("[UNK]",""))
    print("Output: ",  out)

# def predict():
#     decoder, hps, vocab= parm
#     while True:
#         text=input("Please Enter the text here.")
#         data=genrate_Input(text,hps,vocab)
#         out="".join([decoder.predict(d) for d in data])
#         out = out.replace("[UNK]", "")
#         print("Output: ",  out)


if __name__ == '__main__':
    predict()




