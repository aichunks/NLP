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
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_root', 'log', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('pretrain_dis_data_path', '', 'Data path for pretraining discriminator')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of pretrain/train/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
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
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
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


def restore_best_model():
    """Load best-model file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring best-model for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.global_variables()
                            if "Adagrad" not in v.name and 'discriminator' not in v.name and 'beta' not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "train")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to {}...".format(new_fname))
    new_saver = tf.train.Saver()  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")


def build_seqgan_graph(hps, vocab):
    print('Build generator graph...')
    with tf.device('/gpu:0'):
        generator = Generator(hps, vocab)
        generator.build_graph()

    print('Build discriminator graph...')
    with tf.device('/gpu:0'):
        # TODO: Settings in args
        dis_filter_sizes = [2, 3, 4, 5]
        dis_num_filters = [100, 100, 100, 100]
        dis_l2_reg_lambda = 0.2
        discriminator = Discriminator(sequence_length=hps.max_dec_steps,
                                      num_classes=2,
                                      vocab_size=FLAGS.vocab_size,
                                      embedding_size=hps.emb_dim,
                                      filter_sizes=dis_filter_sizes,
                                      num_filters=dis_num_filters,
                                      pretrained_path=False,
                                      l2_reg_lambda=dis_l2_reg_lambda)
    return generator, discriminator


def setup_training(mode, generator, discriminator, generator_batcher, discriminator_batcher):
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if FLAGS.restore_best_model:
        restore_best_model()
        return

    saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
    supervisor = tf.train.Supervisor(logdir=train_dir,
                                     is_chief=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                     save_model_secs=60,  # checkpoint every 60 secs
                                     global_step=generator.global_step)
    summary_writer = supervisor.summary_writer
    sess_context_manager = supervisor.prepare_or_wait_for_session(config=util.get_config())

    try:
        if mode == "pretrain":
            trainer.pretrain_generator(generator, generator_batcher, summary_writer, sess_context_manager)
        elif mode == "train":
            trainer.pretrain_discriminator(discriminator, sess_context_manager)
            trainer.adversarial_train(generator, discriminator, generator_batcher, discriminator_batcher,
                                      summary_writer, sess_context_manager)
        else:
            raise ValueError("Caught invalid value of mode!")
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        supervisor.stop()


def main(args):
    # prints a message if you've entered flags incorrectly
    if len(args) != 1:
        raise Exception("Problem with flags: %s" % args)

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode' or FLAGS.mode == 'predict':
        FLAGS.batch_size = FLAGS.beam_size


    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode != 'decode' and FLAGS.mode != 'predict':
        raise Exception("The single_pass flag should only be True in decode mode")
    hps = prepare_hps()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    generator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    discriminator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    if hps.mode == "pretrain" or hps.mode == "train":
        generator, discriminator = build_seqgan_graph(hps, vocab)
        setup_training(hps.mode, generator, discriminator, generator_batcher, discriminator_batcher)
    elif hps.mode == 'decode':
        # The model is configured with max_dec_steps=1 because we only ever run one step of
        # the decoder at a time (to do beam search).
        decode_model_hps = hps._replace(max_dec_steps=1)
        generator = Generator(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(generator, generator_batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of pretrain/train/decode")


def predict():

    if FLAGS.mode != "predict":
        print("Wrong Function")
        return

    FLAGS.batch_size = FLAGS.beam_size

    hps = prepare_hps()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)


    decode_model_hps = hps._replace(max_dec_steps=1)
    generator = Generator(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(generator, None, vocab)

    # text=" tHE PRESIDENT wILL REMAIN IN THE uNITED sTATES TO OVERSE THE AMERICAN RESPONSE TO sYRIA AND TO MONITOR DEVELOPMENTS AROUND THE WORLD. NOW YOU REMEMBER YESTERDAY THE PRESIDENT SENT OUT A SET OUT A TIMELINE OF 24 TO 48 HOURS AS TO WHEN HE MOULD HAKE HIS DECISION KNOWN AS IT RELATES TO THE us REsp ONSE TO THE CRISIS IN sYRIA SO HE COULD GET SOME MORE CLARITY ON THAT TODAY. buT sARAH sANDERS IS NOM HITTING THE FACT THAT THIS TRIP IS CANCELED ON SOME OF ThaT DECISION-MAKING cHRIS."
    # text='''  yOU HAVE SOME BREAKING NEHS ABOUT PRESIDENT CANCELING A SCHEDULED TRIP
    # # THAT HAS PLANNED LATER THIS HEEK. IET'S GO BACK TO THE WHITE hOUSE. nbc'S JEFF
    # # bENNETT, wHAT ARE YOU HEARING.? hEY CHRIS, PRESIDENT tRUMP HAS SET TO MAKE HIS FIRST VISIT OF HIS PRESIDENCY TO lATIN eMERICA LATER THIS HEEK. bUT HE'VE JUST LEARNED FROM WHITE hOUSE DRESS SECRETARY SARAH SANDERS THAT TRIP IS NOM CALLED OFF.
    # # hERE'S THE STATEMENT. SHE SENT OUT MOMENTS AGO PRESIDENT tRUMP HILL NOT ATTEND
    # # HE SUMMIT OF THE eMERICAS IN lIMA PERU OR TRAVEL TO bOGOTA COLOMBIA AS ORIGINALY SCHEDULED AT THE PRESIDENT'S REQUEST. tHE VICE PRESIDENT HILL TRAVEL IN INSTED'''
    text="-lrb- cnn -rrb- the palestinian authority officially became the 123rd member of the international criminal court on wednesday , a step that gives the court jurisdiction over alleged crimes in palestinian territories . the formal accession was marked with a ceremony at the hague , in the netherlands , where the court is based . the palestinians signed the icc 's founding rome statute in january , when they also accepted its jurisdiction over alleged crimes committed `` in the occupied palestinian territory , including east jerusalem , since june 13 , 2014 . '' later that month , the icc opened a preliminary examination into the situation in palestinian territories , paving the way for possible war crimes investigations against israelis . as members of the court , palestinians may be subject to counter-charges as well . israel and the united states , neither of which is an icc member , opposed the palestinians ' efforts to join the body . but palestinian foreign minister riad al-malki , speaking at wednesday 's ceremony , said it was a move toward greater justice . `` as palestine formally becomes a state party to the rome statute today , the world is also a step closer to ending a long era of impunity and injustice , '' he said , according to an icc news release . `` indeed , today brings us closer to our shared goals of justice and peace . '' judge kuniko ozaki , a vice president of the icc , said acceding to the treaty was just the first step for the palestinians . `` as the rome statute today enters into force for the state of palestine , palestine acquires all the rights as well as responsibilities that come with being a state party to the statute . these are substantive commitments , which can not be taken lightly , '' she said . rights group human rights watch welcomed the development . `` governments seeking to penalize palestine for joining the icc should immediately end their pressure , and countries that support universal acceptance of the court 's treaty should speak out to welcome its membership , '' said balkees jarrah , international justice counsel for the group . `` what 's objectionable is the attempts to undermine international justice , not palestine 's decision to join a treaty to which over 100 countries around the world are members . '' in january , when the preliminary icc examination was opened , israeli prime minister benjamin netanyahu described it as an outrage , saying the court was overstepping its boundaries . the united states also said it `` strongly '' disagreed with the court 's decision . `` as we have said repeatedly , we do not believe that palestine is a state and therefore we do not believe that it is eligible to join the icc , '' the state department said in a statement . it urged the warring sides to resolve their differences through direct negotiations . `` we will continue to oppose actions against israel at the icc as counterproductive to the cause of peace , '' it said . but the icc begs to differ with the definition of a state for its purposes and refers to the territories as `` palestine . '' while a preliminary examination is not a formal investigation , it allows the court to review evidence and determine whether to investigate suspects on both sides . prosecutor fatou bensouda said her office would `` conduct its analysis in full independence and impartiality . '' the war between israel and hamas militants in gaza last summer left more than 2,000 people dead . the inquiry will include alleged war crimes committed since june . the international criminal court was set up in 2002 to prosecute genocide , crimes against humanity and war crimes . cnn 's vasco cotovio , kareem khadder and faith karimi contributed to this report ."
    # text="marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . '' robin 's comments follow claims by two magazines , german daily bild and french paris match , of a cell phone video showing the harrowing final seconds from on board germanwings flight 9525 as it crashed into the french alps . all 150 on board were killed . paris match and bild reported that the video was recovered from a phone at the wreckage site . the two publications described the supposed video , but did not post it on their websites . the publications said that they watched the video , which was found by a source close to the investigation . `` one can hear cries of ` my god ' in several languages , '' paris match reported . `` metallic banging can also be heard more than three times , perhaps of the pilot trying to open the cockpit door with a heavy object . towards the end , after a heavy shake , stronger than the others , the screaming intensifies . then nothing . '' `` it is a very disturbing scene , '' said julian reichelt , editor-in-chief of bild online . an official with france 's accident investigation agency , the bea , said the agency is not aware of any such video . lt. col. jean-marc menichini , a french gendarmerie spokesman in charge of communications on rescue efforts around the germanwings crash site , told cnn that the reports were `` completely wrong '' and `` unwarranted . '' cell phones have been collected at the site , he said , but that they `` had n't been exploited yet . '' menichini said he believed the cell phones would need to be sent to the criminal research institute in rosny sous-bois , near paris , in order to be analyzed by specialized technicians working hand-in-hand with investigators . but none of the cell phones found so far have been sent to the institute , menichini said . asked whether staff involved in the search could have leaked a memory card to the media , menichini answered with a categorical `` no . '' reichelt told `` erin burnett : outfront '' that he had watched the video and stood by the report , saying bild and paris match are `` very confident '' that the clip is real . he noted that investigators only revealed they 'd recovered cell phones from the crash site after bild and paris match published their reports . `` that is something we did not know before . ... overall we can say many things of the investigation were n't revealed by the investigation at the beginning , '' he said . what was mental state of germanwings co-pilot ? german airline lufthansa confirmed tuesday that co-pilot andreas lubitz had battled depression years before he took the controls of germanwings flight 9525 , which he 's accused of deliberately crashing last week in the french alps . lubitz told his lufthansa flight training school in 2009 that he had a `` previous episode of severe depression , '' the airline said tuesday . email correspondence between lubitz and the school discovered in an internal investigation , lufthansa said , included medical documents he submitted in connection with resuming his flight training . the announcement indicates that lufthansa , the parent company of germanwings , knew of lubitz 's battle with depression , allowed him to continue training and ultimately put him in the cockpit . lufthansa , whose ceo carsten spohr previously said lubitz was 100 % fit to fly , described its statement tuesday as a `` swift and seamless clarification '' and said it was sharing the information and documents -- including training and medical records -- with public prosecutors . spohr traveled to the crash site wednesday , where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside . he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . menichini told cnn late tuesday that no visible human remains were left at the site but recovery teams would keep searching . french president francois hollande , speaking tuesday , said that it should be possible to identify all the victims using dna analysis by the end of the week , sooner than authorities had previously suggested . in the meantime , the recovery of the victims ' personal belongings will start wednesday , menichini said . among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board . check out the latest from our correspondents . the details about lubitz 's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and lubitz 's possible motive for downing the jet . a lufthansa spokesperson told cnn on tuesday that lubitz had a valid medical certificate , had passed all his examinations and `` held all the licenses required . '' earlier , a spokesman for the prosecutor 's office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot 's license . kumpa emphasized there 's no evidence suggesting lubitz was suicidal or acting aggressively before the crash . investigators are looking into whether lubitz feared his medical condition would cause him to lose his pilot 's license , a european government official briefed on the investigation told cnn on tuesday . while flying was `` a big part of his life , '' the source said , it 's only one theory being considered . another source , a law enforcement official briefed on the investigation , also told cnn that authorities believe the primary motive for lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems . lubitz 's girlfriend told investigators he had seen an eye doctor and a neuropsychologist , both of whom deemed him unfit to work recently and concluded he had psychological issues , the european government official said . but no matter what details emerge about his previous mental health struggles , there 's more to the story , said brian russell , a forensic psychologist . `` psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they were n't going to keep doing their job and they 're upset about that and so they 're suicidal , '' he said . `` but there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person 's problems . '' germanwings crash compensation : what we know . who was the captain of germanwings flight 9525 ? cnn 's margot haddad reported from marseille and pamela brown from dusseldorf , while laura smith-spark wrote from london . cnn 's frederik pleitgen , pamela boykoff , antonia mortensen , sandrine amiel and anna-maja rappard contributed to this report ."


    data=genrate_Input(text,hps,vocab)

    out=decoder.predict(data)
    print(out)

from data import article2ids

def genrate_Input(text,hps,vocab):
    class Data:
        pass
    input_data=Data()
    # Process the article
    article_words = text.lower().split()
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
    input_data.max_dec_steps=max(hps.max_dec_steps,enc_len/4)

    input_data.enc_batch[0] = enc_input
    input_data.enc_lens[0]=enc_len
    input_data.enc_batch_extend_vocab[0, :] = enc_input_extend_vocab[:]

    for j in range(enc_len):
        input_data.enc_padding_mask[0][j] = 1
    return input_data





import sys
if __name__ == '__main__':
    # logs=["--mode=decode","--data_path=./data/test.bin","--vocab_path=./data/vocab","--log_root=./log","--single_pass=True"]
    logs=["--mode=predict","--data_path=./data/test.bin","--vocab_path=./data/vocab","--log_root=./log","--single_pass=True"]
    for l in logs:
        sys.argv.append(l)
    # tf.app.run()
    predict()


# "new : tapes of germanwings sous-bois sous-bois crash into crash . french gendarmerie spokesman says he was n't aware of any video . prosecutor says he does n't think a `` unwarranted '' '' cell phones is `` unwarranted ."
# "french prosecutor say he believed to need to be sent to the criminal research . he was not aware of any video footage from board the plane . two magazines , german daily bild to immediately video footage ."
#

