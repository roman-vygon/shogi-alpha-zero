"""
Encapsulates the worker which trains ShogiModels using game data from recorded games from a file.
"""
import os
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle

import numpy as np

from shogi_zero.agent.model_shogi import ShogiModel
from shogi_zero.config import Config
# from shogi_zero.env.shogi_env import canon_input_planes, is_black_turn, testeval
from shogi_zero.env.shogi_env import SfenInfo, CanonicalInput
from shogi_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from shogi_zero.lib.model_helper import load_best_model_weight

from keras.optimizers import Adam
from keras.callbacks import TensorBoard

logger = getLogger(__name__)

import os
from keras.utils import Sequence
import tensorflow as tf


def _parse_function(proto):
    feature_description = {
        'state': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)

    state = tf.io.parse_tensor(parsed_features['state'], out_type=tf.float32)
    state = tf.reshape(state, [44, 9, 9])

    policy = tf.io.parse_tensor(parsed_features['policy'], out_type=tf.float32)
    policy = tf.reshape(policy, [4845])

    value = tf.io.parse_tensor(parsed_features['value'], out_type=tf.float32)
    value = tf.reshape(value, [])

    return state, policy, value


class ShogiDataset:
    def __init__(self, folder, batch_size):
        files = tf.data.Dataset.list_files(f"{folder}/dataset*.tfrecord")
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=8)
        parsed_dataset = dataset.map(_parse_function)
        self.dataset = parsed_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def start(config: Config):
    """
    Helper method which just kicks off the optimization using the specified config
    :param Config config: config to use
    """
    return OptimizeWorker(config).start()


class OptimizeWorker:
    """
    Worker which optimizes a ShogiModel by training it on game data

    Attributes:
        :ivar Config config: config for this worker
        :ivar ShogiModel model: model to train
        :ivar dequeue,dequeue,dequeue dataset: tuple of dequeues where each dequeue contains game states,
            target policy network values (calculated based on visit stats
                for each state during the game), and target value network values (calculated based on
                    who actually won the game after that state)
        :ivar ProcessPoolExecutor executor: executor for running all of the training processes
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ShogiModel
        self.dataset = ShogiDataset('kif_dataset', 128)

    def start(self):
        """
        Load the next generation model from disk and start doing the training endlessly.
        """
        self.model = self.load_model()
        self.training()

    '''def training(self):
        """
        Does the actual training of the model, running it on game data. Endless.
        """
        self.compile_model()
        self.filenames = deque(get_game_data_filenames(self.config.resource))
        shuffle(self.filenames)
        total_steps = self.config.trainer.start_total_steps

        while True:
            self.fill_queue()
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            self.save_current_model()
            a, b, c = self.dataset
            while len(a) > self.config.trainer.dataset_size / 2:
                a.popleft()
                b.popleft()
                c.popleft()'''

    def training(self):
        """
        Does the actual training of the model, running it on game data. Endless.
        """
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps

        while True:
            tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=self.config.trainer.batch_size, histogram_freq=1)
            self.model.model.fit(
                self.dataset.dataset,
                epochs=self.config.trainer.epoch_to_checkpoint,
                shuffle=True,
                callbacks=[tensorboard_cb]
            )

            total_steps += 1
            self.save_current_model()

    def train_epoch(self, epochs):
        """
        Runs some number of epochs of training
        :param int epochs: number of epochs
        :return: number of datapoints that were trained on in total
        """
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02,
                             callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        """
        Compiles the model to use optimizer and loss function tuned for supervised learning
        """
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        """
        Saves the current model as the next generation model to the appropriate directory
        """
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def fill_queue(self):
        """
        Fills the self.dataset queues with data from the training dataset.
        """
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                logger.debug(f"loading data from {filename}")
                futures.append(executor.submit(load_data_from_file, filename))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    if y is not None:
                        x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file, filename))

    def collect_all_loaded_data(self):
        """

        :return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        """
        Loads the next generation model from the appropriate directory. If not found, loads
        the best known model.
        """
        model = ShogiModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    try:
        return convert_to_cheating_data(data)
    except KeyError as e:
        return None, None, None
    except TypeError as e:
        return None, None, None


def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    map_count_state = defaultdict(int)
    for aaa in data:
        state_sfen, policy, value = aaa
        sfen_info = SfenInfo(state_sfen)
        map_count_state[sfen_info.board] += 1
        same_state_count = map_count_state[sfen_info.board]
        if sfen_info.turn == 'w':
            sfen_info = sfen_info.get_flipped_sfen_info()

        canonical_input = CanonicalInput(sfen_info, same_state_count)
        state_planes = canonical_input.create()
        if sfen_info.turn == 'w':
            policy = Config.flip_policy(policy)

        move_number = int(state_sfen.split(' ')[3])
        value_certainty = min(5, move_number) / 5  # reduces the noise of the opening... plz train faster
        sl_value = value * value_certainty

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list,
                                                                                                           dtype=np.float32)
