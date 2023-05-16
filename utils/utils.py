import torch
import torch.nn.functional as F
import os
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import logging
from typing import List
from galaxy_datasets.shared import label_metadata
import numpy as np


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def match(df_1, df_2, pixel, df1_name):
    """
    match two catalog
    :param df_1:
    :param df_2:
    :return:
    """
    sdss = SkyCoord(ra=df_1.ra, dec=df_1.dec, unit=u.degree)
    decals = SkyCoord(ra=df_2.ra, dec=df_2.dec, unit=u.degree)
    idx, d2d, d3d = sdss.match_to_catalog_sky(decals)
    max_sep = pixel * 0.262 * u.arcsec
    distance_idx = d2d < max_sep
    sdss_matches = df_1.iloc[distance_idx]
    matches = idx[distance_idx]
    decal_matches = df_2.iloc[matches]
    test = sdss_matches.loc[:].rename(columns={"ra": "%s" % df1_name[0], "dec": "%s" % df1_name[1]})
    test.insert(0, 'ID', range(len(test)))
    decal_matches.insert(0, 'ID', range(len(decal_matches)))
    new_df = pd.merge(test, decal_matches, how="inner", on=["ID"])
    return new_df.drop("ID", axis=1)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def load_dir(dir) -> pd.DataFrame:
    local_files = os.listdir(dir)
    ra, dec = [], []
    for i in range(len(local_files)):
        if ".fits" in local_files[i]:
            t_ra, t_dec = float(local_files[i].split("_")[0]), float(local_files[i].split("_")[1].split(".fits")[0])
            ra.append(t_ra)
            dec.append(t_dec)
    return pd.DataFrame(list(zip(ra, dec)), columns=["ra", "dec"])


def answer_prob(out):
    output = []
    q1 = out[0] + out[1] + out[2]
    q2 = out[3] + out[4]
    q3 = out[5] + out[6]
    q4 = out[7] + out[8] + out[9]
    q5 = out[10] + out[11] + out[12] + out[13] + out[14]
    q6 = out[15] + out[16] + out[17]
    q7 = out[18] + out[19] + out[20]
    q8 = out[21] + out[22] + out[23]
    q9 = out[24] + out[25] + out[26] + out[27] + out[28] + out[29]
    q10 = out[30] + out[31] + out[32] + out[33]

    output.append(
        [out[0] / q1, out[1] / q1, out[2] / q1, out[3] / q2, out[4] / q2, out[5] / q3, out[6] / q3, out[7] / q4,
         out[8] / q4,
         out[9] / q4, out[10] / q5, out[11] / q5, out[12] / q5, out[13] / q5, out[14] / q5, out[15] / q6, out[16] / q6,
         out[17] / q6, out[18] / q7, out[19] / q7, out[20] / q7, out[21] / q8, out[22] / q8, out[23] / q8, out[24] / q9,
         out[25] / q9, out[26] / q9, out[27] / q9, out[28] / q9, out[29] / q9, out[30] / q10, out[31] / q10,
         out[32] / q10, out[33] / q10])
    return np.array(output)


def extract_questions_and_label_cols(question_answer_pairs):
    """
    Convenience wrapper to get list of questions and label_cols from a schema.
    Common starting point for analysis, iterating over questions, etc.

    Args:
        question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}

    Returns:
        list: all questions e.g. [Question('smooth-or-featured'), ...]
        list: label_cols (list of answer strings). See ``label_metadata.py`` for examples.
    """
    questions = list(question_answer_pairs.keys())
    label_cols = [q + answer for q, answers in question_answer_pairs.items() for answer in answers]
    return questions, label_cols


gz2_pairs = {
    'smooth_or_featured': ['_smooth', '_featured_or_disk', '_artifact'],
    'disk_edge_on': ['_yes', '_no'],
    'has_spiral_arms': ['_yes', '_no'],
    'bar': ['_strong', '_weak', '_no'],
    'bulge_size': ['_dominant', '_large', '_moderate', '_small', '_none'],
    'how_rounded': ['_round', '_in_between', '_cigar_shaped'],
    'edge_on_bulge': ['_boxy', '_none', '_rounded'],  # de
    'spiral_winding': ['_tight', '_medium', '_loose'],
    'spiral_arm_count': ['_1', '_2', '_3', '_4', '_more_than_4', '_cant_tell'],
    'merging': ['_none', '_minor_disturbance', '_major_disturbance', '_merger']  # de
}

gz2_questions, gz2_label_cols = extract_questions_and_label_cols(gz2_pairs)

gz2_and_decals_dependencies = {
    'smooth_or_featured': None,  # always asked
    'disk_edge_on': 'smooth_or_featured_featured_or_disk',
    'has_spiral_arms': 'smooth_or_featured_featured_or_disk',
    'bar': 'smooth_or_featured_featured_or_disk',
    'bulge_size': 'smooth_or_featured_featured_or_disk',
    'how_rounded': 'smooth_or_featured_smooth',
    'edge_on_bulge': 'disk_edge_on_yes',
    'spiral_winding': 'has_spiral_arms_yes',
    'spiral_arm_count': 'has_spiral_arms_yes',  # bad naming...
    'merging': None,

}


def softmax_output(input, question_index_groups):
    input_shape = input.shape
    batched_input = input.unsqueeze(0) if len(input_shape) == 1 else input

    result = torch.zeros_like(batched_input)

    for q_n in range(len(question_index_groups)):
        q_indices = question_index_groups[q_n]
        q_start = q_indices[0]
        q_end = q_indices[1]

        # 对输入的子数组进行softmax操作
        input_slice = batched_input[:, q_start:q_end+1]
        # print(input_slice)
        softmax_result = torch.softmax(input_slice, dim=1)
        # softmax_result = torch.softmax(input_slice, dim=1) * 99 + 1

        # 将softmax结果存储在与输入相同形状的数组中
        result[:, q_start:q_end+1] = softmax_result

    if len(input_shape) == 1:
        result = result.squeeze(0)

    return result


class Question():

    def __init__(self, question_text: str, answer_text: List, label_cols: List):
        """
        Class representing decision tree question.
        Requires ``label_cols`` as an input in order to find the index (vs. all questions and answers) of this question and each answer.

        Args:
            question_text (str): e.g. 'smooth-or-featured'
            answer_text (List): e.g. ['smooth', 'featured-or-disk]
            label_cols (List): list of all questions and answers e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', ...]
        """
        self.text = question_text
        self.answers = create_answers(self, answer_text,
                                      label_cols)  # passing a reference to self, will stay up-to-date

        self.start_index = min(a.index for a in self.answers)
        self.end_index = max(a.index for a in self.answers)
        assert [self.start_index <= a.index <= self.end_index for a in self.answers]

        self._asked_after = None

    @property
    def asked_after(self):
        return self._asked_after

    def __repr__(self):
        return f'{self.text}, indices {self.start_index} to {self.end_index}, asked after {self.asked_after}'


class Answer():

    def __init__(self, text, question, index):
        """
        Class representing decision tree answer.

        Each answer includes the answer text (often used as a column header),
        the corresponding question, and its index in ``label_cols`` (often used for slicing model outputs)

        Args:
            text (str): e.g. 'smooth-or-featured_smooth'
            question (Question): Question class for this answer
            index (int): index of answer in label_cols (0-33 for DECaLS)
        """
        self.text = text
        self.question = question

        self.index = index
        self._next_question = None

    @property
    def next_question(self):
        """

        Returns:
            Question: question that follows after this answer. None initially, added by ``set_dependancies``.
        """
        return self._next_question

    def __repr__(self):
        return f'{self.text}, index {self.index}'

    @property
    def pretty_text(self):
        """
        Returns:
            str: Nicely formatted text for plots etc
        """
        return self.text.replace('-', ' ').replace('_', ' ').title()


def create_answers(question: Question, answers_texts: List, label_cols: List):
    """
    Instantiate the Answer classes for a given ``Question``.
    Each answer includes the answer text (often used as a column header),
    the corresponding question, and its index in ``label_cols`` (often used for slicing model outputs)

    Args:
        question (Question): question to which to create answers e.g. Question(smooth-or-featured)
        answers_texts (List): answer strings e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk']
        label_cols (List): list of all questions and answers e.g. ['smooth-or-featured_smooth', 'smooth-or-featured_featured-or-disk', ...]

    Returns:
        List: of Answers to that question e.g. [Answer(smooth-or-featured_smooth), Answer(smooth-or-featured_featured-or-disk)]
    """
    question_text = question.text
    answers = []
    for answer_text in answers_texts:
        answers.append(
            Answer(
                text=question_text + answer_text,  # e.g. smooth-or-featured_smooth
                question=question,
                index=label_cols.index(question_text + answer_text)  # will hopefully throw value error if not found?
                # _next_question not set, set later with dependancies
            )
        )
    return answers


def set_dependencies(questions, dependencies):
    """
    Link each answer to question which follows, and vica versa.
    Acts inplace.

    Specifically, for every answer in every question, set answer._next question to refer to the Question which follows that answer.
    Then for that Question, set question._asked_after to that answer.

    Args:
        questions (List): of questions e.g. [Question('smooth-or-featured'), Question('edge-on-disk')]
        dependencies (dict): dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
    """

    for question in questions:
        prev_answer_text = dependencies[question.text]
        if prev_answer_text is not None:
            try:
                prev_answer = [a for q in questions for a in q.answers if a.text == prev_answer_text][
                    0]  # look through every answer, find those with the same text as "prev answer text" - will be exactly one match
            except IndexError:
                raise ValueError(f'{prev_answer_text} not found in dependencies')
            prev_answer._next_question = question
            question._asked_after = prev_answer


class Schema():
    def __init__(self, question_answer_pairs: dict, dependencies):
        """
        Relate the df label columns tor question/answer groups and to tfrecod label indices
        Requires that labels be continguous by question - easily satisfied

        Args:
            question_answer_pairs (dict): e.g. {'smooth-or-featured: ['_smooth, _featured-or-disk, ...], ...}
            dependencies (dict): dict mapping each question (e.g. disk-edge-on) to the answer on which it depends (e.g. smooth-or-featured_featured-or-disk)
        """
        logging.debug(f'Q/A pairs: {question_answer_pairs}')
        self.question_answer_pairs = question_answer_pairs
        _, self.label_cols = label_metadata.extract_questions_and_label_cols(question_answer_pairs)
        self.dependencies = dependencies
        """
        Be careful:
        - first entry should be the first answer to that question, by df column order
        - second entry should be the last answer to that question, similarly
        - answers in between will be included: these are used to slice
        - df columns must be contigious by question (e.g. not smooth_yes, bar_no, smooth_no) for this to work!
        """
        self.questions = [Question(question_text, answers_text, self.label_cols) for question_text, answers_text in
                          question_answer_pairs.items()]
        if len(self.questions) > 1:
            set_dependencies(self.questions, self.dependencies)

        assert len(self.question_index_groups) > 0
        assert len(self.questions) == len(self.question_index_groups)

    def get_answer(self, answer_text):
        """

        Args:
            answer_text (str): e.g. 'smooth-or-featured_smooth'

        Raises:
            ValueError: No answer with that answer_text found

        Returns:
            Answer: the answer with matching answer_text e.g. Answer('smooth-or-featured_smooth')
        """
        try:
            return [a for q in self.questions for a in q.answers if a.text == answer_text][
                0]  # will be exactly one match
        except IndexError:
            raise ValueError('Answer not found: ', answer_text)

    def get_question(self, question_text):
        """

        Args:
            question_text (str): e.g. 'smooth-or-featured'

        Raises:
            ValueError: No question with that question_text found

        Returns:
            Question: the question with matching question_text e.g. Question('smooth-or-featured')
        """
        try:
            return [q for q in self.questions if q.text == question_text][0]  # will be exactly one match
        except  IndexError:
            raise ValueError('Question not found: ', question_text)

    @property
    def question_index_groups(self):
        """

        Returns:
            Paired (tuple) integers of (first, last) indices of answers to each question, listed for all questions.
            Useful for slicing model predictions by question.
        """
        # start and end indices of answers to each question in label_cols e.g. [[0, 1]. [1, 3]]
        return [(q.start_index, q.end_index) for q in self.questions]

    @property
    def named_index_groups(self):
        """

        Returns:
            dict: mapping each question to the start and end index of its answers in label_cols, e.g. {Question('smooth-or-featured'): [0, 2], ...}
        """
        return dict(zip(self.questions, self.question_index_groups))

    def joint_p(self, prob_of_answers, answer_text):
        """
        Probability of the answer with ``answer_text`` being asked, given the (predicted) probability of every answer.
        Useful for estimating the relevance of an answer e.g. to ignore predictons for answers less than 50% likely to be asked.

        Broadcasts over batch dimension.

        Args:
            prob_of_answers (np.ndarray): prob. of each answer being asked, of shape (galaxies, answers) where the index of answers matches label_cols
            answer_text (str): which answer to find the prob. of being asked e.g. 'edge-on-disk_yes'

        Returns:
            np.ndarray: prob of that answer being asked e.g. 0.5 for 'edge-on-disk_yes' of prob of 'smooth-or-featured_featured-or-disk' is 0.5. Shape (galaxies).
        """
        assert prob_of_answers.ndim == 2  # batch, p. No 'per model', marginalise first
        # prob(answer) = p(that answer|that q asked) * p(that q_asked) i.e...
        # prob(answer) = p(that answer|that q asked) * p(answer before that q)
        answer = self.get_answer(answer_text)
        p_answer_given_question = prob_of_answers[:, answer.index]
        if all(np.isnan(p_answer_given_question)):
            logging.warning(
                f'All p_answer_given_question for {answer_text} ({answer.index}) are nan i.e. all fractions are nan - check that labels for this question are appropriate')

        question = answer.question
        prev_answer = question.asked_after
        if prev_answer is None:
            return p_answer_given_question
        else:
            p_prev_answer = self.joint_p(prob_of_answers, prev_answer.text)  # recursive
            return p_answer_given_question * p_prev_answer

    @property
    def answers(self):
        """

        Returns:
            list: all answers
        """
        answers = []
        for q in self.questions:
            for a in q.answers:
                answers.append(a)
        return answers


def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)

    net_out = model(input_images)
    pred = torch.mean(net_out, dim=0, keepdim=True).cpu().detach().numpy()
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return pred, epistemic, aleatoric


def get_uncertainty_per_batch(model, batch, T=15, normalized=False):
    batch_predictions = []
    net_outs = []
    batches = batch.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    preds = []
    epistemics = []
    aleatorics = []

    for i in range(T):  # for T batches
        net_out, _ = model(batches[i].cuda())
        net_outs.append(net_out)
        if normalized:
            prediction = F.softplus(net_out)
            prediction = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
        else:
            prediction = F.softmax(net_out, dim=1)
        batch_predictions.append(prediction)

    for sample in range(batch.shape[0]):
        # for each sample in a batch
        pred = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in net_outs], dim=0)
        pred = torch.mean(pred, dim=0)
        preds.append(pred)

        p_hat = torch.cat([a_batch[sample].unsqueeze(0) for a_batch in batch_predictions], dim=0).detach().cpu().numpy()
        p_bar = np.mean(p_hat, axis=0)

        temp = p_hat - np.expand_dims(p_bar, 0)
        epistemic = np.dot(temp.T, temp) / T
        epistemic = np.diag(epistemic)
        epistemics.append(epistemic)

        aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
        aleatoric = np.diag(aleatoric)
        aleatorics.append(aleatoric)

    epistemic = np.vstack(epistemics)  # (batch_size, categories)
    aleatoric = np.vstack(aleatorics)  # (batch_size, categories)
    preds = torch.cat([i.unsqueeze(0) for i in preds]).cpu().detach().numpy()  # (batch_size, categories)

    return preds, epistemic, aleatoric
