import os

import argparse
import numpy as np

import config
import entailment
import utils

__author__ = 'chetannaik'


def get_question_frames(question, question_frames):
    """Question frame extractor.

    Args:
        question: A string representing the question without options.
        question_frames: Contents of question_frames.tsv file.

    Returns: A list of python dictionaries containing the the frames for the
        question extracted from question_frames.
    """
    q_sentences = set(question.strip().split('.'))
    q_sentences.add(question.strip())
    q_frames = list()
    for row in question_frames:
        q_frame = dict()
        if any(row[utils.QUESTION].strip() in q for q in q_sentences):
            q_frame[utils.UNDERGOER] = row[utils.UNDERGOER]
            q_frame[utils.ENABLER] = row[utils.ENABLER]
            q_frame[utils.TRIGGER] = row[utils.TRIGGER]
            q_frame[utils.RESULT] = row[utils.RESULT]
            q_frame[utils.UNDERSPECIFIED] = row[utils.UNDERSPECIFIED]
            q_frames.append(q_frame)
    return q_frames


def get_answer_frames(answer, process_db):
    """Answer frame extractor.

    Args:
        answer: A string representing the answer choice.
        process_db: Contents of process_frames.tsv file.


    Returns: a answer_frames (a list) of python dictionaries,
        containing frames for the answer with frame elements
        extracted from process_db.
    """
    answer_frames = list()
    for row in process_db:
        answer_frame = dict()
        if utils.get_lemma(answer) in row[utils.PROCESS].lower():
            answer_frame[utils.UNDERGOER] = row[utils.UNDERGOER]
            answer_frame[utils.ENABLER] = row[utils.ENABLER]
            answer_frame[utils.TRIGGER] = row[utils.TRIGGER]
            answer_frame[utils.RESULT] = row[utils.RESULT]
            answer_frame[utils.UNDERSPECIFIED] = row[utils.UNDERSPECIFIED]
            answer_frame[utils.DEFINITION] = row[utils.DEFINITION]
            answer_frames.append(answer_frame)
    return answer_frames


def ranker(question_frames, answer_choices, process_db):
    """Ranks the answer_choices by calling aligner.

    Args:
        question_frames: A list of python dictionaries containing question
            frames.
        answer_choices: A python list containing answer choices.
        process_db:  Contents of process_frames.tsv file.

    Returns: A ranked list of tuples containing answer choices and their
        scores.
    """
    answer_data = dict()
    for answer in answer_choices:
        answer_frames = get_answer_frames(answer, process_db)
        answer_scores = aligner(question_frames, answer_frames)
        answer_data[answer] = answer_scores
    return answer_data


def get_frame_directional_score(question_frame, answer_frame):
    frame_scores = dict()
    temp_scores = dict()
    temp_scores["FORWARD"] = dict()
    for frame_element in utils.FRAME_ELEMENTS:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if not q_element and not a_element:
            score = 0
        else:
            ret = entailment.get_ai2_textual_entailment(
                a_element, q_element)
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score = mean_a_score * confidence
        temp_scores["FORWARD"][frame_element] = (q_element, a_element,
                                                 score)

    temp_scores["BACKWARD"] = dict()
    for frame_element in utils.FRAME_ELEMENTS:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if not q_element and not a_element:
            score = 0
        else:
            ret = entailment.get_ai2_textual_entailment(
                q_element, a_element)
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score = mean_a_score * confidence
        temp_scores["BACKWARD"][frame_element] = (q_element, a_element,
                                                  score)

    best_score = 0
    best_direction = None
    for k, v in temp_scores.iteritems():
        direction = k
        score = sum(map(lambda x: x[2], v.values()))
        if score >= best_score:
            best_score = score
            best_direction = direction

    for frame_element in utils.FRAME_ELEMENTS:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if q_element and a_element:
            q_list = map(lambda x: x.strip(), q_element.split("|"))
            a_list = map(lambda x: x.strip(), a_element.split("|"))
            if utils.has_filter_keyword(q_list):
                frame_scores[frame_element] = ("", a_element, np.NaN)
            elif utils.has_filter_keyword(a_list):
                frame_scores[frame_element] = (q_element, "", np.NaN)
            else:
                frame_scores[frame_element] = \
                    temp_scores[best_direction][frame_element]
        elif q_element:
            frame_scores[frame_element] = (q_element, "", np.NaN)
        elif a_element:
            frame_scores[frame_element] = ("", a_element, np.NaN)
        else:
            frame_scores[frame_element] = ("", "", np.NaN)
    frame_scores[utils.DEFINITION] = answer_frame[utils.DEFINITION]
    return frame_scores


def get_role_directional_score(question_frame, answer_frame):
    frame_scores = dict()
    for frame_element in utils.FRAME_ELEMENTS:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if not q_element and not a_element:
            score = 0
        else:
            ret = entailment.get_ai2_textual_entailment(
                utils.remove_filter_words(a_element),
                utils.remove_filter_words(q_element))
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score1 = mean_a_score * confidence
            ret = entailment.get_ai2_textual_entailment(
                utils.remove_filter_words(q_element),
                utils.remove_filter_words(a_element))
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score2 = mean_a_score * confidence
            score = max(score1, score2)
        if q_element and q_element:
            if q_element in ["process", "processes"]:
                frame_scores[frame_element] = ("", a_element, np.NaN)
            else:
                frame_scores[frame_element] = (q_element, a_element, score)
        elif q_element:
            frame_scores[frame_element] = (q_element, "", np.NaN)
        elif a_element:
            frame_scores[frame_element] = ("", a_element, np.NaN)
        else:
            frame_scores[frame_element] = ("", "", np.NaN)
    frame_scores[utils.DEFINITION] = answer_frame[utils.DEFINITION]
    return frame_scores


def aligner(question_frames, answer_frames):
    """Aligns a question frame with a answer frame and calls entailment service
    to get a match score.

    Args:
        question_frames: A list of python dictionaries containing question
            frames.
        answer_frames: A list of python dictionaries containg answer frame
            elements.

    Returns: A number representing the match score of question frame with all
        the answer frames.
    """
    answer_scores = []
    for question_frame in question_frames:
        for answer_frame in answer_frames:
            if config.SCORE_DIRECTION_ABSTRACTION == "FRAME":
                frame_scores = get_frame_directional_score(question_frame,
                                                           answer_frame)
            else:
                frame_scores = get_role_directional_score(question_frame,
                                                          answer_frame)
            answer_scores.append(frame_scores)
    return answer_scores


def process_shard(shard_experiment):
    shard_num, experiment = shard_experiment
    ret_list = []
    _, process_db = utils.read_tsv(
        "data/" + experiment + "/frames.cv." + str(shard_num) + ".tsv",
        header=False)
    _, questions = utils.read_tsv(
        "data/" + experiment + "/question.list.cv." + str(
            shard_num) + ".tsv")
    _, question_frames = utils.read_tsv(
        "data/" + experiment + "/question.framepredict.cv." + str(
            shard_num) + ".tsv", header=False)

    row_string = utils.feature_string

    for num, row in enumerate(questions):
        if num > 0 and (num + 1) % 60 == 0:
            print "."
        else:
            print ".",
        question = row[utils.QUESTION]
        q_frames = get_question_frames(question, question_frames)
        answer_choices = [row[utils.OPTION_A], row[utils.OPTION_B],
                          row[utils.OPTION_C], row[utils.OPTION_D]]
        valid_answer_choices = filter(bool, answer_choices)
        answer_data = ranker(q_frames, valid_answer_choices, process_db)
        correct_answer = row[utils.ANS_MAP[row[utils.ANSWER]]]
        for answer, data in answer_data.iteritems():
            for roles in data:
                out_row = row_string.format(question,
                                            roles[utils.UNDERGOER][0],
                                            roles[utils.ENABLER][0],
                                            roles[utils.TRIGGER][0],
                                            roles[utils.RESULT][0],
                                            roles[utils.UNDERSPECIFIED][0],
                                            answer,
                                            roles[utils.DEFINITION],
                                            roles[utils.UNDERGOER][1],
                                            roles[utils.ENABLER][1],
                                            roles[utils.TRIGGER][1],
                                            roles[utils.RESULT][1],
                                            roles[utils.UNDERSPECIFIED][1],
                                            roles[utils.UNDERGOER][2],
                                            roles[utils.ENABLER][2],
                                            roles[utils.TRIGGER][2],
                                            roles[utils.RESULT][2],
                                            roles[utils.UNDERSPECIFIED][2],
                                            correct_answer)
                ret_list.append(out_row)
    return ret_list


def main(num_shards, experiment):
    utils.get_filter_words()
    if not os.path.exists("output/" + experiment):
        os.makedirs("output/" + experiment)
    fh = open("output/" + experiment + "/features.tsv", "wt")
    row_string = utils.feature_string

    header = row_string.format("QUESTION", "Q_UNDERGOER", "Q_ENABLER",
                               "Q_TRIGGER", "Q_RESULT", "Q_UNDERSPECIFIED",
                               "ANSWER_CHOICE", "ANSWER_SENTENCE",
                               "A_UNDERGOER", "A_ENABLER", "A_TRIGGER",
                               "A_RESULT", "A_UNDERSPECIFIED",
                               "UNDERGOER_SCORE", "ENABLER_SCORE",
                               "TRIGGER_SCORE", "RESULT_SCORE",
                               "UNDERSPECIFIED_SCORE", "CORRECT_ANSWER")
    fh.write(header + "\n")

    result = []
    for shard_num in range(num_shards):
        result.extend(process_shard((shard_num, experiment)))
    fh.write("\n".join(result))
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame alignment for QA.')
    parser.add_argument('--num_shards', type=int,
                        help='number of shards/folds')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.num_shards, args.experiment)
