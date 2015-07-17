import argparse
import pandas as pd

__author__ = 'chetannaik'


def main(experiment):
    features = pd.read_csv("output/" + experiment + "/features.tsv", sep="\t")
    predictions = pd.read_csv("output/" + experiment + "/predictions.tsv",
                              sep="\t")

    dfq = features.groupby("QUESTION")
    questions = dfq.groups.keys()
    df_columns = ['QUESTION', 'Q_UNDERGOER', 'Q_ENABLER', 'Q_TRIGGER',
                  'Q_RESULT', 'Q_UNDERSPECIFIED', 'ANSWER_CHOICE',
                  'ANSWER_SENTENCE', 'A_UNDERGOER', 'A_ENABLER', 'A_TRIGGER',
                  'A_RESULT', 'A_UNDERSPECIFIED', 'UNDERGOER_SCORE',
                  'ENABLER_SCORE', 'TRIGGER_SCORE', 'RESULT_SCORE',
                  'UNDERSPECIFIED_SCORE', 'CORRECT_ANSWER', 'PREDICTED_ANSWER',
                  'PREDICTION']
    result_df = pd.DataFrame(columns=df_columns)

    for question in questions:
        qdf = dfq.get_group(question)
        frame_len = len(qdf)
        predicted_answer = predictions[predictions["QUESTION"] == question][
            "MEAN_PREDICTION"].any()
        correct_answer = qdf.CORRECT_ANSWER.any()
        prediction = predicted_answer == correct_answer
        predicted_answer_df = pd.DataFrame([[predicted_answer] * frame_len],
                                           index=["PREDICTED_ANSWER"],
                                           columns=qdf.index.tolist())
        prediction_df = pd.DataFrame([[prediction] * frame_len],
                                     index=["PREDICTION"],
                                     columns=qdf.index.tolist())
        qdf = pd.concat([qdf.T, predicted_answer_df, prediction_df]).T
        result_df = pd.concat([result_df, qdf])
    result_df = result_df.reset_index(drop=True)
    result_df.columns = ['QUESTION', 'QUESTION UNDERGOER', 'QUESTION ENABLER',
                         'QUESTION TRIGGER', 'QUESTION RESULT',
                         'QUESTION UNDERSPECIFIED', 'ANSWER CHOICE',
                         'ANSWER SENTENCE', 'ANSWER UNDERGOER',
                         'ANSWER ENABLER', 'ANSWER TRIGGER', 'ANSWER RESULT',
                         'ANSWER UNDERSPECIFIED', 'UNDERGOER SCORE',
                         'ENABLER SCORE', 'TRIGGER SCORE', 'RESULT SCORE',
                         'UNDERSPECIFIED SCORE', 'CORRECT ANSWER',
                         'PREDICTED ANSWER', 'PREDICTION']
    result_df.to_csv("output/" + experiment + "/analysis.tsv", sep="\t")
    print ("\nGenerated output/" + experiment
           + "/analysis.tsv file for analysis.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate analysis tsv file.')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.experiment)
