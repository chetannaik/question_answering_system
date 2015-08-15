import argparse

import pandas as pd

__author__ = 'chetannaik'


def main(experiment):
    features = pd.read_csv("output/" + experiment + "/features.tsv", sep="\t",
                           index_col=0)
    predictions = pd.read_csv("output/" + experiment + "/predictions.tsv",
                              sep="\t", index_col=0)

    dfq = features.groupby("QUESTION")
    questions = dfq.groups.keys()
    df_columns = ['QUESTION',
                  'ANSWER_CHOICE',
                  # 'ANSWER_SENTENCE',
                  'Q_UNDERGOER', 'A_UNDERGOER', 'UNDERGOER_SCORE',
                  'Q_ENABLER', 'A_ENABLER', 'ENABLER_SCORE',
                  'Q_TRIGGER', 'A_TRIGGER', 'TRIGGER_SCORE',
                  'Q_THEME', 'A_THEME', 'THEME_SCORE',
                  'Q_RESULT', 'A_RESULT', 'RESULT_SCORE',
                  'Q_MEDIUM', 'A_MEDIUM', 'MEDIUM_SCORE',
                  'Q_SOURCE', 'A_SOURCE', 'SOURCE_SCORE',
                  'Q_TARGET', 'A_TARGET', 'TARGET_SCORE',
                  'Q_LOCATION', 'A_LOCATION', 'LOCATION_SCORE',
                  'Q_DIRECTION', 'A_DIRECTION', 'DIRECTION_SCORE',
                  'CORRECT_ANSWER',
                  'PREDICTED_ANSWER',
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
    result_df.to_csv("output/" + experiment + "/analysis.tsv", sep="\t")
    print ("\nGenerated output/" + experiment
           + "/analysis.tsv file for analysis.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate analysis tsv file.')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.experiment)
