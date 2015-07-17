__author__ = 'chetannaik'


# EXPERIMENTS = ["SRLManual", "SRLQA", "SRLQADSv1", "SRLQADSv2", "SRLQADSv2Top5", "MATE_AutoTrigger"]
EXPERIMENTS = ["MATE_AutoTrigger"]

SHARDS = {"SRLManual": 1,
          "SRLQA": 5,
          "SRLQADSv1": 5,
          "SRLQADSv2": 5,
          "SRLQADSv2Top5": 5,
          "MATE_AutoTrigger": 1}

# SCORE_TYPES = ["ROW_SCORE", "COLUMN_SCORE", "MAX_ROLE_SCORE"]
SCORE_TYPE = "MAX_ROLE_SCORE"

# SCORE_DIRECTION_ABSTRACTION = ["FRAME", "ROLE"]
SCORE_DIRECTION_ABSTRACTION = "ROLE"

GROUP_WEIGHTS = {5: 1, 4: 0.8, 3: 0.24, 2: 0.072, 1: 0.0216, 0: 0}

SCORES = ['UNDERGOER_SCORE', 'ENABLER_SCORE', 'TRIGGER_SCORE', 'RESULT_SCORE',
          'UNDERSPECIFIED_SCORE']

ADF_COLUMNS = ['QUESTION', 'ANSWER_CHOICE', 'UNDERGOER_SCORE', 'ENABLER_SCORE',
               'TRIGGER_SCORE', 'RESULT_SCORE', 'UNDERSPECIFIED_SCORE',
               'CORRECT_ANSWER']