"""
Splits data into test data (one block) and training data (remaining five block) 
for cross validation.
"""
import pandas as pd
import numpy as np

df1 = pd.read_csv("meg_behavdata_all_idx.csv", sep="\t")
df1 = df1.dropna(axis=0, how="any")

for i in range(2, 3):  # repetitive
    df2 = df1.query(f"subject == {i}")
    df3 = df2.query(
        'condition == "Repetitive"'
    )  # df2.loc[df2['condition'] == 'repetitive']

    df4 = pd.DataFrame(
        {
            "response": df3["response"],
            "target": df3["target"],
            "coherence": df3["coherence"],
            "session": df3["session"],
            "block": df3["block"],
            "idx": df3["idx"],
        }
    )

    df4 = df4[["block", "session", "coherence", "target", "response", "idx"]]
    data_session_1 = df4.query("session == 1")
    data_session_2 = df4.query("session == 2")
    data_session_3 = df4.query("session == 3")

    blocks_session_1 = np.unique(data_session_1["block"])
    blocks_session_2 = np.unique(data_session_2["block"])
    blocks_session_3 = np.unique(data_session_3["block"])

    def get_data(blocks_session, data_session):
        if len(blocks_session) >= 1:
            data_session_block_1 = data_session[
                data_session["block"] == blocks_session[0]
            ]
        else:
            data_session_block_1 = pd.DataFrame()
        if len(blocks_session) == 2:
            data_session_block_2 = data_session[
                data_session["block"] == blocks_session[1]
            ]
        else:
            data_session_block_2 = pd.DataFrame()
        return data_session_block_1, data_session_block_2

    data_session_1_block_1, data_session_1_block_2 = get_data(
        blocks_session_1, data_session_1
    )
    data_session_2_block_1, data_session_2_block_2 = get_data(
        blocks_session_2, data_session_2
    )
    data_session_3_block_1, data_session_3_block_2 = get_data(
        blocks_session_3, data_session_3
    )

    def get_train_test_folds(data_test_fold):
        blocks = [
            data_session_1_block_1,
            data_session_1_block_2,
            data_session_2_block_1,
            data_session_2_block_2,
            data_session_3_block_1,
            data_session_3_block_2,
        ]
        blocks = pd.concat(blocks)
        test_block = int(np.unique(data_test_fold.block))
        test_sess = int(np.unique(data_test_fold.session))
        test = blocks.query(f"session == {test_sess} & block == {test_block}")
        data_training_fold = blocks.drop(test.index)
        return data_test_fold, data_training_fold

    if not data_session_3_block_2.empty:
        data_test_fold_1, data_training_fold_1 = get_train_test_folds(
            data_test_fold=data_session_3_block_2
        )
    if not data_session_3_block_1.empty:
        data_test_fold_2, data_training_fold_2 = get_train_test_folds(
            data_test_fold=data_session_3_block_1
        )
    if not data_session_2_block_2.empty:
        data_test_fold_3, data_training_fold_3 = get_train_test_folds(
            data_test_fold=data_session_2_block_2
        )
    if not data_session_2_block_1.empty:
        data_test_fold_4, data_training_fold_4 = get_train_test_folds(
            data_test_fold=data_session_2_block_1
        )
    if not data_session_1_block_2.empty:
        data_test_fold_5, data_training_fold_5 = get_train_test_folds(
            data_test_fold=data_session_1_block_2
        )
    if not data_session_1_block_1.empty:
        data_test_fold_6, data_training_fold_6 = get_train_test_folds(
            data_test_fold=data_session_1_block_1
        )

    for data_training_fold, fold_num in zip(
        [
            data_training_fold_1,
            data_training_fold_2,
            data_training_fold_3,
            data_training_fold_4,
            data_training_fold_5,
            data_training_fold_6,
        ],
        range(1, 7),
    ):
        try:
            data_training_fold.to_csv(
                f"P{i}_repetitive_meg_behav_unique_blocks_training_fold_{fold_num}_idx.csv",
                sep="\t",
                header=True,
                encoding="utf-8",
            )
            data_training_fold.drop(columns="idx").to_csv(
                f"P{i}_repetitive_meg_behav_unique_blocks_training_fold_{fold_num}.csv",
                sep="\t",
                header=False,
                encoding="utf-8",
                index=False,
            )
        except:
            pass

    for data_test_fold, fold_num in zip(
        [
            data_test_fold_1,
            data_test_fold_2,
            data_test_fold_3,
            data_test_fold_4,
            data_test_fold_5,
            data_test_fold_6,
        ],
        range(1, 7),
    ):
        try:
            data_test_fold.to_csv(
                f"P{i}_repetitive_meg_behav_unique_blocks_test_fold_{fold_num}_idx.csv",
                sep="\t",
                header=True,
                encoding="utf-8",
            )
            data_test_fold.drop(columns="idx").to_csv(
                f"P{i}_repetitive_meg_behav_unique_blocks_test_fold_{fold_num}.csv",
                sep="\t",
                header=False,
                encoding="utf-8",
                index=False,
            )
        except:
            pass
