"""
TEAM 고객의 미래를 새롭게

코드 목적: Shapley Value로 찾은 주요 특성들이 각 군집별로 어떠한 분포를 띠고 있는지 확인합니다.ㄷ
"""
from cs_shapley import *

cols = ['DMST_AST_EVAL_M1', 'DMST_AST_PCHS_M1', 'DMST_ITM_CNT_M1',
        'SEL_AMT_M1_2', 'SEL_ITM_CNT_M1_1', 'BUY_AMT_M1_2',
        'CONN_DYS_M1_2', 'MTS_DYS_M1_2', 'CONN_DYS_M1_3',
        'DMST_BUY_YM_DIFF', 'DMST_TR_MONTHS_CNT', 'AGE_TCD',
        'MONTHS_TR_RATIO', 'HLD_TR_RATIO', 'SWING_TR_RATIO',
        'LABEL']

data = pd.read_pickle('./res_pp_categories/res_pp_label.pkl')
res = data[cols]
res.to_pickle('./res_shapley.pkl')


def label_extraction(label: int) -> pd.DataFrame:
    return res[res['LABEL'] == label]


def label_mean(label: int):
    return label_extraction(label).mean(axis=0)


def label_random_extraction(rnd_num: int) -> pd.DataFrame:
    rnd_df = pd.DataFrame()
    for label in range(5):
        label_df = res[res['LABEL'] == label]
        rnd_df = pd.concat(
            [rnd_df, label_df.sample(n=rnd_num, random_state=42)])
    return rnd_df
