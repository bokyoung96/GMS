"""
TEAM 고객의 미래를 새롭게

코드 목적: 모든 코드를 순서에 따라 실행합니다.

세부 사항:
main.py는 DATA STORAGE가 동일 경로에 존재하는 것을 가정합니다.

DATA STORAGE:
1. mirae_apy_itm.csv
2. mirae_cs.csv
3. mirae_mkt_idx.csv
4. cs_ticker_eikon.csv
"""
from loader import *
from loader_cs import *
from cs_analysis import *
from cs_analysis import *
from loader_cs_adj import *
from cs_ticker import *
from cs_preprocess import *
from cs_autoencoder import *


if __name__ == "__main__":
    # CS 데이터 1차 전처리 진행 및 저장
    customer_analysis = CustomerAnalysis()
    res = customer_analysis.get_convert()
    res.to_pickle("res.pkl")
    print("\n ***** res.pkl SAVED ***** \n")
    print("MOVING ON...")

    # 1차 전처리가 진행된 CS 데이터 카테고리 재분류 및 저장
    customer_loader = CustomerLoader()
    res_categories, res_categories_flatten = customer_loader.get_all(
        save_pkl="Y")
    print("MOVING ON...")

    # AST & TRS 카테고리에 속한 티커 데이터 전처리 및 저장
    item_type = ['ast', 'trs']
    for item in item_type:
        customer_ticker = CustomerTicker(item_type=item)
        res_ticker = customer_ticker.data_transfer()
        res_ticker.to_pickle('./res_categories/res_{}_adj.pkl'.format(item))
        print("\n ***** res_{}_adj.pkl SAVED *****".format(item))
        print("MOVING ON...")

    # CS 데이터 2차 전처리 진행 및 저장
    dir_name = "./res_pp_categories/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    item_type = ['ast', 'trs', 'acs', 'snp', 'rto']
    for item in item_type:
        customer_preprocess = CustomerPreprocess(item_type=item)
        res_preprocess = customer_preprocess.pp_load
        res_preprocess.to_pickle(
            './res_pp_categories/res_pp_{}.pkl'.format(item))
        print("\n ***** res_pp_{}.pkl SAVED *****".format(item))
        print("MOVING ON...")
    print("***** TASK COMPLETED *****")

    # 이후의 과정 서술
