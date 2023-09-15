# GMS (2023 미래에셋증권 빅데이터 페스티벌)
고미새: 고객의 미래를 새롭게
<br/>
1. 예선(통과)
2. 본선(08/09 ~ 09/13)
<br/>

# GUIDELINE
안녕하세요. 2023년 미래에셋증권 빅데이터 페스티벌 팀 <고미새: 고객의 미래를 새롭게>입니다.<br/>
본 README 파일은 코드 실행에 가이드라인을 제시하고자 제작되었습니다.<br/>
<br/>
파일 실행 순서는 ORDER OF CODES를 참고해주세요.<br/>
1. ORDER OF CODES는 파일의 실행 순서와 각 파일에 대해 간략히 소개합니다.
2. 각 파일의 구성 요소는 [파일명 – 코드 설명 – 사용한 클래스 – 결과]로 구성됩니다.
3. [결과]의 경우, TYPE과 DATA로 나뉘며, TYPE은 코드 실행 시 최종 결과의 실행 / 저장 등 상태 여부, DATA는 저장된 데이터가 있을 경우 데이터의 이름을 나타냅니다.
4. 대괄호 main.py로 묶인 파일은 main.py 실행 시 자동으로 실행되며, 각 코드에 대한 결과(실행 / 저장 등)가 도출됩니다.
5. 대괄호 ETC로 묶인 파일은 main.py에 무관하게 실행되는 코드입니다.<br/> (예외: cs_shapley_analysis.py의 경우, main.py가 실행된 뒤 정상적으로 실행됩니다.)
6. cs_ticker_eikon_download.ipynb는 Eikon Refinitiv API를 활용해 각 종목의 재무 데이터를 다운로드하는 코드입니다.<br/> (Eikon Refinitiv API를 사용하기 위해서는 API Key가 요구되며, 이는 <고미새> 팀원 최보경, 민현하가 학생으로 소속된 KAIST 경영대학원에서 제공받았습니다. 이에 따라, Eikon Refinitiv API를 실행하는 코드는 <[빅데이터 / 고미새] 소스코드.zip>에 존재하나, 실행은 불가하며, 대신 cs_ticker_eikon.csv로 실행 결과가 저장된 csv 파일을 제공합니다.)
7. 사용된 특수 라이브러리는 requirements.txt에 기록해 두었습니다.
<br/>

# ORDER OF CODES

![Code flow chart](https://github.com/bokyoung96/GMS/assets/49546804/8c5684b3-1975-4255-be29-2915abc30598)

