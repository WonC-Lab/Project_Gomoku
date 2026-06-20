@echo off
chcp 65001 > nul
echo ==========================================
echo       깃허브 자동 업로드 시작합니다.
echo ==========================================

:: 현재 날짜와 시간 가져오기 (커밋 메시지용)
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set year=%datetime:~0,4%
set month=%datetime:~4,2%
set day=%datetime:~6,2%
set hour=%datetime:~8,2%
set minute=%datetime:~10,2%
set second=%datetime:~12,2%
set commit_msg=Auto-upload: %year%-%month%-%day% %hour%:%minute%:%second%

echo.
echo [1/3] 변경된 파일들을 추가(add) 중...
git add .

echo.
echo [2/3] 변경 사항 커밋(commit) 중...
git commit -m "%commit_msg%"

echo.
echo [3/3] 깃허브 원격 저장소에 업로드(push) 중...
git push origin main

echo.
echo ==========================================
echo  업로드가 완료되었습니다. (3초 후 자동 종료)
echo ==========================================
timeout /t 3
