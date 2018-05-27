@echo off
set /p msg=Enter MSG:
git add -A
git commit -m "%msg%"
git push origin master 