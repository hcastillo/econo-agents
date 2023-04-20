@echo off

if %errorlevel%==0 (
 jupytext *.py --to notebook
 git add .
 git commit -a
 git push
)