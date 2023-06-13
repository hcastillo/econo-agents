@echo off

jupytext fluctuations.py --to notebook
git add .
git commit -a
git push

