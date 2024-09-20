@echo off

jupytext fluctuations.py --to notebook
git add *.py
git commit -a
git push

