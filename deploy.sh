#!/bin/bash
cp dashboard/app.py app.py
git add .
git commit -m "update"
git push
echo "Deployed!"
