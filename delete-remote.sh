#!/bin/bash

# git ls-remote origin | grep "refs/heads/users/" | awk '{print $2}' | sed 's/refs\/heads\///' | \
#   xargs -n 1 git push origin --delete



git ls-remote origin | grep "refs/heads/users/" | awk '{print $2}' | sed 's/refs\/heads\///' | wc -l


