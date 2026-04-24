#!/bin/bash

# wget http://snap.stanford.edu/jodie/reddit.csv -P data/
# wget http://snap.stanford.edu/jodie/wikipedia.csv -P data/
# wget http://snap.stanford.edu/jodie/mooc.csv -P data/
# wget http://snap.stanford.edu/jodie/lastfm.csv -P data/


curl -o data/reddit.csv http://snap.stanford.edu/jodie/reddit.csv
curl -o data/wikipedia.csv http://snap.stanford.edu/jodie/wikipedia.csv
curl -o data/mooc.csv http://snap.stanford.edu/jodie/mooc.csv
curl -o data/lastfm.csv http://snap.stanford.edu/jodie/lastfm.csv
