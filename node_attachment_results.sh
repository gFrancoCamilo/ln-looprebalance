#!/bin/bash

cd src
for i in $(seq 1 10); do
#	python3 get-results.py node-attachment -c 10 &> ../results/node_attachment_results/progress
	python3 get-results.py node-attachment -t barabasi-albert -c 10 &> ../results/node_attachment_results/progress
	python3 get-results.py node-attachment -t watts-strogatz -c 10 &> ../results/node_attachment_results/progress
done
