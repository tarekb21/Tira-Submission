#!/usr/bin/env bash
# This wrapper ensures $inputDataset and $outputDir get passed through to your Python script.
exec python3 script.py -i "$inputDataset" -o "$outputDir"
