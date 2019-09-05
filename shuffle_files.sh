#!/usr/bin/env bash

# Temp file to hold shuffled order
shuffile=$(mktemp)

# Create shuffled order
lines=$(wc -l < "$1")
digits=$(printf "%d" $lines | wc -c)
fmt=$(printf "%%0%d.0f" $digits)
seq -f "$fmt" $lines | shuf > $shuffile

# Shuffle each file in same way
for fname in "$@"; do
    paste $shuffile "$fname" | sort | cut -f 2- > "$fname.shuf"
done

# Clean up
rm $shuffile