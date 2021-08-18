cat save/features.csv | sed -e 's/,/  /g' | tr  '\n' ' ' > Player-Data/Input-P$party-0
cat save/labels.csv   | sed -e 's/,/  /g' | tr  '\n' ' ' >> Player-Data/Input-P$party-0