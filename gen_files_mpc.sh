cat save/features.csv | sed -e 's/,/  /g' | tr  '\n' ' ' > Input-P0-0
cat save/labels.csv   | sed -e 's/,/  /g' | tr  '\n' ' ' >> Input-P0-0

cp Input-P0-0 Input-P1-0
