mkdir build 2> ./null
mkdir src/modulePython 2> ./null
mkdir output 2> ./null
mkdir matrixLabyrinth 2> ./null
rm -f ./null
chmod +x run.sh
if ! [ -e output/results.out ] 
then echo "ExperimentName ArrivalTime Dispersion" >> output/results.out
fi