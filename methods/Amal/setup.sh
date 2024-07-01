module use /soft/modulefiles 
module load conda; conda activate

conda env create -f environment.yml

unzip LA_ALL_2018.zip