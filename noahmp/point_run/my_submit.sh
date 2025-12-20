#PBS -l walltime=24:00:00
#PBS -P 11004044
#PBS -l select=1:ncpus=128:mem=256gb
###PBS -q ic007
#PBS -e my_submit.sh_err.txt
#PBS -o my_submit.sh_out.txt
#PBS -N noahmp_test

module load cray-mpich
module load cray-pals
module load cray-netcdf cray-hdf5
module list

cd $HOME/scratch/NOAHMP/testrun
aprun -n 24 ./hrldas.exe > output_2012.txt
