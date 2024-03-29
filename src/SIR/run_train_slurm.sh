#!/bin/bash
#SBATCH --job-name=RCC_TEST
#SBATCH --ntasks=1
#SBATCH -A genacc_q
#SBATCH --nodes=1   




conda init bash
#this makes
source ~/.bashrc
# this is how it actually finds the conda environment you activate
conda activate /gpfs/home/aa18dg/.conda/envs/new_torch/
#you can also do source activate /gpfs/home/aa18dg/.conda/envs/new_torch/, and sometimes the conda command
#doing "conda list" shows the list of modules installed in that environment. 
#doing "printenv" which is a linux command, shows all the ENVIRONMENT VARIABLES you have.
conda info --envs
#this "--envs" command is very important, and it shows you which environment you're on right now
# it has better performance with "srun python3" instead of python, but you can experiment with or without srun

srun python3 Train_CDF.py -index 0 
srun python3 Train_CDF.py -index 1 
srun python3 Train_CDF.py -index 2 
srun python3 Train_CDF.py -index 3 
srun python3 Train_CDF.py -index 4 
srun python3 Train_CDF.py -index 5 
srun python3 Train_CDF.py -index 6 
srun python3 Train_CDF.py -index 7 
srun python3 Train_CDF.py -index 8 
srun python3 Train_CDF.py -index 9 
srun python3 Train_CDF.py -index 10 
srun python3 Train_CDF.py -index 11 
srun python3 Train_CDF.py -index 12 
srun python3 Train_CDF.py -index 13 
srun python3 Train_CDF.py -index 14 
srun python3 Train_CDF.py -index 15 
srun python3 Train_CDF.py -index 16 
srun python3 Train_CDF.py -index 17 
srun python3 Train_CDF.py -index 18 
srun python3 Train_CDF.py -index 19 
srun python3 Train_CDF.py -index 20 
srun python3 Train_CDF.py -index 21 
srun python3 Train_CDF.py -index 22 
srun python3 Train_CDF.py -index 23 
srun python3 Train_CDF.py -index 24 
srun python3 Train_CDF.py -index 25 
srun python3 Train_CDF.py -index 26 
srun python3 Train_CDF.py -index 27 
srun python3 Train_CDF.py -index 28 
srun python3 Train_CDF.py -index 29 
srun python3 Train_CDF.py -index 30 
srun python3 Train_CDF.py -index 31 
srun python3 Train_CDF.py -index 32 
srun python3 Train_CDF.py -index 33 
srun python3 Train_CDF.py -index 34 
srun python3 Train_CDF.py -index 35 
srun python3 Train_CDF.py -index 36 
srun python3 Train_CDF.py -index 37 
srun python3 Train_CDF.py -index 38 
srun python3 Train_CDF.py -index 39 
srun python3 Train_CDF.py -index 40 
srun python3 Train_CDF.py -index 41 
srun python3 Train_CDF.py -index 42 
srun python3 Train_CDF.py -index 43 
srun python3 Train_CDF.py -index 44 
srun python3 Train_CDF.py -index 45 
srun python3 Train_CDF.py -index 46 
srun python3 Train_CDF.py -index 47 
srun python3 Train_CDF.py -index 48 
srun python3 Train_CDF.py -index 49 
srun python3 Train_CDF.py -index 50 
srun python3 Train_CDF.py -index 51 
srun python3 Train_CDF.py -index 52 
srun python3 Train_CDF.py -index 53 
srun python3 Train_CDF.py -index 54 
srun python3 Train_CDF.py -index 55 
srun python3 Train_CDF.py -index 56 
srun python3 Train_CDF.py -index 57 
srun python3 Train_CDF.py -index 58 
srun python3 Train_CDF.py -index 59 
srun python3 Train_CDF.py -index 60 
srun python3 Train_CDF.py -index 61 
srun python3 Train_CDF.py -index 62 
srun python3 Train_CDF.py -index 63 
srun python3 Train_CDF.py -index 64 
srun python3 Train_CDF.py -index 65 
srun python3 Train_CDF.py -index 66 
srun python3 Train_CDF.py -index 67 
srun python3 Train_CDF.py -index 68 
srun python3 Train_CDF.py -index 69 
srun python3 Train_CDF.py -index 70 
srun python3 Train_CDF.py -index 71 
srun python3 Train_CDF.py -index 72 
srun python3 Train_CDF.py -index 73 
srun python3 Train_CDF.py -index 74 
srun python3 Train_CDF.py -index 75 
srun python3 Train_CDF.py -index 76 
srun python3 Train_CDF.py -index 77 
srun python3 Train_CDF.py -index 78 
srun python3 Train_CDF.py -index 79 
srun python3 Train_CDF.py -index 80 
srun python3 Train_CDF.py -index 81 
srun python3 Train_CDF.py -index 82 
srun python3 Train_CDF.py -index 83 
srun python3 Train_CDF.py -index 84 
srun python3 Train_CDF.py -index 85 
srun python3 Train_CDF.py -index 86 
srun python3 Train_CDF.py -index 87 
srun python3 Train_CDF.py -index 88 
srun python3 Train_CDF.py -index 89 
srun python3 Train_CDF.py -index 90 
srun python3 Train_CDF.py -index 91 
srun python3 Train_CDF.py -index 92 
srun python3 Train_CDF.py -index 93 
srun python3 Train_CDF.py -index 94 
srun python3 Train_CDF.py -index 95 
srun python3 Train_CDF.py -index 96 
srun python3 Train_CDF.py -index 97 
srun python3 Train_CDF.py -index 98 
srun python3 Train_CDF.py -index 99 
srun python3 Train_CDF.py -index 100 
srun python3 Train_CDF.py -index 101 
srun python3 Train_CDF.py -index 102 
srun python3 Train_CDF.py -index 103 
srun python3 Train_CDF.py -index 104 
srun python3 Train_CDF.py -index 105 
srun python3 Train_CDF.py -index 106 
srun python3 Train_CDF.py -index 107 
srun python3 Train_CDF.py -index 108 
srun python3 Train_CDF.py -index 109 
srun python3 Train_CDF.py -index 110 
srun python3 Train_CDF.py -index 111 
srun python3 Train_CDF.py -index 112 
srun python3 Train_CDF.py -index 113 
srun python3 Train_CDF.py -index 114 
srun python3 Train_CDF.py -index 115 
srun python3 Train_CDF.py -index 116 
srun python3 Train_CDF.py -index 117 
srun python3 Train_CDF.py -index 118 
srun python3 Train_CDF.py -index 119 
srun python3 Train_CDF.py -index 120 
srun python3 Train_CDF.py -index 121 
srun python3 Train_CDF.py -index 122 
srun python3 Train_CDF.py -index 123 
srun python3 Train_CDF.py -index 124 
srun python3 Train_CDF.py -index 125 
srun python3 Train_CDF.py -index 126 
srun python3 Train_CDF.py -index 127 
srun python3 Train_CDF.py -index 128 
srun python3 Train_CDF.py -index 129 
srun python3 Train_CDF.py -index 130 
srun python3 Train_CDF.py -index 131 
srun python3 Train_CDF.py -index 132 
srun python3 Train_CDF.py -index 133 
srun python3 Train_CDF.py -index 134 
srun python3 Train_CDF.py -index 135 
srun python3 Train_CDF.py -index 136 
srun python3 Train_CDF.py -index 137 
srun python3 Train_CDF.py -index 138 
srun python3 Train_CDF.py -index 139 
srun python3 Train_CDF.py -index 140 
srun python3 Train_CDF.py -index 141 
srun python3 Train_CDF.py -index 142 
srun python3 Train_CDF.py -index 143 
srun python3 Train_CDF.py -index 144 
srun python3 Train_CDF.py -index 145 
srun python3 Train_CDF.py -index 146 
srun python3 Train_CDF.py -index 147 
srun python3 Train_CDF.py -index 148 
srun python3 Train_CDF.py -index 149 
srun python3 Train_CDF.py -index 150 
srun python3 Train_CDF.py -index 151 
srun python3 Train_CDF.py -index 152 
srun python3 Train_CDF.py -index 153 
srun python3 Train_CDF.py -index 154 
srun python3 Train_CDF.py -index 155 
srun python3 Train_CDF.py -index 156 
srun python3 Train_CDF.py -index 157 
srun python3 Train_CDF.py -index 158 
srun python3 Train_CDF.py -index 159 
srun python3 Train_CDF.py -index 160 
srun python3 Train_CDF.py -index 161 
srun python3 Train_CDF.py -index 162 
srun python3 Train_CDF.py -index 163 
srun python3 Train_CDF.py -index 164 
srun python3 Train_CDF.py -index 165 
srun python3 Train_CDF.py -index 166 
srun python3 Train_CDF.py -index 167 
srun python3 Train_CDF.py -index 168 
srun python3 Train_CDF.py -index 169 
srun python3 Train_CDF.py -index 170 
srun python3 Train_CDF.py -index 171 
srun python3 Train_CDF.py -index 172 
srun python3 Train_CDF.py -index 173 
srun python3 Train_CDF.py -index 174 
srun python3 Train_CDF.py -index 175 
srun python3 Train_CDF.py -index 176 
srun python3 Train_CDF.py -index 177 
srun python3 Train_CDF.py -index 178 
srun python3 Train_CDF.py -index 179 
srun python3 Train_CDF.py -index 180 
srun python3 Train_CDF.py -index 181 
srun python3 Train_CDF.py -index 182 
srun python3 Train_CDF.py -index 183 
srun python3 Train_CDF.py -index 184 
srun python3 Train_CDF.py -index 185 
srun python3 Train_CDF.py -index 186 
srun python3 Train_CDF.py -index 187 
srun python3 Train_CDF.py -index 188 
srun python3 Train_CDF.py -index 189 
srun python3 Train_CDF.py -index 190 
srun python3 Train_CDF.py -index 191 
srun python3 Train_CDF.py -index 192 
srun python3 Train_CDF.py -index 193 
srun python3 Train_CDF.py -index 194 
srun python3 Train_CDF.py -index 195 
srun python3 Train_CDF.py -index 196 
srun python3 Train_CDF.py -index 197 
srun python3 Train_CDF.py -index 198 
srun python3 Train_CDF.py -index 199 
srun python3 Train_CDF.py -index 200 