# README


Currently in the process of fixing this.

There are a lot of hardcoded envrionemtn labels. This setup likely will fail at generating  general symbolic rules (as performed in run.sh) despite the current fixes



## To run 

1. place the following contents of this google drive zip file at in data/alfdata directory of this repository (link: https://drive.google.com/drive/folders/1zmOOAg9InkbXNwPxxqe3gzcIHwk2_FTr?usp=sharing)
2. inside run.sh set your designated api keys
3. use the run.sh file with "./run.sh"


## differences from original code
1. removed unecessary environment packages 
2. included alfworld data
3. minor fixes to some (but not all) hardcoded repository layouts in python code
4. fix some problems iwth prompt loading
5. initalized missing directories
6. 


## extra tips

- for each api key setting, you only need to input the api key for that specific model (i.e. if you use gemini as the set llm in main, you only need to use the api for google's gemini services_)