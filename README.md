
# relief-project
Solution for an assignment of Analysis of Algorithms course. The objective is to choose only the necessary features from a data set with [Relief Algorithm](https://en.wikipedia.org/wiki/Relief_(feature_selection)). Relief Algorithm is applied to data set concurrently using the [Master/slave model](https://en.wikipedia.org/wiki/Master/slave_(technology)). Communication between processors is established with Message Passing Interface. 


## Requirements
- OpenMPI library
-  `g++`


## Building / Running
```
git clone https://github.com/omarr09/relief-project
cd https://github.com/omarr09/relief-project
```
To compile:
```
make
```
To compile and run:
```
NP=<np> ARG=<file> make run
```
where \<np\> is the number of processors to be used and  \<file\> is the path of the input file.


## Usage
Input data set consists of instances. Each instance have a feature array and a class (either 0 or 1). First line of the input file should have *processor count*. The second line should have *feature count of each instance*, *iteration count of the algorithm*, *number of top features needed*. The rest of the input file should list instances, one instance per line.

After running the program as explained above, the results of each process and the result of the master processor will be printed to standard output. 