#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <set>
using namespace std;
constexpr int bufferSize = 1024;        // size of the buffer array for string operations

// A class to represent an instance
class Instance {
public:
    vector<double> features;            // feature array of the instance
    int cls;                            // class of the instance
};

// Functions related to relief algorithm
namespace Relief {
    // Auxiliary functions that are only required for relief function
    namespace {
        // Calculates Manhattan distance between two vectors. Assumes vector sizes are equal
        double manhattanDistance(const vector<double>& vec1, const vector<double>& vec2) {
            double result = 0;
            for(int i=0; i<vec1.size(); i++) {
                result += abs(vec1[i] - vec2[i]);
            }
            return result;
        }

        // diff function. Takes minimum and maximum values of features as parameters
        double diff(int index, const Instance& i1, const Instance& i2, const vector<double>& minimums, const vector<double>& maximums) {
            return abs((i1.features[index]-i2.features[index])) / (maximums[index] - minimums[index]);
        }

        // Finds nearest hit and miss for instances[index] and returns their indexes
        pair<int, int> findNearestHitAndMiss(const vector<Instance>& instances, int index) {
            double minHitDistance = numeric_limits<double>::max();
            double minMissDistance = numeric_limits<double>::max();
            int minHitIndex = -1;
            int minMissIndex = -1;

            for(int i=0; i<instances.size(); i++) {
                if(i==index) continue;
                double distance = manhattanDistance(instances[i].features, instances[index].features);
                if(instances[i].cls == instances[index].cls && distance < minHitDistance) {
                    minHitDistance = distance;
                    minHitIndex = i;
                }
                if(instances[i].cls != instances[index].cls && distance < minMissDistance) {
                    minMissDistance = distance;
                    minMissIndex = i;
                }
            }

            return {minHitIndex, minMissIndex};
        }

        // Finds minimum and maximum values of all features in instances and returns them
        pair<vector<double>, vector<double>> getMinAndMax(const vector<Instance>& instances) {
            int fCount = instances[0].features.size();
            vector<double> minimums(fCount, numeric_limits<double>::max());
            vector<double> maximums(fCount, numeric_limits<double>::min());

            for(int i=0; i<instances.size(); i++) {
                for(int j=0; j<fCount; j++) {
                    double val = instances[i].features[j];
                    if(val < minimums[j]) minimums[j] = val;
                    if(val > maximums[j]) maximums[j] = val;
                }
            }

            return {move(minimums), move(maximums)};
        }
    }

    // Relief algorithm.
    // Finds minimum and maximum values of all features before beginning the algorithm.
    vector<double> relief(const vector<Instance>& instances, int iterCount) {
        auto minMax = getMinAndMax(instances);
        auto minimums = move(minMax.first);
        auto maximums = move(minMax.second);
        int fCount = instances[0].features.size();
        vector<double> weights(fCount, 0);

        for(int i=0; i<iterCount; i++) {
            auto hitMiss = findNearestHitAndMiss(instances, i);
            const Instance& Ri = instances[i];
            const Instance& H = instances[hitMiss.first];
            const Instance& M = instances[hitMiss.second];

            for(int A=0; A<fCount; A++) {
                weights[A] += diff(A, Ri, M, minimums, maximums) / iterCount - diff(A, Ri, H, minimums, maximums) / iterCount;
            }
        }

        return weights;
    }
}

// Functions for IO / string operations
namespace IO {
    // Gets a line from file stream and returns it
    string getLine(ifstream& file) {
        string line;
        getline(file, line);
        return line;
    }

    // Splits a line into vector of numbers
    vector<double> getNumbersFromLine(const string& line) {
        vector<double> numbers;
        double temp;
        stringstream ss(line);
        while(ss >> temp) numbers.push_back(temp);
        return numbers;
    }

    // Reads instance from line and returns it
    Instance getInstanceFromLine(const string& line) {
        auto numbers = getNumbersFromLine(line);
        int cls = (int) numbers.back();
        numbers.pop_back();

        return {move(numbers), cls};
    }

    // Prints results (top feature array) for processor pIndex
    void printResults(int pIndex, const vector<int>& topF) {
        cout << (pIndex==0 ? "Master P0 : " : "Slave P" + to_string(pIndex) + " : ");
        for(int i=0; i<topF.size()-1; i++) {
            cout << topF[i] << ' ';
        }
        cout << topF[topF.size()-1] << endl;
    }
}

// Functions related to MPI interface
// Includes functions for sending/receiving messages and sequentially printing results
namespace MPI {
    void sendSignal(int to) {                               // Sends an empty message with tag 1
        int temp;
        MPI_Send(&temp, 1, MPI_INT, to, 1, MPI_COMM_WORLD);
    }
    void receiveSignal(int from) {                          // Receives an empty message with tag 1
        int temp;
        MPI_Recv(&temp, 1, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    void sendInt(int to, int val) {                         // Sends an integer
        MPI_Send(&val, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
    }
    int receiveInt(int from) {                              // Receives an integer
        int result;
        MPI_Recv(&result, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return result;
    }
    void sendString(int to, const string& str) {            // Sends a string
        MPI_Send(str.c_str(), str.size()+1, MPI_CHAR, to, 0, MPI_COMM_WORLD);
    }
    string receiveString(int from) {                        // Receives a string
        char temp[bufferSize];
        MPI_Recv(temp, bufferSize, MPI_CHAR, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return string(temp);
    }
    void sendIntVec(int to, const vector<int>& vec) {       // Sends a vector<int> (sends its size first)
        sendInt(to, vec.size());
        for(int num : vec) {
            sendInt(to, num);
        }
    }
    vector<int> receiveIntVec(int from) {                   // Receives a vector<int> (receives its size first)
        int size = receiveInt(from);
        vector<int> result(size);
        for(int i=0; i<size; i++) {
            result[i] = receiveInt(from);
        }
        return result;
    }
    void sendStrVec(int to, const vector<string>& vec) {    // Sends a vector<string> (sends its size first)
        sendInt(to, vec.size());
        for(const string& str : vec) {
            sendString(to, str);
        }
    }
    vector<string> receiveStrVec(int from) {                // Receives a vector<string> (receives its size first)
        int size = receiveInt(from);
        vector<string> result(size);
        for(int i=0; i<size; i++) {
            result[i] = receiveString(from);
        }
        return result;
    }

    // Serializes processor prints such that first slaves from 1 to N-1, then master prints their results.
    void printProcessor(int pIndex, int pCount, const vector<int>& topF) {
        if(pIndex == 1) {                               // 1
            IO::printResults(1, topF);
            sendSignal(2);
        } else if(pIndex > 1 && pIndex < pCount-1) {    // 2 .. N-2
            receiveSignal(pIndex-1);
            IO::printResults(pIndex, topF);
            sendSignal(pIndex+1);
        } else if(pIndex == pCount-1) {                 // N-1
            receiveSignal(pIndex-1);
            IO::printResults(pIndex, topF);
            sendSignal(0);
        } else {                                        // 0
            receiveSignal(pCount-1);
            IO::printResults(0, topF);
        }
    }
}

// Function for master processor
// Creates the file stream and reads P, N, A, M, T from the stream.
// Sends each slave their lines, iteration count, and top feature count
// Receives feature arrays from each slave, arranges them in a set, prints results
void masterFunction(const string& filePath, int pCount) {
    ifstream file(filePath);
    IO::getLine(file);  // ignore first line (P), we get pCount from argument instead

    auto numbers = IO::getNumbersFromLine(IO::getLine(file));
    int iCount = numbers[0];
    // int fCount = numbers[1];     // no need to use feature count
    int iterCount = numbers[2];
    int topFCount = numbers[3];
    int linesPerSlave = iCount/(pCount-1);

    for(int i=1; i<pCount; i++) {
        vector<string> lines(linesPerSlave);
        for(int j=0; j<linesPerSlave; j++) {
            lines[j] = IO::getLine(file);
        }
        MPI::sendStrVec(i, lines);
        MPI::sendInt(i, iterCount);
        MPI::sendInt(i, topFCount);
    }

    set<int> topFSet;
    for(int i=1; i<pCount; i++) {
        auto features = MPI::receiveIntVec(i);
        topFSet.insert(features.begin(), features.end());
    }

    MPI::printProcessor(0, pCount, {topFSet.begin(), topFSet.end()});
}

// Function for slave processor
// Gets lines, iteration count, and top feature count
// Creates instances vector, applies relief algorithm, gets top features, prints it, then sends it to master
void slaveFunction(int pIndex, int pCount) {
    auto lines = MPI::receiveStrVec(0);
    int iterCount = MPI::receiveInt(0);
    int topFCount = MPI::receiveInt(0);

    vector<Instance> instances(lines.size());
    for(int i=0; i<instances.size(); i++) {
        instances[i] = IO::getInstanceFromLine(lines[i]);
    }

    int fCount = instances[0].features.size();
    auto weights = Relief::relief(instances, iterCount);

    vector<int> features(fCount);
    for(int i=0; i<fCount; i++) features[i] = i;
    sort(features.begin(), features.end(), [&](int x, int y) {return weights[x]>weights[y];});
    features.resize(topFCount);
    sort(features.begin(), features.end());
    MPI::printProcessor(pIndex, pCount, features);
    MPI::sendIntVec(0, features);
}

// Sets index of the processor and processor count, applies either masterFunction or slaveFunction based on the index
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int pIndex, pCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &pIndex);
    MPI_Comm_size(MPI_COMM_WORLD, &pCount);

    if(pIndex == 0) {
        masterFunction(string(argv[1]), pCount);
    } else {
        slaveFunction(pIndex, pCount);
    }

    MPI_Finalize();
}
