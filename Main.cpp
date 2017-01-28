#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility> 
#include "KNearestNeighbors.h"

using namespace std;

void displayVector (std::vector<double> &v) {
	for (double &feature : v) {
		std::cout << feature << " ";
	}
	std::cout << std::endl;
}

void displayTrainingSet(vector<FeaturedVector> &data) {
	for (FeaturedVector &a : data) {
		displayVector(a);
	}
}

int main(int argc, char const *argv[]) {

	ifstream file;
	string filename = "breast-cancer.data";

	file.open(filename, std::ifstream::in);

	vector<FeaturedVector> data;

	while (file.good()) {
		string line;
		getline(file, line);
		istringstream buffer(line);

		double feature;
		FeaturedVector v;

		while (buffer >> feature) {
			if (buffer.peek() == ',') {
				buffer.ignore();
			}
			v.push_back(feature);
		}

		if (!v.empty()) {
			FeaturedVector v1(&v[1], &v[v.size()]);
			data.push_back(v1);
		}
	}

	file.close();

	vector<FeaturedVector> trainingSet;
	vector<FeaturedVector> testSet;

	unsigned len = 0.7 * data.size();

	for (unsigned i = 0; i < len; ++i) {
		trainingSet.push_back(data[i]);
	}

	for (unsigned i = len; i < data.size(); ++i) {
		testSet.push_back(data[i]);
	}

	KNearestNeighbors knn(trainingSet);
	int acc = 0;

	for (unsigned i = 0; i < testSet.size(); ++i) {
		FeaturedVector &p = testSet[i];
		FeaturedVector v(&p[0], &p[p.size() - 1]);
		double predict = knn.predict(v);
		if (predict == p.back())
			acc++;
	}

	cout << "KNN ACCURACY: " << acc / (double) testSet.size() << endl;

	return 0;
}
