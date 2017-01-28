#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <assert.h>

typedef std::vector<double> FeaturedVector;

class KNearestNeighbors {

public:

	KNearestNeighbors() {};
	KNearestNeighbors(std::string filename);
	KNearestNeighbors(std::vector<FeaturedVector> &vector);
	void loadInputData(std::string filename);
	double predict(FeaturedVector &v, unsigned k = 5);
	std::vector<FeaturedVector> trainingSet;

private:
	void displayTrainingSet() const;
	void displayVector (const std::vector<double> &v) const;
	double euclidianDistance(FeaturedVector &v1, FeaturedVector &v2);
	double mostCommonElement(std::vector<double> v);
		
};

#endif // K_NEAREST_NEIGHBORS_H
