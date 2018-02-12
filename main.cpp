#include <iostream>
#include <vector>
#include <random>

#define ARMA_64BIT_WORD  
#include "armadillo/armadillo"

#include "Var.h"
#include "Ops.h"
#include "Fct.h"
#include "MatrixOps.h"

using namespace arma;
using namespace netn;

void xor() {

	int s = 2;

	int X[] = { 0, 0, 0, 1, 1, 1, 1, 0};
	int Y[] = { 0, 1, 0, 1 };

	Matrix W(s, 2, fill::randu);
	Matrix b(s, 1, fill::randu);
	Matrix H(1, s, fill::randu);
	Matrix b2(1, 1, fill::randu);

	Matrix x(2, 1, fill::zeros);
	Scalar y(0);

	arma::mat & _x = *x;
	double & _y = *y;

	auto yr = sum(sigmoid(H * sigmoid(W * x + b) + b2));
	auto model = pow(yr - y, 2);

	double step = 0.01;

	// recherche d'un xor
	for (int i = 0; i < 60000; i++) {

		double total = 0;

		_x(0) = X[0];
		_x(1) = X[1];

		_y = Y[0];

		auto dW = model.computeGradient(W);
		auto db = model.computeGradient(b);
		auto dH = model.computeGradient(H);
		auto db2 = model.computeGradient(b2);

		if (i % 1000 == 0) total += model.eval();

		for (int j = 1; j < 4; j++) {
			_x(0) = X[2 * j];
			_x(1) = X[2 * j + 1];

			_y = Y[j];

			dW += model.computeGradient(W);
			db += model.computeGradient(b);
			dH += model.computeGradient(H);
			db2 += model.computeGradient(b2);

			if (i % 1000 == 0) total += model.eval();
		}

		*W = *W - step * dW;
		*b = *b - step * db;
		*H = *H - step * dH;
		*b2 = *b2 - step * db2;

		if (i % 1000 == 0) std::cout << i << " | " << total << std::endl;
	}

	for (int i = 0; i < 4; i++) {
		_x(0) = (i / 2) % 2;
		_x(1) = ((i + 1) / 2) % 2;
		std::cout << "Résultat final : " << _x(0) << " xor " << _x(1) << " donne " << yr.eval() << " | " << ((int)_x.at(0) != (int)_x.at(1) ? 1 : 0) << std::endl;
	}
}

int main() {
	arma_rng::set_seed_random();
	srand(time(NULL));

	xor();
	// auto gradients = computeGradients(model, W, b);
}