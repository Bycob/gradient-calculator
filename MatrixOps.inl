#pragma once

#include "MatrixOps.h"

namespace netn {
	MatSum::MatSum(const MatSum & other) 
		: _matrix(other._matrix) {}

	inline MatSum::MatSum(const Model<arma::mat> & model) 
		: _matrix(model.toModel()) {}
	
	inline double netn::MatSum::eval() const {
		auto matrix = _matrix->eval();
		return arma::accu(matrix);
	}

	inline double netn::MatSum::derivPart(const Component & component) const {
		auto deriv = _matrix->derivPart(component);
		return arma::accu(deriv);
	}

	inline std::shared_ptr<Model<double>> MatSum::toModel() const {
		return std::make_shared<MatSum>(*this);
	}
}