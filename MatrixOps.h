#pragma once

#include <memory>
#include "armadillo/armadillo"

#include "Model.h"

namespace netn {
	class MatSum : public Model<double> {
	public:
		MatSum(const MatSum & other);
		MatSum(const Model<arma::mat> & model);

		double eval() const override;
		double derivPart(const Component & component) const override;
		
		std::shared_ptr<Model<double>> toModel() const override;
	private:
		std::shared_ptr<Model<arma::mat>> _matrix;
	};

	MatSum sum(const Model<arma::mat> & model) {
		return MatSum(model);
	}
}

#include "MatrixOps.inl"