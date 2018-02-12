#include "armadillo/armadillo"

#include "Model.h"
#include "Var.h"

namespace netn {

	template<typename T>
	inline Fct<T>::Fct(const Model<T> & var, const func_t & func, const func_t & deriv)
		: _var(var.toModel()), _func(func), _deriv(deriv) {}

	template<typename T>
	inline Fct<T>::Fct(const Fct & other)
		: _var(other._var), _func(other._func), _deriv(other._deriv) {}
	
	template<typename T>
	inline T Fct<T>::eval() const {
		return _func(_var->eval());
	}

	template<>
	inline arma::mat Fct<arma::mat>::eval() const {
		arma::mat result(_var->eval());
		result.for_each([&](arma::mat::elem_type & elem) {elem = _func(elem); });
		return result;
	}

	template<typename T>
	inline T Fct<T>::derivPart(const Component & component) const {
		return _var->derivPart(component) * _deriv(_var->eval());
	}

	template<>
	inline arma::mat Fct<arma::mat>::derivPart(const Component & component) const {
		arma::mat result(_var->eval());
		arma::mat deriv(_var->derivPart(component));

		for (int i = 0; i < result.size(); i++) {
			result.at(i) = deriv.at(i) * _deriv(result.at(i));
		}
		return result;
	}

	template<typename T>
	inline std::shared_ptr<Model<T>> Fct<T>::toModel() const {
		return std::make_shared<Fct<T>>(*this);
	}
}