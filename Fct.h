#pragma once

#include <memory>
#include <functional>
#include <math.h>

namespace netn {

	template <typename T> class Model;
	struct Component;
	typedef std::function<double(double)> func_t;

	template <typename T>
	class Fct : public Model<T> {
	public:
		Fct(const Model<T> & var, const func_t & func, const func_t & deriv);
		Fct(const Fct & other);
		virtual ~Fct() = default;

		T eval() const override;
		T derivPart(const Component & component) const override;
		std::shared_ptr<Model<T>> toModel() const override;
	private:
		func_t _func;
		func_t _deriv;
		std::shared_ptr<Model<T>> _var;
	};

	template <typename T>
	Fct<T> sin(const Model<T> & model) {
		return Fct<T>(model,
			[](double x) {return std::sin(x); },
			[](double x) {return std::cos(x); });
	}

	template <typename T>
	Fct<T> cos(const Model<T> & model) {
		return Fct<T>(model,
			[](double x) {return std::cos(x); },
			[](double x) {return - std::sin(x); });
	}

	template <typename T>
	Fct<T> exp(const Model<T> & model) {
		return Fct<T>(model,
			[](double x) {return std::exp(x); },
			[](double x) {return std::exp(x); });
	}

	template <typename T>
	Fct<T> log(const Model<T> & model) {
		return Fct<T>(model,
			[](double x) {return std::log(x); },
			[](double x) {return 1 / x; });
	}

	template <typename T>
	Fct<T> pow(const Model<T> & model, double exp) {
		return Fct<T>(model,
			[exp](double x) {return std::pow(x, exp); },
			[exp](double x) {return exp * std::pow(x, exp - 1); });
	}

	template <typename T>
	Fct<T> sigmoid(const Model<T> & model) {
		return Fct<T>(model,
			[](double x) {return 1 / (1 + std::exp(-x)); },
			[](double x) {double ex = std::exp(x); return ex / ((1 + ex) * (1 + ex)); });
	}
}

#include "Fct.inl"