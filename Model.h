#pragma once

#include <memory>
#include <tuple>

namespace netn {

	struct Component;

	class IModel {
	public:
		virtual ~IModel() = default;
	};

	template <typename T>
	class Model : public IModel {
	public:
		typedef T value_t;

		virtual ~Model() = default;

		virtual value_t eval() const = 0;
		virtual value_t derivPart(const Component & component) const = 0;
		virtual std::shared_ptr<Model<T>> toModel() const = 0;

		// Calcul de gradients

		template <typename Var_T>
		Var_T computeGradient(const Var<Var_T> & var);
	};

	template <typename T, typename... Vars>
	std::tuple<Vars...> computeGradients(const Model<T> & model, Vars... vars);
}

#include "Model.inl"