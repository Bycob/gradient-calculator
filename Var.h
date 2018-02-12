#pragma once

#include <memory>

#include "armadillo/armadillo"

namespace netn {

	template <typename T> class Model;
	struct Component;

	class IVectorizable {
	public:
		virtual ~IVectorizable() = default;

		virtual int dimension() const = 0;

		Component component(int id) const;
	};

	struct Component {
		Component(const Component & other) : v(other.v), id(other.id) {}
		Component(const IVectorizable & v, int id) : v(v), id(id) {}

		const IVectorizable & v;
		int id;
	};

	template <typename T>
	class Var : public Model<T>, public IVectorizable {
	public:
		template <typename... Args>
		Var(Args... args) : _value(std::make_shared<T>(args...)) {}
		Var(const Var & other) : _value(other._value) {}
		virtual ~Var() = default;

		int dimension() const override { return 1; }

		T createEmptyCopy() const;
		void setElementOfCopy(int i, T & emptyCopy, double value) const;

		T & operator*() { return *_value; }
		const T & operator*() const { return *_value; }
		const bool operator==(const IVectorizable & other) const;

		T eval() const override { return *_value; }
		T derivPart(const Component & component) const override;

		std::shared_ptr<Model<T>> toModel() const override;
	private:
		std::shared_ptr<T> _value;
	};

	typedef Var<arma::mat> Matrix;
	typedef Var<double> Scalar;
}

#include "Model.h"

namespace netn {
	inline Component IVectorizable::component(int id) const {
		return Component(*this, id);
	}

	template <>
	inline int Var<arma::mat>::dimension() const {
		return _value->size();
	}

	template <typename T> 
	inline T Var<T>::createEmptyCopy() const {
		return *_value - *_value;
	}

	template <typename T>
	inline void Var<T>::setElementOfCopy(int i, T & copy, double value) const {
		copy = value;
	}
	
	template <>
	inline void Var<arma::mat>::setElementOfCopy(int i, arma::mat & copy, double value) const {
		copy.at(i) = value;
	}

	template<typename T>
	inline const bool Var<T>::operator==(const IVectorizable & other) const {
		try {
			const Var<T> & var = dynamic_cast<const Var<T> &>(other);
			return var._value == _value;
		}
		catch (const std::bad_cast &) {
			return false;
		}
	}

	template <typename T>
	inline T Var<T>::derivPart(const Component & component) const {
		if (*this == component.v) {
			return 1;
		}
		else {
			return 0;
		}
	}

	template <>
	inline arma::mat Var<arma::mat>::derivPart(const Component & component) const {
		arma::mat zeros(_value->n_rows, _value->n_cols, arma::fill::zeros);

		if (*this == component.v) {
			zeros.at(component.id) = 1;
			return zeros;
		}
		else {
			return zeros;
		}
	}

	template <typename T>
	inline std::shared_ptr<Model<T>> Var<T>::toModel() const {
		return std::make_shared<Var<T>>(*this);
	}

}
