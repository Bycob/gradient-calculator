#include <iostream>

#include "Model.h"
#include "Var.h"

namespace netn {

	template<typename L, typename R, typename Op>
	inline Ops<L, R, Op>::Ops(const Model<L> & lhs, const Model<R> & rhs)
		: _lhs(lhs.toModel()), _rhs(rhs.toModel()) {}

	template<typename L, typename R, typename Op>
	inline std::shared_ptr<Model<typename Ops<L, R, Op>::value_t>> Ops<L, R, Op>::toModel() const {
		return std::make_shared<Ops<L, R, Op>>(*this);
	}

	template<typename L, typename R, typename Fct>
	inline typename Ops<L, R, Fct>::value_t Ops<L, R, Fct>::eval() const {
		return Fct::eval(_lhs->eval(), _rhs->eval());
	}

	template<typename L, typename R, typename Fct>
	inline typename Ops<L, R, Fct>::value_t Ops<L, R, Fct>::derivPart(const Component & component) const {
		return Fct::derivPart(_lhs->eval(), _rhs->eval(), _lhs->derivPart(component), _rhs->derivPart(component));
	}

	template <typename L, typename R>
	inline AddOp<L, R> operator+(const Model<L> & lhs, const Model<R> & rhs) {
		return AddOp<L, R>(lhs, rhs);
	}

	template <typename L, typename R>
	inline SubOp<L, R> operator-(const Model<L> & lhs, const Model<R> & rhs) {
		return SubOp<L, R>(lhs, rhs);
	}

	template <typename L, typename R>
	inline MulOp<L, R> operator*(const Model<L> & lhs, const Model<R> & rhs) {
		return MulOp<L, R>(lhs, rhs);
	}

	template <typename L, typename R>
	inline DivOp<L, R> operator/(const Model<L> & lhs, const Model<R> & rhs) {
		return DivOp<L, R>(lhs, rhs);
	}

	template <typename T>
	inline MulOp<double, T> operator*(double lhs, const Model<T> & rhs) {
		return MulOp<double, T>(Var<double>(lhs), rhs);
	}
}