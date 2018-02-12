#pragma once

#include <type_traits>
#include <memory>

#include "Type_Ops.h"

namespace netn {

	template <typename T> class Model;
	struct Component;

	template <typename L, typename R>
	struct AddFct {
		typedef typename add_t<L, R>::type result_t;

		static result_t eval(const L & lhs, const R & rhs) {
			return lhs + rhs;
		}

		static result_t derivPart(const L & lhs, const R & rhs, const L & dlhs, const R & drhs) {
			return dlhs + drhs;
		}
	};

	template <typename L, typename R>
	struct SubFct {
		typedef typename sub_t<L, R>::type result_t;

		static result_t eval(const L & lhs, const R & rhs) {
			return lhs - rhs;
		}

		static result_t derivPart(const L & lhs, const R & rhs, const L & dlhs, const R & drhs) {
			return dlhs - drhs;
		}
	};

	template <typename L, typename R>
	struct MulFct {
		typedef typename mul_t<L, R>::type result_t;

		static result_t eval(const L & lhs, const R & rhs) {
			return lhs * rhs;
		}

		static result_t derivPart(const L & lhs, const R & rhs, const L & dlhs, const R & drhs) {
			return lhs * drhs + dlhs * rhs;
		}
	};

	template <typename L, typename R>
	struct DivFct {
		typedef typename div_t<L, R>::type result_t;

		static result_t eval(const L & lhs, const R & rhs) {
			return lhs / rhs;
		}

		static result_t derivPart(const L & lhs, const R & rhs, const L & dlhs, const R & drhs) {
			return (dlhs * rhs - lhs * drhs) / (rhs * rhs);
		}
	};

	template<typename L, typename R, typename Fct>
	class Ops : public Model<typename Fct::result_t> {
	public:
		Ops(const Model<L> & lhs, const Model<R> & rhs);
		Ops(const Ops & other)
			: _lhs(other._lhs), _rhs(other._rhs) {}

		value_t eval() const override;
		value_t derivPart(const Component & component) const override;

		std::shared_ptr<Model<value_t>> toModel() const override;
	private:

		std::shared_ptr<Model<L>> _lhs;
		std::shared_ptr<Model<R>> _rhs;
	};

	template <typename L, typename R> using AddOp = Ops<L, R, AddFct<L, R>>;
	template <typename L, typename R> using SubOp = Ops<L, R, SubFct<L, R>>;
	template <typename L, typename R> using MulOp = Ops<L, R, MulFct<L, R>>;
	template <typename L, typename R> using DivOp = Ops<L, R, DivFct<L, R>>;

	template <typename L, typename R>
	AddOp<L, R> operator+(const Model<L> & lhs, const Model<R> & rhs);

	template <typename L, typename R>
	SubOp<L, R> operator-(const Model<L> & lhs, const Model<R> & rhs);

	template <typename L, typename R>
	MulOp<L, R> operator*(const Model<L> & lhs, const Model<R> & rhs);

	template <typename L, typename R>
	DivOp<L, R> operator/(const Model<L> & lhs, const Model<R> & rhs);

	template <typename T>
	MulOp<double, T> operator*(double lhs, const Model<T> & rhs);
}

#include "Ops.inl"