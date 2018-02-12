
namespace netn {

	template<typename T>
	template<typename Var_T>
	inline Var_T Model<T>::computeGradient(const Var<Var_T> & var) {
		int size = var.dimension();
		Var_T gradient = var.createEmptyCopy();

		for (int i = 0; i < size; i++) {
			T deriv = derivPart({ var, i });
			// si T n'est pas convertible en double, la fonction ne compile pas...
			// TODO il faudrait une erreur plus explicite à la compilation dans ce cas (du genre "invalid type : matrix")
			var.setElementOfCopy(i, gradient, deriv);
		}
		return gradient;
	}

	template <typename T>
	std::tuple<> computeGradients(const Model<T> & model) {
		return std::make_tuple();
	}

	template<typename T, typename Var_T, typename ...Vars>
	std::tuple<Vars...> computeGradients(const Model<T>& model, const Var_T & var, Vars... vars) {
		Var_T gradient = computeGradient(model, var);
		return std::tuple_cat(gradient, computeGradients(model, vars...));
	}
}